#include "serving/flow.h"

#include <cstring>
#include <unordered_map>

namespace onetrans {

ScoreFlow::ScoreFlow(const EmbeddingFrontend& frontend, const TwoStageRunner& runner,
                     LocalKVStore& store, const LookupFn& lookup, Metrics& metrics,
                     PythonComputeBridge* bridge, std::string model_version, Config cfg)
    : frontend_(frontend),
      runner_(runner),
      store_(store),
      lookup_(lookup),
      metrics_(metrics),
      bridge_(bridge),
      model_version_(std::move(model_version)),
      cfg_(cfg) {
    if (cfg_.compute_threads <= 0) {
        unsigned int hw = std::thread::hardware_concurrency();
        cfg_.compute_threads = hw > 0 ? static_cast<int>(hw) : 4;
    }
    lookup_pool_ = executors_.make("embed_lookup", cfg_.lookup_threads, cfg_.queue_cap);
    encode_pool_ = executors_.make("frontend_encode", cfg_.encode_threads, cfg_.queue_cap);
    kv_pool_ = executors_.make("kv_io", cfg_.kv_threads, cfg_.queue_cap);
    compute_pool_ = executors_.make("compute_cpp", cfg_.compute_threads, cfg_.queue_cap);
}

ScoreFlow::~ScoreFlow() { stop(); }

void ScoreFlow::start() {
    if (running_.exchange(true)) return;
    batch_thread_ = std::thread([this] { batch_loop(); });
}

void ScoreFlow::stop() {
    if (!running_.exchange(false)) return;
    {
        std::lock_guard<std::mutex> lk(batch_mu_);
        batch_stop_ = true;
    }
    batch_cv_.notify_all();
    if (batch_thread_.joinable()) batch_thread_.join();
    executors_.stop_all();
}

std::string ScoreFlow::backend() const {
    return (bridge_ && bridge_->available()) ? "python" : "cpp";
}

void ScoreFlow::submit(ScoreInput input, ScoreDone done) {
    auto ctx = std::make_shared<Ctx>();
    ctx->input = std::move(input);
    ctx->done = std::move(done);
    inflight_.fetch_add(1, std::memory_order_relaxed);
    stage_lookup(ctx);
}

std::future<ScoreOutcome> ScoreFlow::submit(ScoreInput input) {
    auto promise = std::make_shared<std::promise<ScoreOutcome>>();
    auto fut = promise->get_future();
    submit(std::move(input), [promise](ScoreOutcome out, const std::string& err) {
        if (err.empty())
            promise->set_value(std::move(out));
        else
            promise->set_exception(
                std::make_exception_ptr(std::runtime_error("flow: " + err)));
    });
    return fut;
}

void ScoreFlow::fail_ctx(const CtxPtr& ctx, const std::string& err) {
    if (ctx->done) ctx->done(ScoreOutcome{}, err);
    inflight_.fetch_sub(1, std::memory_order_relaxed);
}

// ---- 阶段 1：EmbedLookup Pool（IO）——查表 + mean-bag ---------------------- //
void ScoreFlow::stage_lookup(const CtxPtr& ctx) {
    try {
        lookup_pool_->add([this, ctx] {
            try {
                {
                    Metrics::Timer t(metrics_, "flow.stage_lookup");
                    ctx->emb = frontend_.lookup_ns(ctx->input, lookup_);
                }
                stage_encode(ctx);
            } catch (const std::exception& e) {
                fail_ctx(ctx, e.what());
            }
        });
    } catch (const ExecutorOverloaded& e) {
        metrics_.count("flow.overload_lookup");
        fail_ctx(ctx, e.what());
    }
}

// ---- 阶段 2：Frontend Encode Pool（CPU）——piecewise + MLP + RMSNorm -------- //
void ScoreFlow::stage_encode(const CtxPtr& ctx) {
    try {
        encode_pool_->add([this, ctx] {
            try {
                {
                    Metrics::Timer t(metrics_, "flow.stage_encode");
                    ctx->ns_emb = frontend_.encode_ns_with(ctx->input, ctx->emb);
                    ctx->emb = NsEmbeddings{};  // 释放查表中间态
                }
                stage_kv(ctx, std::move(ctx->ns_emb));
            } catch (const std::exception& e) {
                fail_ctx(ctx, e.what());
            }
        });
    } catch (const ExecutorOverloaded& e) {
        metrics_.count("flow.overload_encode");
        fail_ctx(ctx, e.what());
    }
}

// ---- 阶段 3：KV I/O Pool（IO）——mget + 攒批入队 ----------------------------- //
void ScoreFlow::stage_kv(const CtxPtr& ctx, Tensor ns_emb) {
    try {
        kv_pool_->add([this, ctx, ns_emb = std::move(ns_emb)]() mutable {
            try {
                {
                    Metrics::Timer t(metrics_, "flow.stage_kv");
                    ctx->rec = store_.get(KVKey{model_version_, ctx->input.user_id});
                }
                ctx->ns_emb = std::move(ns_emb);
                {
                    std::unique_lock<std::mutex> lk(batch_mu_);
                    if (batch_stop_) {
                        lk.unlock();
                        fail_ctx(ctx, "shutting down");
                        return;
                    }
                    batch_q_.push_back(ctx);
                }
                batch_cv_.notify_one();
            } catch (const std::exception& e) {
                fail_ctx(ctx, e.what());
            }
        });
    } catch (const ExecutorOverloaded& e) {
        metrics_.count("flow.overload_kv");
        fail_ctx(ctx, e.what());
    }
}

// ---- 攒批线程：等首条（无限期）→ max_wait 窗口 → 满 max_batch 出批 ---------- //
void ScoreFlow::batch_loop() {
    while (running_) {
        auto jobs = take_batch();
        if (jobs.empty()) break;  // stop 唤醒
        on_batch(std::move(jobs));
    }
}

std::vector<ScoreFlow::CtxPtr> ScoreFlow::take_batch() {
    std::unique_lock<std::mutex> lk(batch_mu_);
    batch_cv_.wait(lk, [&] { return !batch_q_.empty() || batch_stop_; });
    if (batch_q_.empty()) return {};
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(cfg_.batch_wait_ms);
    while (batch_q_.size() < cfg_.max_batch) {
        if (batch_cv_.wait_until(lk, deadline) == std::cv_status::timeout) break;
    }
    std::vector<CtxPtr> batch;
    batch.reserve(std::min(cfg_.max_batch, batch_q_.size()));
    while (!batch_q_.empty() && batch.size() < cfg_.max_batch) {
        batch.push_back(std::move(batch_q_.front()));
        batch_q_.pop_front();
    }
    return batch;
}

void ScoreFlow::on_batch(std::vector<CtxPtr> jobs) {
    Metrics::Timer t(metrics_, "flow.batch");
    const int64_t T = runner_.model().head_w.shape[0];

    // 1. miss 前置回填（不进计算批，§7.4.8：miss 行无需前向）
    std::vector<CtxPtr> hits;
    hits.reserve(jobs.size());
    for (auto& ctx : jobs) {
        if (ctx->rec) {
            metrics_.count("kv.hit");
            hits.push_back(std::move(ctx));
        } else {
            metrics_.count("kv.miss");
            ScoreOutcome out;
            out.kv_hit = false;
            out.logits = Tensor::zeros(
                {static_cast<int64_t>(ctx->input.candidates.size()), T});
            if (ctx->done) ctx->done(std::move(out), "");
            inflight_.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    if (hits.empty()) return;

    // 2. 拼接命中行：ns_blob fp32 [rows, Ns, D] + KV payload 去重 + row→payload
    int64_t rows = 0;
    for (auto& c : hits) rows += c->ns_emb.shape[0];
    const int64_t Ns = hits[0]->ns_emb.shape[1];
    const int64_t D = hits[0]->ns_emb.shape[2];

    BridgeBatch bb;
    bb.n_rows = rows;
    bb.ns = Ns;
    bb.d_model = D;
    bb.ns_blob.resize(static_cast<size_t>(rows * Ns * D * sizeof(float)));

    std::unordered_map<std::string, int32_t> dedup;  // payload → 下标
    std::vector<int64_t> row0s;                       // 每命中请求在批内的首行
    size_t off = 0;
    int64_t row0 = 0;
    for (auto& c : hits) {
        std::memcpy(bb.ns_blob.data() + off, c->ns_emb.data.data(),
                    c->ns_emb.data.size() * sizeof(float));
        off += c->ns_emb.data.size() * sizeof(float);
        row0s.push_back(row0);
        row0 += c->ns_emb.shape[0];
        auto it = dedup.find(c->rec->payload);
        if (it == dedup.end()) {
            it = dedup.emplace(c->rec->payload, static_cast<int32_t>(bb.kv_payloads.size()))
                     .first;
            bb.kv_payloads.push_back(c->rec->payload);
        }
        for (int64_t m = 0; m < c->ns_emb.shape[0]; ++m) bb.row_kv_idx.push_back(it->second);
        c->kv_idx = it->second;  // C++ 降级路径复用去重结果
    }

    metrics_.count("online.qps", static_cast<double>(jobs.size()));
    metrics_.count("online.candidate_throughput", static_cast<double>(rows));
    metrics_.gauge("online.batch_size", static_cast<double>(rows));

    // 3. 计算后端选择：bridge 可用 → Python（PyTorch 算子下发）；否则 C++ CPU
    if (bridge_ && bridge_->available()) {
        bb.on_ok = [this, hits, row0s, T](std::string blob) {
            fill_outcomes(hits, row0s, blob, T);
        };
        bb.on_fail = [this, hits](const std::string& err) {
            for (auto& c : hits) fail_ctx(c, "bridge: " + err);
        };
        if (bridge_->submit(std::move(bb))) {
            metrics_.count("flow.backend.python");
            return;
        }
        metrics_.count("flow.bridge_overflow");  // 桥队列满 → 降级 C++
    }
    (void)row0s;  // C++ 路径自行重算行基（dispatch_cpp 内重建）

    dispatch_cpp(std::move(hits));
}

// C++ CPU 降级：Compute Pool 上一次 score_ns_batch（数值与 Python 等价，golden 对拍）
void ScoreFlow::dispatch_cpp(std::vector<CtxPtr> hits) {
    metrics_.count("flow.backend.cpp");
    const int64_t T = runner_.model().head_w.shape[0];
    compute_pool_->add([this, hits = std::move(hits), T] {
        std::unordered_map<int32_t, std::unique_ptr<UserKV>> kvs;  // kv_idx → UserKV
        std::vector<int64_t> row0s;
        int64_t rows = 0;
        try {
            std::vector<const UserKV*> kv_rows;
            std::vector<Tensor> embs;
            for (auto& c : hits) {
                auto it = kvs.find(c->kv_idx);
                if (it == kvs.end()) {
                    it = kvs.emplace(c->kv_idx,
                                     std::make_unique<UserKV>(kv_deserialize(c->rec->payload)))
                             .first;
                }
                row0s.push_back(rows);
                for (int64_t m = 0; m < c->ns_emb.shape[0]; ++m) kv_rows.push_back(it->second.get());
                embs.push_back(c->ns_emb);
                rows += c->ns_emb.shape[0];
            }
            // 拼接 [rows, Ns, D]
            const int64_t Ns = embs[0].shape[1], D = embs[0].shape[2];
            Tensor ns_cat({rows, Ns, D});
            int64_t off = 0;
            for (auto& e : embs) {
                std::copy(e.data.begin(), e.data.end(), ns_cat.data.begin() + off);
                off += e.data.size();
            }
            Tensor logits;
            {
                Metrics::Timer t(metrics_, "flow.score_cpp");
                logits = runner_.score_ns_batch(kv_rows, ns_cat);
            }
            std::string blob(reinterpret_cast<const char*>(logits.data.data()),
                             logits.data.size() * sizeof(float));
            fill_outcomes(hits, row0s, blob, T);
        } catch (const std::exception& e) {
            for (auto& c : hits) fail_ctx(c, e.what());
        }
    });
}

void ScoreFlow::fill_outcomes(const std::vector<CtxPtr>& hits,
                              const std::vector<int64_t>& row0s, const std::string& logits_blob,
                              int64_t T) {
    const float* lg = reinterpret_cast<const float*>(logits_blob.data());
    for (size_t i = 0; i < hits.size(); ++i) {
        const CtxPtr& c = hits[i];
        const int64_t M = c->ns_emb.shape[0];
        ScoreOutcome out;
        out.kv_hit = true;
        out.logits = Tensor({M, T});
        std::memcpy(out.logits.data.data(), lg + row0s[i] * T,
                    static_cast<size_t>(M * T) * sizeof(float));
        if (c->done) c->done(std::move(out), "");
        inflight_.fetch_sub(1, std::memory_order_relaxed);
    }
}

}  // namespace onetrans
