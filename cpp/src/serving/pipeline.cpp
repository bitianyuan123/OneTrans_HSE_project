#include "serving/pipeline.h"

#include <sstream>
#include <stdexcept>

namespace onetrans {

namespace {
int64_t now_unix() { return static_cast<int64_t>(std::time(nullptr)); }
}

// --------------------------------------------------------------------------- //
// Metrics
// --------------------------------------------------------------------------- //
void Metrics::count(const std::string& name, double v) {
    std::lock_guard<std::mutex> lk(mu_);
    values_[name] += v;
}

void Metrics::gauge(const std::string& name, double v) {
    std::lock_guard<std::mutex> lk(mu_);
    values_[name] = v;
}

double Metrics::get(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = values_.find(name);
    return it == values_.end() ? 0.0 : it->second;
}

Metrics::Timer::Timer(Metrics& m, std::string name) : m_(m), name_(std::move(name)) {
    t0_ = std::chrono::steady_clock::now();
}

Metrics::Timer::~Timer() {
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - t0_)
                  .count();
    m_.count(name_ + "_us_total", static_cast<double>(us));
    m_.count(name_ + "_n", 1.0);
}

std::string Metrics::dump() const {
    std::unordered_map<std::string, double> snap;
    {
        std::lock_guard<std::mutex> lk(mu_);
        snap = values_;
    }
    std::ostringstream os;
    for (const auto& [k, v] : snap) os << k << " " << v << "\n";
    return os.str();
}

// --------------------------------------------------------------------------- //
// NearlineWorker
// --------------------------------------------------------------------------- //
IngestResult NearlineWorker::ingest(const IngestInput& in) {
    IngestResult res;
    try {
        Metrics::Timer timer(metrics_, "nearline.ingest");
        auto [s_emb, s_mask] = frontend_.encode_s(in.item_ids, lookup_);
        UserKV kv = runner_.encode_s(s_emb, s_mask);
        std::string payload = kv_serialize(kv);

        UserKVRecord rec;
        rec.key = KVKey{model_version_, in.user_id};
        rec.s_len = kv.s_len;
        rec.per_layer_len = kv.per_layer_len;
        rec.seq_ts_last = in.timestamps.empty() ? 0 : in.timestamps.back();
        rec.created_at = now_unix();
        rec.payload = std::move(payload);
        res.checksum = rec.checksum();
        res.accepted = store_.put(rec);

        metrics_.count("nearline.events_ingested", static_cast<double>(in.item_ids.size()));
        metrics_.gauge("kv.obj_cnt", static_cast<double>(store_.size()));
    } catch (const std::exception& e) {
        res.accepted = false;
        res.reason = e.what();
    }
    return res;
}

// --------------------------------------------------------------------------- //
// OnlineWorker
// --------------------------------------------------------------------------- //
ScoreOutcome OnlineWorker::score(const ScoreInput& in) {
    Metrics::Timer timer(metrics_, "online.score");
    Tensor ns_emb = frontend_.encode_ns(in, lookup_);
    const int64_t M = ns_emb.shape[0];
    const int64_t T = runner_.model().head_w.shape[0];

    auto rec = store_.get(KVKey{model_version_, in.user_id});
    if (!rec) {
        // KV miss（冷启动/过期/搬迁）：全零 legal logits 降级（对齐 G8 语义）
        metrics_.count("kv.miss");
        ScoreOutcome out;
        out.logits = Tensor::zeros({M, T});
        out.kv_hit = false;
        return out;
    }
    metrics_.count("kv.hit");

    UserKV kv = kv_deserialize(rec->payload);
    ScoreOutcome out;
    out.logits = runner_.score_ns(kv, ns_emb);
    metrics_.count("online.qps");
    metrics_.count("online.candidate_throughput", static_cast<double>(M));
    return out;
}

std::vector<ScoreOutcome> OnlineWorker::score_batch(const std::vector<ScoreInput>& batch) {
    Metrics::Timer timer(metrics_, "online.score_batch");
    if (batch.empty()) return {};

    // 1. 逐请求前处理
    std::vector<Tensor> ns_embs;
    ns_embs.reserve(batch.size());
    for (const auto& in : batch) ns_embs.push_back(frontend_.encode_ns(in, lookup_));

    // 2. 一次 mget
    std::vector<KVKey> keys;
    keys.reserve(batch.size());
    for (const auto& in : batch) keys.push_back(KVKey{model_version_, in.user_id});
    auto recs = store_.mget(keys);

    // 3. 命中请求：K/V 反序列化 + 候选展平；miss 请求记录位置
    std::vector<std::unique_ptr<UserKV>> kvs;
    std::vector<Tensor> hit_embs;
    std::vector<int> hit_of_req(batch.size(), -1);  // 请求 → 命中展平行
    int64_t total = 0;
    for (size_t r = 0; r < batch.size(); ++r) {
        const int64_t M = ns_embs[r].shape[0];
        if (recs[r]) {
            metrics_.count("kv.hit");
            kvs.push_back(std::make_unique<UserKV>(kv_deserialize(recs[r]->payload)));
            hit_embs.push_back(ns_embs[r]);
            hit_of_req[r] = static_cast<int>(hit_embs.size()) - 1;
        } else {
            metrics_.count("kv.miss");
        }
        total += M;
    }

    // 4. 一次 score_ns_batch：每候选一行（同用户重复指向其 KV）
    const int64_t T = runner_.model().head_w.shape[0];
    Tensor hit_logits;
    if (!kvs.empty()) {
        int64_t rows = 0;
        for (const auto& e : hit_embs) rows += e.shape[0];
        Tensor ns_cat({rows, ns_embs[0].shape[1], ns_embs[0].shape[2]});
        std::vector<const UserKV*> kv_rows;
        kv_rows.reserve(static_cast<size_t>(rows));
        int64_t off = 0;
        for (size_t h = 0; h < hit_embs.size(); ++h) {
            const auto& e = hit_embs[h];
            std::copy(e.data.begin(), e.data.end(), ns_cat.data.begin() + off);
            off += e.data.size();
            for (int64_t m = 0; m < e.shape[0]; ++m) kv_rows.push_back(kvs[h].get());
        }
        hit_logits = runner_.score_ns_batch(kv_rows, ns_cat);
    }

    // 5. 回填：miss 全零行，顺序与 batch 一致
    std::vector<ScoreOutcome> out(batch.size());
    for (size_t r = 0; r < batch.size(); ++r) {
        const int64_t M = ns_embs[r].shape[0];
        out[r].kv_hit = hit_of_req[r] >= 0;
        out[r].logits = Tensor::zeros({M, T});
        if (hit_of_req[r] >= 0) {
            int64_t row0 = 0;
            for (int h = 0; h < hit_of_req[r]; ++h) row0 += hit_embs[h].shape[0];
            std::copy(hit_logits.data.begin() + row0 * T, hit_logits.data.begin() + (row0 + M) * T,
                      out[r].logits.data.begin());
        }
    }
    metrics_.count("online.qps", static_cast<double>(batch.size()));
    metrics_.count("online.candidate_throughput", static_cast<double>(total));
    metrics_.gauge("online.batch_size", static_cast<double>(total));
    return out;
}

// --------------------------------------------------------------------------- //
// BatchScheduler
// --------------------------------------------------------------------------- //
void BatchScheduler::submit(ScoreJob job) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(std::move(job));
    }
    cv_.notify_one();
}

std::vector<ScoreJob> BatchScheduler::next_batch() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return !queue_.empty() || shutdown_; });
    if (queue_.empty()) return {};  // shutdown 唤醒
    auto deadline = std::chrono::steady_clock::now() + max_wait_;
    while (queue_.size() < max_batch_) {
        if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) break;
    }
    std::vector<ScoreJob> batch;
    batch.reserve(std::min(max_batch_, queue_.size()));
    while (!queue_.empty() && batch.size() < max_batch_) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();
    }
    return batch;
}

void BatchScheduler::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        shutdown_ = true;
    }
    cv_.notify_all();
}

size_t BatchScheduler::pending() const {
    std::lock_guard<std::mutex> lk(mu_);
    return queue_.size();
}

// --------------------------------------------------------------------------- //
// Dispatcher
// --------------------------------------------------------------------------- //
Dispatcher::Dispatcher(OnlineWorker& worker, BatchScheduler& sched, int num_threads,
                       Metrics& metrics)
    : worker_(worker), sched_(sched), metrics_(metrics), num_threads_(num_threads) {
    if (num_threads_ <= 0) num_threads_ = 1;
}

Dispatcher::~Dispatcher() { stop(); }

std::future<ScoreOutcome> Dispatcher::submit(ScoreInput input) {
    auto promise = std::make_shared<std::promise<ScoreOutcome>>();
    auto fut = promise->get_future();
    sched_.submit(ScoreJob{std::move(input), promise});
    return fut;
}

void Dispatcher::start() {
    if (running_.exchange(true)) return;
    for (int i = 0; i < num_threads_; ++i) threads_.emplace_back([this] { worker_loop(); });
}

void Dispatcher::stop() {
    if (!running_.exchange(false)) return;
    sched_.shutdown();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
}

void Dispatcher::worker_loop() {
    while (running_) {
        auto batch = sched_.next_batch();
        if (batch.empty()) break;  // shutdown
        std::vector<ScoreInput> inputs;
        inputs.reserve(batch.size());
        for (auto& job : batch) inputs.push_back(std::move(job.input));
        try {
            auto outcomes = worker_.score_batch(inputs);
            for (size_t i = 0; i < batch.size(); ++i)
                batch[i].promise->set_value(std::move(outcomes[i]));
        } catch (const std::exception& e) {
            for (auto& job : batch)
                job.promise->set_exception(std::make_exception_ptr(std::runtime_error(e.what())));
        }
    }
}

}  // namespace onetrans
