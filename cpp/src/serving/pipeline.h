// Nearline/Online 编排（两阶段服务 worker）+ 攒批调度。
//
// 与 onetrans/serving/pipeline.py 对齐：
// - NearlineWorker：Stage I，编码用户历史并写入 User KV（全量 prefill + put 幂等基线）；
// - OnlineWorker：Stage II，读 User KV 并对候选交叉打分；KV miss 降级为全零 logits；
// - BatchScheduler：FIFO 攒批，满批或超时触发批量打分（时延有界）；
// - Dispatcher：N 个打分线程消费批次，future 化异步提交。
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/tensor.h"
#include "engine/frontend.h"
#include "engine/two_stage.h"
#include "kv/serialize.h"
#include "kv/store.h"
#include "serving/embed_lookup.h"

namespace onetrans {

// --------------------------------------------------------------------------- //
// 指标（counter/gauge，原子锁保护；dump 为 Prometheus 风格文本）
// --------------------------------------------------------------------------- //
class Metrics {
public:
    void count(const std::string& name, double v = 1.0);
    void gauge(const std::string& name, double v);
    double get(const std::string& name) const;  // 缺失返回 0

    // 析构计时辅助：累计 {name}_us_total 与 {name}_n
    class Timer {
    public:
        Timer(Metrics& m, std::string name);
        ~Timer();

    private:
        Metrics& m_;
        std::string name_;
        std::chrono::steady_clock::time_point t0_;
    };

    std::string dump() const;

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, double> values_;
};

// --------------------------------------------------------------------------- //
// Nearline：ingest → encode_s → put(UserKV)
// --------------------------------------------------------------------------- //
struct IngestResult {
    bool accepted = false;
    std::string checksum;
    std::string reason;
};

class NearlineWorker {
public:
    NearlineWorker(const EmbeddingFrontend& frontend, const TwoStageRunner& runner,
                   LocalKVStore& store, Metrics& metrics, LookupFn lookup,
                   std::string model_version)
        : frontend_(frontend),
          runner_(runner),
          store_(store),
          metrics_(metrics),
          lookup_(std::move(lookup)),
          model_version_(std::move(model_version)) {}

    // item_ids 时间升序（最旧在前、最新在尾）；重复 ingest 幂等覆盖
    IngestResult ingest(const IngestInput& in);

private:
    const EmbeddingFrontend& frontend_;
    const TwoStageRunner& runner_;
    LocalKVStore& store_;
    Metrics& metrics_;
    LookupFn lookup_;
    std::string model_version_;
};

// --------------------------------------------------------------------------- //
// Online：score → encode_ns → get(UserKV) → score_ns
// --------------------------------------------------------------------------- //
struct ScoreOutcome {
    Tensor logits;  // [M, T]
    bool kv_hit = true;
};

class OnlineWorker {
public:
    OnlineWorker(const EmbeddingFrontend& frontend, const TwoStageRunner& runner,
                 LocalKVStore& store, Metrics& metrics, LookupFn lookup,
                 std::string model_version)
        : frontend_(frontend),
          runner_(runner),
          store_(store),
          metrics_(metrics),
          lookup_(std::move(lookup)),
          model_version_(std::move(model_version)) {}

    ScoreOutcome score(const ScoreInput& in);

    // 批量：一次 mget + 一次 score_ns_batch；miss 请求降级全零行，顺序保持
    std::vector<ScoreOutcome> score_batch(const std::vector<ScoreInput>& batch);

private:
    const EmbeddingFrontend& frontend_;
    const TwoStageRunner& runner_;
    LocalKVStore& store_;
    Metrics& metrics_;
    LookupFn lookup_;
    std::string model_version_;
};

// --------------------------------------------------------------------------- //
// 攒批调度器（满批或超时返回 ≥1 条）
// --------------------------------------------------------------------------- //
struct ScoreJob {
    ScoreInput input;
    std::shared_ptr<std::promise<ScoreOutcome>> promise;
};

class BatchScheduler {
public:
    BatchScheduler(size_t max_batch_size, std::chrono::milliseconds max_wait)
        : max_batch_(max_batch_size), max_wait_(max_wait) {}

    void submit(ScoreJob job);

    // 阻塞取一个批次（≥1 条；满批或超时返回）。shutdown 后队列空时返回空批次。
    std::vector<ScoreJob> next_batch();

    void shutdown();
    size_t pending() const;

private:
    size_t max_batch_;
    std::chrono::milliseconds max_wait_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<ScoreJob> queue_;
    bool shutdown_ = false;
};

// --------------------------------------------------------------------------- //
// Dispatcher：打分线程池消费批次
// --------------------------------------------------------------------------- //
class Dispatcher {
public:
    Dispatcher(OnlineWorker& worker, BatchScheduler& sched, int num_threads, Metrics& metrics);
    ~Dispatcher();

    std::future<ScoreOutcome> submit(ScoreInput input);

    void start();
    void stop();

private:
    void worker_loop();

    OnlineWorker& worker_;
    BatchScheduler& sched_;
    Metrics& metrics_;
    int num_threads_;
    std::vector<std::thread> threads_;
    std::atomic<bool> running_{false};
};

}  // namespace onetrans
