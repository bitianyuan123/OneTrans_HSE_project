// SEDA 分阶段在线打分流水线（§7.4.3）。
//
// 阶段链（每阶段独立线程池，有界队列背压）：
//   EmbedLookup Pool → Frontend Encode Pool → KV I/O Pool → 攒批线程
//   → PythonComputeBridge（持 GIL 的 PyTorch 前向）或 Compute Pool（C++ CPU 降级）
//
// GIL 边界：除最终计算批投递给桥线程外，全部阶段在纯 C++ 线程池上运行
// （docs/detailed_design.md §7.4.2 线程池清单 / §7.4.9 锁分析）。
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/executor.h"
#include "engine/frontend.h"
#include "engine/two_stage.h"
#include "kv/serialize.h"
#include "kv/store.h"
#include "serving/compute_bridge.h"
#include "serving/pipeline.h"  // Metrics / ScoreOutcome / ScoreInput

namespace onetrans {

class ScoreFlow {
public:
    struct Config {
        int lookup_threads = 4;    // EmbedLookup Pool（IO）
        int encode_threads = 2;    // Frontend Encode Pool（CPU）
        int kv_threads = 4;        // KV I/O Pool（IO）
        int compute_threads = 0;   // Compute Pool（C++ 降级路径；≤0 → 硬件核数）
        size_t queue_cap = 1024;   // 各池入口有界队列容量
        size_t max_batch = 16;     // 攒批上限（请求数）
        int batch_wait_ms = 5;     // 攒批窗口
    };

    // bridge 可为 nullptr（强制 C++ 路径）
    ScoreFlow(const EmbeddingFrontend& frontend, const TwoStageRunner& runner,
              KVStore& store, const LookupFn& lookup, Metrics& metrics,
              PythonComputeBridge* bridge, std::string model_version, Config cfg);
    ~ScoreFlow();

    void start();
    void stop();

    // 回调式提交（生产路径）：完成线程直接调用 done
    // （err 非空 = 流水线失败，outcome 无效）。done 保证恰好调用一次。
    using ScoreDone = std::function<void(ScoreOutcome out, const std::string& err)>;
    void submit(ScoreInput input, ScoreDone done);

    // future 便利版（测试/工具）：内部包装回调
    std::future<ScoreOutcome> submit(ScoreInput input);

    // 当前实际生效的计算后端（观测 / healthz）
    std::string backend() const;
    size_t inflight() const { return inflight_.load(std::memory_order_relaxed); }

private:
    struct Ctx {  // 单请求贯穿流水线的共享状态（阶段间 move 传递所有权）
        ScoreInput input;
        ScoreDone done;                          // 完成回调（恰好一次）
        NsEmbeddings emb;                         // lookup 阶段产出
        Tensor ns_emb;                           // encode 阶段产出 [M, Ns, D]
        std::shared_ptr<const UserKVRecord> rec;  // kv 阶段产出（miss 为 null）
        int32_t kv_idx = -1;                      // 批内 KV payload 去重下标
    };
    using CtxPtr = std::shared_ptr<Ctx>;

    // 阶段（各自 add 到对应池；满时快速失败回填 503 语义异常）
    void stage_lookup(const CtxPtr& ctx);
    void stage_encode(const CtxPtr& ctx);
    void stage_kv(const CtxPtr& ctx, Tensor ns_emb);

    // 攒批线程主循环
    void batch_loop();
    std::vector<CtxPtr> take_batch();
    // 出批处理：拼 ns_blob + KV 去重 + 槽位表 → bridge / compute pool
    void on_batch(std::vector<CtxPtr> jobs);
    // C++ CPU 降级路径（在 Compute Pool 上执行 score_ns_batch）
    void dispatch_cpp(std::vector<CtxPtr> hits);
    // 回填：logits blob fp32 [rows, T] 按槽位切分（miss 已在 on_batch 前置回填）
    void fill_outcomes(const std::vector<CtxPtr>& hits, const std::vector<int64_t>& row0s,
                       const std::string& logits_blob, int64_t T);
    void fail_ctx(const CtxPtr& ctx, const std::string& err);

    const EmbeddingFrontend& frontend_;
    const TwoStageRunner& runner_;
    KVStore& store_;
    LookupFn lookup_;
    Metrics& metrics_;
    PythonComputeBridge* bridge_;
    std::string model_version_;
    Config cfg_;

    ExecutorSet executors_;
    std::shared_ptr<Executor> lookup_pool_;
    std::shared_ptr<Executor> encode_pool_;
    std::shared_ptr<Executor> kv_pool_;
    std::shared_ptr<Executor> compute_pool_;

    // 攒批队列（独立于各池：批线程单消费者）
    std::mutex batch_mu_;
    std::condition_variable batch_cv_;
    std::deque<CtxPtr> batch_q_;
    bool batch_stop_ = false;
    std::thread batch_thread_;

    std::atomic<size_t> inflight_{0};
    std::atomic<bool> running_{false};
};

}  // namespace onetrans
