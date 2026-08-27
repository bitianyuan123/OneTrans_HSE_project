// Python 计算桥（§7.4.5）：嵌入式解释器 + 专用线程持 GIL 下发 PyTorch 算子。
//
// 边界契约（与 docs/detailed_design.md §7.4 一致）：
// - C++ 编排（lookup/encode/KV mget/攒批/回填）全部在 folly 语义线程池上运行，
//   不触碰 GIL；
// - 唯一持 GIL 的是本桥的专用线程：从有界队列取批 → PyGILState_Ensure →
//   调 bridge_score.score_batch（PyTorch CUDA/CPU 前向）→ 回填回调 → Release；
// - 队列满 / 解释器不可用 / torch 缺失 → available()==false 或 submit 返回
//   false，上层降级为 C++ CPU score_ns_batch（数值等价，性能降级）；
// - 多 GPU 扩展：每 GPU 一个桥实例（互不共享队列）。
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace onetrans {

// 一个待计算批（C++ 侧已完成全部前处理）
struct BridgeBatch {
    std::string ns_blob;                     // fp32 LE [n_rows, ns, d]
    std::vector<std::string> kv_payloads;     // 去重后的 UserKV payload（C++ kv_serialize）
    std::vector<int32_t> row_kv_idx;          // 每行 → payload 下标（长度 n_rows）
    int64_t n_rows = 0, ns = 0, d_model = 0;

    // 回调（桥线程调用；on_ok 的 blob 为 fp32 LE [n_rows, T]）
    std::function<void(std::string logits_blob)> on_ok;
    std::function<void(const std::string& err)> on_fail;
};

class PythonComputeBridge {
public:
    // module_dir：bridge_score.py 所在目录（sys.path 注入）
    // weights_dir：weights.bin/manifest.json 所在目录
    // init_timeout_s：import torch + 权重上载的最长等待（超时视为不可用，降级）
    bool start(const std::string& module_dir, const std::string& weights_dir,
               int init_timeout_s = 120);
    void stop();

    bool available() const { return ready_.load(std::memory_order_acquire); }
    const std::string& last_error() const { return last_error_; }

    // 投递一批（队列满返回 false → 调用方降级）
    bool submit(BridgeBatch batch);

    size_t queued() const;

private:
    void run();
    bool init_python(const std::string& module_dir, const std::string& weights_dir);
    // 持 GIL 执行一次前向；返回 logits blob，失败抛异常
    std::string call_score(BridgeBatch& b);

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<BridgeBatch> q_;
    size_t cap_ = 16;
    bool stopped_ = true;
    std::atomic<bool> ready_{false};
    std::string last_error_;
    std::thread thread_;
    // Python 对象句柄（实现细节，void* 隔离 Python.h 泄漏到头文件）
    void* py_module_ = nullptr;
    void* py_score_fn_ = nullptr;
    void* py_meta_fn_ = nullptr;
    // 首轮 init 后已 PyEval_SaveThread（重试时需走 Ensure/Release 而非再 Save）
    bool py_thread_saved_ = false;
};

}  // namespace onetrans
