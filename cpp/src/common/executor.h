// SEDA 功能分池执行器（§7.4.2）——基于 folly 线程池，不再维护自研 std::thread 池。
//
// 引入 folly（CPUThreadPoolExecutor / IOThreadPoolExecutor）承接任务调度：
// - IO 池（EmbedLookup / KV I/O）→ folly::IOThreadPoolExecutor，承接阻塞式网络/外部调用；
// - CPU 池（Frontend Encode / Compute）→ folly::CPUThreadPoolExecutor（有界队列），承接计算；
// - Executor                 ：统一 add/stop 句柄，屏蔽两类 folly 执行器的差异；
// - 有界背压（ExecutorOverloaded）：add 前以 getPendingTaskCount() 与容量快照判定，
//   超限快速失败，信号逐级上传至接入层，防止单池无界堆积拖垮整体时延（§7.4.7 SEDA 契约）。
#pragma once

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/IOThreadPoolExecutor.h>

#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace onetrans {

// 队列满：上游应快速拒绝（HTTP 429/503）或降级，不得无界排队
struct ExecutorOverloaded : std::runtime_error {
    explicit ExecutorOverloaded(const std::string& name)
        : std::runtime_error("executor '" + name + "' queue full") {}
};

// folly 执行器句柄：统一 add/stop 接口 + 容量背压（快照式软判定）。
class Executor {
public:
    // io=true → IOThreadPoolExecutor；否则 CPUThreadPoolExecutor
    Executor(std::string name, int num_threads, size_t queue_capacity, bool io);
    ~Executor();

    // 队列满抛 ExecutorOverloaded（快速失败）；已 stop 抛 logic_error
    void add(std::function<void()> fn);

    // 停机：先置 stopped_ 拒收，再 join 排空在途任务并回收线程
    void stop();

    const std::string& name() const { return name_; }
    bool is_io() const { return io_; }

private:
    std::string name_;
    size_t cap_;
    bool io_;
    bool stopped_ = false;
    std::mutex mu_;  // 保护 stopped_ 与容量判定（软背压，非强一致）

    std::shared_ptr<folly::CPUThreadPoolExecutor> cpu_;
    std::shared_ptr<folly::IOThreadPoolExecutor> io_exec_;
};

// 功能分池集合（§7.4.2 表格的实例化；持有并统一停机）
class ExecutorSet {
public:
    std::shared_ptr<Executor> make(const std::string& name, int num_threads,
                                   size_t queue_capacity, bool io);
    void stop_all();

private:
    std::vector<std::shared_ptr<Executor>> pools_;
    std::mutex mu_;
};

}  // namespace onetrans