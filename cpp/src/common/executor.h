// SEDA 线程池执行器（§7.4.2）。
//
// 对齐 folly 语义的最小实现（工程环境无 folly 依赖时的等价物）：
// - Executor                ≈ folly::Executor（add 派发一个任务）
// - ThreadPoolExecutor      ≈ folly::IOThreadPoolExecutor / CPUThreadPoolExecutor
//   （IO 池与 CPU 池共用实现；语义差异由实例命名/线程数/队列容量体现：
//    IO 池线程数少、承接阻塞式网络调用；CPU 池线程数=核、承接计算）
// - 有界 FIFO + 满时快速失败（ExecutorOverloaded）：背压信号逐级上传至接入层，
//   防止单池无界堆积拖垮整体时延（§7.4.7 SEDA 契约）。
#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace onetrans {

// 队列满：上游应快速拒绝（HTTP 429/503）或降级，不得无界排队
struct ExecutorOverloaded : std::runtime_error {
    explicit ExecutorOverloaded(const std::string& name)
        : std::runtime_error("executor '" + name + "' queue full") {}
};

class Executor {
public:
    virtual ~Executor() = default;
    // 队列满抛 ExecutorOverloaded（快速失败，不阻塞提交线程）
    virtual void add(std::function<void()> fn) = 0;
};

class ThreadPoolExecutor : public Executor {
public:
    ThreadPoolExecutor(std::string name, int num_threads, size_t queue_capacity);
    ~ThreadPoolExecutor() override;

    void add(std::function<void()> fn) override;
    void stop();
    const std::string& name() const { return name_; }
    size_t pending() const;
    size_t threads() const { return threads_.size(); }

private:
    void loop();

    std::string name_;
    size_t cap_;
    std::vector<std::thread> threads_;
    std::deque<std::function<void()>> q_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool stopped_ = false;
};

// 功能分池集合（§7.4.2 表格的实例化；持有并统一停机）
class ExecutorSet {
public:
    // name/io 线程/cap 逐池构建；线程数 ≤0 时取 1
    std::shared_ptr<ThreadPoolExecutor> make(const std::string& name, int num_threads,
                                             size_t queue_capacity);
    void stop_all();

private:
    std::vector<std::shared_ptr<ThreadPoolExecutor>> pools_;
    std::mutex mu_;
};

}  // namespace onetrans
