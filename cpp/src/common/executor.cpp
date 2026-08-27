#include "common/executor.h"

namespace onetrans {

ThreadPoolExecutor::ThreadPoolExecutor(std::string name, int num_threads, size_t queue_capacity)
    : name_(std::move(name)), cap_(queue_capacity == 0 ? 1 : queue_capacity) {
    const int n = num_threads > 0 ? num_threads : 1;
    threads_.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        threads_.emplace_back([this] { loop(); });
}

ThreadPoolExecutor::~ThreadPoolExecutor() { stop(); }

void ThreadPoolExecutor::add(std::function<void()> fn) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopped_) throw std::runtime_error("executor '" + name_ + "' stopped");
        if (q_.size() >= cap_) throw ExecutorOverloaded(name_);
        q_.push_back(std::move(fn));
    }
    cv_.notify_one();
}

void ThreadPoolExecutor::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopped_) return;
        stopped_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
}

size_t ThreadPoolExecutor::pending() const {
    std::lock_guard<std::mutex> lk(mu_);
    return q_.size();
}

void ThreadPoolExecutor::loop() {
    while (true) {
        std::function<void()> fn;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stopped_ || !q_.empty(); });
            if (q_.empty()) return;  // stop 且队列已 drain（快速停机：丢弃未执行任务）
            fn = std::move(q_.front());
            q_.pop_front();
        }
        fn();  // 异常不上抛：任务自捕获（阶段链均有 try/catch 包裹）
    }
}

std::shared_ptr<ThreadPoolExecutor> ExecutorSet::make(const std::string& name, int num_threads,
                                                      size_t queue_capacity) {
    auto pool = std::make_shared<ThreadPoolExecutor>(name, num_threads, queue_capacity);
    std::lock_guard<std::mutex> lk(mu_);
    pools_.push_back(pool);
    return pool;
}

void ExecutorSet::stop_all() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& p : pools_) p->stop();
    pools_.clear();
}

}  // namespace onetrans
