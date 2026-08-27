#include "common/executor.h"

#include <folly/executors/thread_factory/NamedThreadFactory.h>

namespace onetrans {

Executor::Executor(std::string name, int num_threads, size_t queue_capacity, bool io)
    : name_(std::move(name)),
      cap_(queue_capacity == 0 ? 1 : queue_capacity),
      io_(io) {
    const size_t n = num_threads > 0 ? static_cast<size_t>(num_threads) : 1;
    if (io_) {
        // IO 池（EmbedLookup / KV I/O）：每线程独立 event base，承接阻塞式网络/外部调用。
        io_exec_ = std::make_shared<folly::IOThreadPoolExecutor>(
            n, std::make_shared<folly::NamedThreadFactory>(name_));
    } else {
        // CPU 池（Frontend Encode / Compute）：folly 绑核线程 + 无界队列，
        // 有界性由 add() 侧的 pending 快照软判定兜底（避免阻塞式满队列死锁）。
        cpu_ = std::make_shared<folly::CPUThreadPoolExecutor>(
            n, std::make_shared<folly::NamedThreadFactory>(name_));
    }
}

Executor::~Executor() { stop(); }

void Executor::add(std::function<void()> fn) {
    size_t pending = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopped_) throw std::logic_error("executor '" + name_ + "' stopped");
        // 快照式软背压：判定与入队非原子；瞬时超限由接入层降级吸收即可。
        pending = cpu_ ? cpu_->getPendingTaskCount() : io_exec_->getPendingTaskCount();
        if (pending >= cap_) throw ExecutorOverloaded(name_);
    }
    if (cpu_) {
        cpu_->add(std::move(fn));
    } else {
        io_exec_->add(std::move(fn));
    }
}

void Executor::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopped_) return;
        stopped_ = true;
    }
    // join：folly 在退出时排空已入队任务并回收线程（与自研池的 drain+join 语义对齐）。
    if (cpu_) cpu_->join();
    if (io_exec_) io_exec_->join();
}

std::shared_ptr<Executor> ExecutorSet::make(const std::string& name, int num_threads,
                                            size_t queue_capacity, bool io) {
    auto pool = std::make_shared<Executor>(name, num_threads, queue_capacity, io);
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