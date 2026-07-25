#include "detail.hpp"

namespace rem::detail {

ThreadPool::ThreadPool(std::uint32_t requested_threads) {
    if (requested_threads == 0) {
        requested_threads = std::thread::hardware_concurrency();
    }
    thread_count_ = std::max<std::uint32_t>(1, requested_threads);
    workers_.reserve(thread_count_ > 0 ? thread_count_ - 1 : 0);
    try {
        for (std::uint32_t tid = 1; tid < thread_count_; ++tid) {
            workers_.emplace_back([this, tid] { worker_loop(tid); });
        }
    } catch (...) {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        work_cv_.notify_all();
        workers_.clear();
        throw;
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        ++generation_;
    }
    work_cv_.notify_all();
    workers_.clear();
}

std::pair<std::size_t, std::size_t> ThreadPool::range_for(
    std::size_t size, std::uint32_t tid) const noexcept {
    const std::size_t begin = size * tid / thread_count_;
    const std::size_t end = size * (tid + 1U) / thread_count_;
    return {begin, end};
}

void ThreadPool::run(std::size_t size, void* context, Task task) {
    if (thread_count_ == 1 || size == 1) {
        task(context, 0, size, 0);
        return;
    }
    {
        std::lock_guard lock(mutex_);
        task_size_ = size;
        task_context_ = context;
        task_ = task;
        task_exception_ = nullptr;
        remaining_workers_ = static_cast<std::uint32_t>(workers_.size());
        ++generation_;
    }
    work_cv_.notify_all();
    const auto [begin, end] = range_for(size, 0);
    try {
        task(context, begin, end, 0);
    } catch (...) {
        std::lock_guard lock(mutex_);
        task_exception_ = std::current_exception();
    }
    std::unique_lock lock(mutex_);
    done_cv_.wait(lock, [this] { return remaining_workers_ == 0; });
    const std::exception_ptr exception = task_exception_;
    task_ = nullptr;
    task_context_ = nullptr;
    lock.unlock();
    if (exception) {
        std::rethrow_exception(exception);
    }
}

void ThreadPool::worker_loop(std::uint32_t tid) {
    std::uint64_t seen_generation = 0;
    for (;;) {
        std::unique_lock lock(mutex_);
        work_cv_.wait(lock, [this, seen_generation] {
            return stopping_ || generation_ != seen_generation;
        });
        if (stopping_) {
            return;
        }
        seen_generation = generation_;
        const std::size_t size = task_size_;
        void* context = task_context_;
        Task task = task_;
        const auto [begin, end] = range_for(size, tid);
        lock.unlock();
        try {
            if (begin < end) {
                task(context, begin, end, tid);
            }
        } catch (...) {
            lock.lock();
            if (!task_exception_) {
                task_exception_ = std::current_exception();
            }
            lock.unlock();
        }
        lock.lock();
        if (--remaining_workers_ == 0) {
            done_cv_.notify_one();
        }
    }
}

}  // namespace rem::detail
