#pragma once

#include "rem/api.hpp"

#include <ankerl/unordered_dense.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rem::detail {

using Clock = std::chrono::steady_clock;

template <class T, std::size_t Alignment>
class AlignedAllocator {
public:
    using value_type = T;
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return static_cast<T*>(
            ::operator new(count * sizeof(T), std::align_val_t(Alignment)));
    }
    void deallocate(T* pointer, std::size_t) noexcept {
        ::operator delete(pointer, std::align_val_t(Alignment));
    }
    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <class T, class U, std::size_t Alignment>
bool operator==(
    const AlignedAllocator<T, Alignment>&,
    const AlignedAllocator<U, Alignment>&) noexcept {
    return true;
}

using AlignedDoubles = std::vector<double, AlignedAllocator<double, 64>>;

inline double elapsed_seconds(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

inline std::size_t words_for(std::uint32_t qubits) {
    return (static_cast<std::size_t>(qubits) + 63U) / 64U;
}

inline std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return a + b;
}

inline std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return a * b;
}

class ThreadPool {
public:
    explicit ThreadPool(std::uint32_t requested_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]] std::uint32_t size() const noexcept { return thread_count_; }

    template <class Function>
    void parallel_for(std::size_t size, Function&& function) {
        if (size == 0) {
            return;
        }
        using Fn = std::remove_reference_t<Function>;
        Fn& fn_ref = function;
        auto thunk = [](void* context, std::size_t begin, std::size_t end, std::uint32_t tid) {
            (*static_cast<Fn*>(context))(begin, end, tid);
        };
        run(size, &fn_ref, thunk);
    }

private:
    using Task = void (*)(void*, std::size_t, std::size_t, std::uint32_t);
    void run(std::size_t size, void* context, Task task);
    void worker_loop(std::uint32_t tid);
    std::pair<std::size_t, std::size_t> range_for(
        std::size_t size, std::uint32_t tid) const noexcept;

    std::uint32_t thread_count_ = 1;
    std::vector<std::jthread> workers_;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::uint64_t generation_ = 0;
    std::size_t task_size_ = 0;
    void* task_context_ = nullptr;
    Task task_ = nullptr;
    std::uint32_t remaining_workers_ = 0;
    std::exception_ptr task_exception_;
    bool stopping_ = false;
};

struct PackedStates {
    std::uint32_t qubits = 0;
    std::size_t words_per_state = 0;
    std::vector<std::uint64_t> words;

    [[nodiscard]] std::size_t size() const noexcept {
        return words_per_state == 0 ? 0 : words.size() / words_per_state;
    }
    [[nodiscard]] std::span<const std::uint64_t> state(std::size_t index) const {
        return {words.data() + index * words_per_state, words_per_state};
    }
    [[nodiscard]] std::span<std::uint64_t> state(std::size_t index) {
        return {words.data() + index * words_per_state, words_per_state};
    }
    [[nodiscard]] bool bit(std::size_t index, std::uint32_t qubit) const noexcept {
        return (words[index * words_per_state + qubit / 64U] >> (qubit % 64U)) & 1U;
    }
};

class StateIndex {
public:
    explicit StateIndex(const PackedStates* states, std::size_t expected_size);
    void reset_states(const PackedStates* states) noexcept { states_ = states; }
    std::pair<std::uint32_t, bool> insert_or_find(std::span<const std::uint64_t> state);
    [[nodiscard]] std::uint32_t find(std::span<const std::uint64_t> state) const noexcept;

private:
    static std::uint64_t hash_words(std::span<const std::uint64_t> state) noexcept;
    [[nodiscard]] bool equal(std::uint32_t index, std::span<const std::uint64_t> state) const noexcept;
    void insert_index(std::uint32_t index);

    const PackedStates* states_ = nullptr;
    ankerl::unordered_dense::map<std::uint64_t, std::uint32_t> fast64_;
    std::vector<std::uint32_t> buckets_;
    std::size_t mask_ = 0;
};

struct PreparedInput {
    PackedStates states;
    std::vector<double> probabilities;  // RHS-major.
    std::vector<std::uint32_t> original_to_unique;
    std::size_t right_hand_sides = 1;
};

PreparedInput prepare_input(
    std::span<const std::uint64_t> packed,
    std::size_t count,
    std::uint32_t qubits,
    std::span<const double> probabilities,
    std::size_t right_hand_sides);

std::uint32_t hamming_distance(
    std::span<const std::uint64_t> lhs,
    std::span<const std::uint64_t> rhs,
    std::uint32_t stop_after = std::numeric_limits<std::uint32_t>::max()) noexcept;

std::uint64_t hamming_ball_size(std::uint32_t qubits, std::uint32_t distance) noexcept;

template <class Callback>
void for_each_ball_candidate_64(
    std::uint64_t state, std::uint32_t qubits, std::uint32_t distance, Callback&& callback) {
    callback(state);
    if (distance == 0) {
        return;
    }
    for (std::uint32_t a = 0; a < qubits; ++a) {
        callback(state ^ (1ULL << a));
    }
    if (distance == 1) {
        return;
    }
    for (std::uint32_t a = 0; a < qubits; ++a) {
        for (std::uint32_t b = a + 1; b < qubits; ++b) {
            callback(state ^ (1ULL << a) ^ (1ULL << b));
        }
    }
    if (distance == 2) {
        return;
    }
    for (std::uint32_t a = 0; a < qubits; ++a) {
        for (std::uint32_t b = a + 1; b < qubits; ++b) {
            for (std::uint32_t c = b + 1; c < qubits; ++c) {
                callback(state ^ (1ULL << a) ^ (1ULL << b) ^ (1ULL << c));
            }
        }
    }
}

void for_each_ball_candidate_generic(
    std::span<const std::uint64_t> state,
    std::uint32_t qubits,
    std::uint32_t distance,
    const std::function<void(std::span<const std::uint64_t>)>& callback);

template <class Callback>
void for_each_ball_candidate_generic_scratch(
    std::span<const std::uint64_t> state,
    std::uint32_t qubits,
    std::uint32_t distance,
    std::span<std::uint64_t> scratch,
    Callback&& callback) {
    std::copy(state.begin(), state.end(), scratch.begin());
    callback(std::span<const std::uint64_t>(scratch));
    distance = std::min(distance, qubits);
    if (distance == 0) {
        return;
    }
    auto flip = [&](std::uint32_t qubit) {
        scratch[qubit / 64U] ^= 1ULL << (qubit % 64U);
    };
    for (std::uint32_t a = 0; a < qubits; ++a) {
        flip(a);
        callback(std::span<const std::uint64_t>(scratch));
        flip(a);
    }
    if (distance == 1) {
        return;
    }
    for (std::uint32_t a = 0; a < qubits; ++a) {
        flip(a);
        for (std::uint32_t b = a + 1; b < qubits; ++b) {
            flip(b);
            callback(std::span<const std::uint64_t>(scratch));
            flip(b);
        }
        flip(a);
    }
    if (distance == 2) {
        return;
    }
    for (std::uint32_t a = 0; a < qubits; ++a) {
        flip(a);
        for (std::uint32_t b = a + 1; b < qubits; ++b) {
            flip(b);
            for (std::uint32_t c = b + 1; c < qubits; ++c) {
                flip(c);
                callback(std::span<const std::uint64_t>(scratch));
                flip(c);
            }
            flip(b);
        }
        flip(a);
    }
    if (distance == 3) {
        return;
    }
    auto visit = [&](auto&& self, std::uint32_t start, std::uint32_t remaining) -> void {
        if (remaining == 0) {
            callback(std::span<const std::uint64_t>(scratch));
            return;
        }
        for (std::uint32_t q = start; q + remaining <= qubits; ++q) {
            flip(q);
            self(self, q + 1U, remaining - 1U);
            flip(q);
        }
    };
    for (std::uint32_t rank = 4; rank <= distance; ++rank) {
        visit(visit, 0, rank);
    }
}

struct CsrGraph {
    std::vector<std::uint64_t> row_offsets;
    std::vector<std::uint32_t> column_indices;
};

GraphStrategy choose_graph_strategy(
    std::size_t states,
    std::uint32_t qubits,
    std::uint32_t distance,
    std::size_t words_per_state);

CsrGraph build_hamming_graph(
    const PackedStates& states,
    const StateIndex& index,
    std::uint32_t distance,
    GraphStrategy strategy,
    ThreadPool& pool,
    std::uint64_t memory_budget,
    Diagnostics& diagnostics);

struct Calibration {
    std::uint32_t qubits = 0;
    std::vector<double> values;
    std::vector<double> ratios;  // [q][source bit]
    std::vector<double> base;
    std::vector<std::uint8_t> ratio_safe;
    double near_zero = 1e-15;
};

Calibration prepare_calibration(
    const PackedStates& states,
    std::span<const double> matrices,
    double near_zero,
    ThreadPool& pool,
    Diagnostics& diagnostics);

double direct_element(
    const PackedStates& states,
    const Calibration& calibration,
    std::uint32_t target,
    std::uint32_t source) noexcept;

double fast_element(
    const PackedStates& states,
    const Calibration& calibration,
    std::uint32_t target,
    std::uint32_t source) noexcept;

enum class OperatorStorage {
    Dense,
    TopologyCsr,
    ValuedCsr,
    MatrixFreeBall,
    MatrixFreeScan,
};

struct AssignmentOperator {
    const PackedStates* states = nullptr;
    const StateIndex* state_index = nullptr;
    const Calibration* calibration = nullptr;
    ThreadPool* pool = nullptr;
    OperatorStorage storage = OperatorStorage::MatrixFreeScan;
    std::uint32_t distance = 0;
    CsrGraph graph;
    AlignedDoubles column_norms;
    AlignedDoubles jacobi_inverse;
    AlignedDoubles dense_column_major;
    AlignedDoubles csr_values;
    mutable std::vector<std::uint64_t> matrix_free_scratch;

    [[nodiscard]] std::size_t size() const noexcept { return states->size(); }
    void matvec(std::span<const double> input, std::span<double> output) const;
};

struct MemoryEstimate {
    std::uint64_t dense = 0;
    std::uint64_t topology_csr = 0;
    std::uint64_t valued_csr = 0;
    std::uint64_t matrix_free = 0;
    std::uint64_t krylov = 0;
    std::uint64_t projected_edges = 0;
    double sampled_density = 0.0;
};

MemoryEstimate estimate_memory(
    const PackedStates& states,
    std::uint32_t distance,
    const Options& options,
    std::size_t right_hand_sides);

AssignmentOperator build_operator(
    const PackedStates& states,
    const StateIndex& index,
    const Calibration& calibration,
    const Options& options,
    Backend backend,
    SparseStorage sparse_storage,
    GraphStrategy graph_strategy,
    std::size_t right_hand_sides,
    ThreadPool& pool,
    Diagnostics& diagnostics);

struct SolverResult {
    std::vector<double> x;
    std::uint32_t iterations = 0;
    double residual = 0.0;
    double relative_residual = 0.0;
    bool converged = false;
    std::string warning;
};

SolverResult solve_gmres(
    const AssignmentOperator& op,
    std::span<const double> rhs,
    const Options& options);

SolverResult solve_bicgstab(
    const AssignmentOperator& op,
    std::span<const double> rhs,
    const Options& options);

std::vector<double> solve_dense_many(
    AlignedDoubles matrix_column_major,
    std::span<const double> rhs_major,
    std::size_t right_hand_sides,
    std::size_t dimension,
    std::string& error);

double l2_norm(std::span<const double> values) noexcept;
double true_residual(
    const AssignmentOperator& op,
    std::span<const double> x,
    std::span<const double> rhs,
    std::vector<double>& workspace);

}  // namespace rem::detail
