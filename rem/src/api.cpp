#include "detail.hpp"

#include <sstream>

namespace rem {
namespace {

using detail::AssignmentOperator;
using detail::MemoryEstimate;
using detail::PreparedInput;
using detail::SolverResult;

std::uint64_t comfortable_budget(std::uint64_t budget) noexcept {
    return budget - budget / 5U;
}

std::uint64_t two_thirds_budget(std::uint64_t budget) noexcept {
    return budget - budget / 3U;
}

bool auto_sparse_uses_values(
    const Options& options, const MemoryEstimate& memory) noexcept {
    if (options.cache_values &&
        memory.valued_csr <= comfortable_budget(options.memory_budget_bytes)) {
        return true;
    }
    return memory.valued_csr <= two_thirds_budget(options.memory_budget_bytes);
}

void validate_options(const Options& options) {
    if (!std::isfinite(options.relative_tolerance) || options.relative_tolerance < 0.0 ||
        !std::isfinite(options.absolute_tolerance) || options.absolute_tolerance < 0.0) {
        throw std::invalid_argument("solver tolerances must be finite and nonnegative");
    }
    if (options.relative_tolerance == 0.0 && options.absolute_tolerance == 0.0) {
        throw std::invalid_argument("at least one solver tolerance must be positive");
    }
    if (options.gmres_restart == 0 || options.max_iterations == 0) {
        throw std::invalid_argument("GMRES restart and max iterations must be positive");
    }
    if (options.memory_budget_bytes < 1024U * 1024U) {
        throw std::invalid_argument("memory budget must be at least 1 MiB");
    }
}

Backend select_backend(
    const Options& options,
    const MemoryEstimate& memory,
    std::size_t count,
    Diagnostics& diagnostics) {
    if (options.backend != Backend::Auto) {
        std::uint64_t required = memory.matrix_free;
        if (options.backend == Backend::Dense) {
            required = memory.dense;
        } else if (options.backend == Backend::ExplicitSparse) {
            const bool valued =
                options.sparse_storage == SparseStorage::Valued ||
                (options.sparse_storage == SparseStorage::Auto &&
                 auto_sparse_uses_values(options, memory));
            required = valued ? memory.valued_csr : memory.topology_csr;
        }
        if (required > options.memory_budget_bytes) {
            std::ostringstream message;
            message << "forced " << to_string(options.backend) << " backend needs about "
                    << required << " bytes, exceeding the " << options.memory_budget_bytes
                    << "-byte budget";
            throw std::runtime_error(message.str());
        }
        diagnostics.backend_reason = "backend was explicitly forced";
        diagnostics.estimated_memory_bytes = required;
        return options.backend;
    }

    constexpr std::size_t kDirectFloorDimension = 32;
    constexpr double kDenseDensity = 0.35;
    const std::uint64_t comfortable =
        comfortable_budget(options.memory_budget_bytes);
    if (memory.dense <= comfortable &&
        (count <= kDirectFloorDimension ||
         memory.sampled_density >= kDenseDensity)) {
        diagnostics.backend_reason =
            count <= kDirectFloorDimension
                ? "dense direct solve is retained for K <= 32"
                : "sampled retained density is high and dense workspace fits the budget";
        diagnostics.estimated_memory_bytes = memory.dense;
        return Backend::Dense;
    }
    if (options.cache_topology && memory.topology_csr <= comfortable) {
        diagnostics.backend_reason =
            "projected sparse graph fits the comfortable memory budget";
        diagnostics.estimated_memory_bytes = memory.topology_csr;
        return Backend::ExplicitSparse;
    }
    if (memory.matrix_free > options.memory_budget_bytes) {
        std::ostringstream message;
        message << "no backend fits the " << options.memory_budget_bytes
                << "-byte budget; the minimum matrix-free estimate is "
                << memory.matrix_free << " bytes";
        throw std::runtime_error(message.str());
    }
    diagnostics.backend_reason =
        "dense and projected CSR working sets do not fit the configured budget";
    diagnostics.estimated_memory_bytes = memory.matrix_free;
    return Backend::MatrixFree;
}

SparseStorage select_sparse_storage(
    const Options& options,
    const MemoryEstimate& memory,
    Backend backend) {
    if (backend != Backend::ExplicitSparse) {
        return SparseStorage::Auto;
    }
    if (options.sparse_storage != SparseStorage::Auto) {
        return options.sparse_storage;
    }
    if (options.cache_values &&
        memory.valued_csr <= comfortable_budget(options.memory_budget_bytes)) {
        return SparseStorage::Valued;
    }
    // Values normally amortize after a few Krylov matvecs when they fit.
    if (memory.valued_csr <= two_thirds_budget(options.memory_budget_bytes)) {
        return SparseStorage::Valued;
    }
    return SparseStorage::TopologyOnly;
}

Solver select_solver(const Options& options, Backend backend) {
    if (options.solver == Solver::Auto) {
        return backend == Backend::Dense ? Solver::Direct : Solver::GMRES;
    }
    if (options.solver == Solver::Direct && backend != Backend::Dense) {
        throw std::invalid_argument("the direct solver requires the dense backend");
    }
    return options.solver;
}

std::uint64_t selected_peak_estimate(
    const MemoryEstimate& memory, Backend backend, SparseStorage sparse) {
    if (backend == Backend::Dense) {
        return memory.dense;
    }
    if (backend == Backend::ExplicitSparse) {
        return sparse == SparseStorage::Valued ? memory.valued_csr
                                               : memory.topology_csr;
    }
    return memory.matrix_free;
}

std::uint32_t select_thread_count(
    const Options& options,
    Backend backend,
    std::size_t count,
    double sampled_density) {
    if (options.threads != 0) {
        return options.threads;
    }
    const std::uint32_t hardware =
        std::max<std::uint32_t>(1, std::thread::hardware_concurrency());
    if (backend == Backend::Dense &&
        (count <= 32 || (count < 768 && sampled_density < 0.35))) {
        return 1;
    }
    if (backend == Backend::ExplicitSparse && count < 2048) {
        return 1;
    }
#if defined(__APPLE__) && defined(__aarch64__)
    // On the 6P+4E M2 Pro, six workers were faster than all ten logical cores
    // for the measured mid-size sparse setup while ten won again at K=20000.
    if (backend == Backend::ExplicitSparse && count < 8192) {
        return std::min<std::uint32_t>(6, hardware);
    }
#endif
    if (backend == Backend::MatrixFree && count < 8192) {
        return std::min<std::uint32_t>(4, hardware);
    }
    return hardware;
}

Orthogonalization select_orthogonalization(
    const Options& options, std::size_t count) {
    if (options.orthogonalization != Orthogonalization::Auto) {
        return options.orthogonalization;
    }
#if defined(REM_USE_ACCELERATE)
    if (count >= 4096) {
        return Orthogonalization::ClassicalGramSchmidt;
    }
#endif
    return Orthogonalization::ModifiedGramSchmidt;
}

Result make_result(
    const PreparedInput& input,
    std::size_t rhs,
    std::vector<double> quasi,
    Diagnostics diagnostics,
    const Options& options) {
    Result result;
    result.quasi_probabilities = std::move(quasi);
    result.unique_packed_states = input.states.words;
    result.original_to_unique = input.original_to_unique;
    if (options.project_to_probability_simplex) {
        const auto projection_start = detail::Clock::now();
        result.projected_probabilities =
            project_probability_simplex(result.quasi_probabilities);
        diagnostics.projection_seconds = detail::elapsed_seconds(projection_start);
    }
    (void)rhs;
    result.diagnostics = std::move(diagnostics);
    return result;
}

}  // namespace

Result mitigate(
    std::span<const std::uint64_t> packed_bitstrings,
    std::size_t bitstring_count,
    std::uint32_t measured_qubits,
    std::span<const double> raw_probabilities,
    std::span<const double> confusion_matrices,
    const Options& options) {
    auto results = mitigate_many(
        packed_bitstrings, bitstring_count, measured_qubits, raw_probabilities, 1,
        confusion_matrices, options);
    return std::move(results.front());
}

std::vector<Result> mitigate_many(
    std::span<const std::uint64_t> packed_bitstrings,
    std::size_t bitstring_count,
    std::uint32_t measured_qubits,
    std::span<const double> raw_probabilities,
    std::size_t right_hand_sides,
    std::span<const double> confusion_matrices,
    const Options& options) {
    validate_options(options);
    const auto setup_start = detail::Clock::now();
    PreparedInput input = detail::prepare_input(
        packed_bitstrings, bitstring_count, measured_qubits, raw_probabilities,
        right_hand_sides);
    Diagnostics diagnostics;
    diagnostics.input_states = bitstring_count;
    diagnostics.unique_states = input.states.size();
    diagnostics.hamming_ball_size = detail::hamming_ball_size(
        measured_qubits, std::min(measured_qubits, options.max_hamming_distance));
    const MemoryEstimate memory =
        detail::estimate_memory(
            input.states, options.max_hamming_distance, options,
            right_hand_sides);
    diagnostics.sampled_density = memory.sampled_density;
    const Backend backend =
        select_backend(options, memory, input.states.size(), diagnostics);
    const SparseStorage sparse_storage =
        select_sparse_storage(options, memory, backend);
    const Solver solver = select_solver(options, backend);
    diagnostics.selected_backend = backend;
    diagnostics.selected_sparse_storage = sparse_storage;
    diagnostics.selected_solver = solver;
    diagnostics.selected_orthogonalization =
        select_orthogonalization(options, input.states.size());
    diagnostics.peak_working_memory_bytes =
        selected_peak_estimate(memory, backend, sparse_storage);
    diagnostics.estimated_memory_bytes = diagnostics.peak_working_memory_bytes;
    detail::ThreadPool pool(select_thread_count(
        options, backend, input.states.size(), memory.sampled_density));
    diagnostics.threads = pool.size();
    detail::StateIndex index(&input.states, input.states.size());
    auto calibration = detail::prepare_calibration(
        input.states, confusion_matrices, options.near_zero_threshold, pool,
        diagnostics);
    GraphStrategy graph_strategy = options.graph_strategy;
    if (graph_strategy == GraphStrategy::Auto) {
        graph_strategy = detail::choose_graph_strategy(
            input.states.size(), measured_qubits,
            std::min(measured_qubits, options.max_hamming_distance),
            input.states.words_per_state);
    }

    AssignmentOperator op = detail::build_operator(
        input.states, index, calibration, options, backend, sparse_storage,
        graph_strategy, right_hand_sides, pool, diagnostics);
    diagnostics.setup_seconds = detail::elapsed_seconds(setup_start);
    std::vector<Result> results;
    results.reserve(right_hand_sides);
    const std::size_t dimension = input.states.size();

    if (solver == Solver::Direct) {
        const auto solve_start = detail::Clock::now();
        std::string error;
        auto dense_for_factorization = std::move(op.dense_column_major);
        // LAPACK overwrites A. Reuse the calibration-based operator for final
        // true residuals instead of retaining a second dense matrix.
        op.storage = detail::OperatorStorage::MatrixFreeScan;
        std::vector<double> solutions = detail::solve_dense_many(
            std::move(dense_for_factorization), input.probabilities,
            right_hand_sides, dimension, error);
        const double solve_seconds = detail::elapsed_seconds(solve_start);
        if (!error.empty()) {
            throw std::runtime_error(error);
        }
        for (std::size_t rhs = 0; rhs < right_hand_sides; ++rhs) {
            Diagnostics rhs_diagnostics = diagnostics;
            rhs_diagnostics.solve_seconds = solve_seconds;
            std::span<const double> solution(
                solutions.data() + rhs * dimension, dimension);
            std::span<const double> source_rhs(
                input.probabilities.data() + rhs * dimension, dimension);
            std::vector<double> residual_workspace;
            rhs_diagnostics.residual_norm =
                detail::true_residual(op, solution, source_rhs, residual_workspace);
            const double rhs_norm = detail::l2_norm(source_rhs);
            rhs_diagnostics.relative_residual =
                rhs_norm == 0.0 ? rhs_diagnostics.residual_norm
                                : rhs_diagnostics.residual_norm / rhs_norm;
            const double tolerance =
                options.absolute_tolerance + options.relative_tolerance * rhs_norm;
            rhs_diagnostics.converged = rhs_diagnostics.residual_norm <= tolerance;
            results.push_back(make_result(
                input, rhs, std::vector<double>(solution.begin(), solution.end()),
                std::move(rhs_diagnostics), options));
        }
        return results;
    }

    for (std::size_t rhs = 0; rhs < right_hand_sides; ++rhs) {
        std::span<const double> source_rhs(
            input.probabilities.data() + rhs * dimension, dimension);
        const auto solve_start = detail::Clock::now();
        SolverResult solved =
            solver == Solver::GMRES
                ? detail::solve_gmres(op, source_rhs, options)
                : detail::solve_bicgstab(op, source_rhs, options);
        Diagnostics rhs_diagnostics = diagnostics;
        rhs_diagnostics.solve_seconds = detail::elapsed_seconds(solve_start);
        rhs_diagnostics.iterations = solved.iterations;
        rhs_diagnostics.residual_norm = solved.residual;
        rhs_diagnostics.relative_residual = solved.relative_residual;
        rhs_diagnostics.converged = solved.converged;
        if (!solved.warning.empty()) {
            if (!rhs_diagnostics.warning.empty()) {
                rhs_diagnostics.warning += "; ";
            }
            rhs_diagnostics.warning += solved.warning;
        }
        results.push_back(make_result(
            input, rhs, std::move(solved.x), std::move(rhs_diagnostics), options));
    }
    return results;
}

std::vector<double> build_reduced_matrix(
    std::span<const std::uint64_t> packed_unique_states,
    std::size_t state_count,
    std::uint32_t measured_qubits,
    std::span<const double> confusion_matrices,
    std::uint32_t max_hamming_distance,
    double near_zero_threshold) {
    std::vector<double> dummy_probabilities(state_count, 0.0);
    PreparedInput input = detail::prepare_input(
        packed_unique_states, state_count, measured_qubits, dummy_probabilities, 1);
    if (input.states.size() != state_count) {
        throw std::invalid_argument(
            "build_reduced_matrix requires already-unique bitstrings");
    }
    Options options;
    options.backend = Backend::Dense;
    options.solver = Solver::Direct;
    options.max_hamming_distance = max_hamming_distance;
    options.near_zero_threshold = near_zero_threshold;
    options.memory_budget_bytes = std::numeric_limits<std::uint64_t>::max();
    detail::ThreadPool pool(1);
    Diagnostics diagnostics;
    detail::StateIndex index(&input.states, input.states.size());
    auto calibration = detail::prepare_calibration(
        input.states, confusion_matrices, near_zero_threshold, pool, diagnostics);
    AssignmentOperator op = detail::build_operator(
        input.states, index, calibration, options, Backend::Dense,
        SparseStorage::Auto, GraphStrategy::Auto, 1, pool, diagnostics);
    return {op.dense_column_major.begin(), op.dense_column_major.end()};
}

const char* to_string(Backend value) noexcept {
    switch (value) {
        case Backend::Auto:
            return "auto";
        case Backend::Dense:
            return "dense";
        case Backend::ExplicitSparse:
            return "explicit_sparse";
        case Backend::MatrixFree:
            return "matrix_free";
    }
    return "unknown";
}

const char* to_string(Solver value) noexcept {
    switch (value) {
        case Solver::Auto:
            return "auto";
        case Solver::Direct:
            return "direct";
        case Solver::GMRES:
            return "gmres";
        case Solver::BiCGSTAB:
            return "bicgstab";
    }
    return "unknown";
}

const char* to_string(GraphStrategy value) noexcept {
    switch (value) {
        case GraphStrategy::Auto:
            return "auto";
        case GraphStrategy::HammingBall:
            return "hamming_ball";
        case GraphStrategy::BlockedPairwise:
            return "blocked_pairwise";
    }
    return "unknown";
}

const char* to_string(SparseStorage value) noexcept {
    switch (value) {
        case SparseStorage::Auto:
            return "auto";
        case SparseStorage::TopologyOnly:
            return "topology_only";
        case SparseStorage::Valued:
            return "valued";
    }
    return "unknown";
}

const char* to_string(Orthogonalization value) noexcept {
    switch (value) {
        case Orthogonalization::Auto:
            return "auto";
        case Orthogonalization::ModifiedGramSchmidt:
            return "modified_gram_schmidt";
        case Orthogonalization::ClassicalGramSchmidt:
            return "classical_gram_schmidt";
    }
    return "unknown";
}

}  // namespace rem
