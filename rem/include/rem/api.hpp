#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rem {

enum class Backend {
    Auto,
    Dense,
    ExplicitSparse,
    MatrixFree,
};

enum class Solver {
    Auto,
    Direct,
    GMRES,
    BiCGSTAB,
};

enum class GraphStrategy {
    Auto,
    HammingBall,
    BlockedPairwise,
};

enum class SparseStorage {
    Auto,
    TopologyOnly,
    Valued,
};

enum class Orthogonalization {
    Auto,
    ModifiedGramSchmidt,
    ClassicalGramSchmidt,
};

struct Options {
    Backend backend = Backend::Auto;
    Solver solver = Solver::Auto;
    GraphStrategy graph_strategy = GraphStrategy::Auto;
    SparseStorage sparse_storage = SparseStorage::Auto;
    Orthogonalization orthogonalization = Orthogonalization::Auto;
    std::uint32_t max_hamming_distance = 3;
    double relative_tolerance = 1e-6;
    double absolute_tolerance = 0.0;
    std::uint32_t gmres_restart = 30;
    std::uint32_t max_iterations = 200;
    std::uint32_t threads = 0;
    std::uint64_t memory_budget_bytes = 6ULL * 1024ULL * 1024ULL * 1024ULL;
    bool project_to_probability_simplex = false;
    bool cache_topology = true;
    bool cache_values = false;
    double near_zero_threshold = 1e-15;
};

struct Diagnostics {
    Backend selected_backend = Backend::Auto;
    Solver selected_solver = Solver::Auto;
    GraphStrategy selected_graph_strategy = GraphStrategy::Auto;
    SparseStorage selected_sparse_storage = SparseStorage::Auto;
    Orthogonalization selected_orthogonalization = Orthogonalization::Auto;
    std::uint64_t input_states = 0;
    std::uint64_t unique_states = 0;
    std::uint64_t retained_edges = 0;
    std::uint64_t hamming_ball_size = 0;
    std::uint64_t estimated_memory_bytes = 0;
    std::uint64_t peak_working_memory_bytes = 0;
    std::uint32_t iterations = 0;
    std::uint32_t threads = 1;
    double sampled_density = 0.0;
    double residual_norm = 0.0;
    double relative_residual = 0.0;
    double setup_seconds = 0.0;
    double solve_seconds = 0.0;
    double projection_seconds = 0.0;
    bool converged = false;
    bool used_safe_element_fallback = false;
    bool jacobi_had_zero_diagonal = false;
    std::string backend_reason;
    std::string threading_backend = "std::jthread";
    std::string warning;
};

struct Result {
    std::vector<double> quasi_probabilities;
    std::optional<std::vector<double>> projected_probabilities;
    // State-major packed words for the stable unique-state order.
    std::vector<std::uint64_t> unique_packed_states;
    std::vector<std::uint32_t> original_to_unique;
    Diagnostics diagnostics;
};

// packed_bitstrings is state-major and contains
// bitstring_count * ceil(measured_qubits / 64) words.
// confusion_matrices has shape [measured_qubits, 2, 2], flattened in C order
// as C[q][measured_bit][true_bit].
Result mitigate(
    std::span<const std::uint64_t> packed_bitstrings,
    std::size_t bitstring_count,
    std::uint32_t measured_qubits,
    std::span<const double> raw_probabilities,
    std::span<const double> confusion_matrices,
    const Options& options = {});

// Multiple raw distributions sharing states, calibration, and cutoff. The input
// and output are RHS-major. Dense mode factorizes A once for the whole batch.
std::vector<Result> mitigate_many(
    std::span<const std::uint64_t> packed_bitstrings,
    std::size_t bitstring_count,
    std::uint32_t measured_qubits,
    std::span<const double> raw_probabilities,
    std::size_t right_hand_sides,
    std::span<const double> confusion_matrices,
    const Options& options = {});

// Column-major normalized matrix, intended for validation and small problems.
std::vector<double> build_reduced_matrix(
    std::span<const std::uint64_t> packed_unique_states,
    std::size_t state_count,
    std::uint32_t measured_qubits,
    std::span<const double> confusion_matrices,
    std::uint32_t max_hamming_distance,
    double near_zero_threshold = 1e-15);

std::vector<double> project_probability_simplex(std::span<const double> values);

const char* to_string(Backend value) noexcept;
const char* to_string(Solver value) noexcept;
const char* to_string(GraphStrategy value) noexcept;
const char* to_string(SparseStorage value) noexcept;
const char* to_string(Orthogonalization value) noexcept;

}  // namespace rem
