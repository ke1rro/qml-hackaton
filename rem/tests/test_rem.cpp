#include "rem/api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                                    \
    do {                                                                                    \
        if (!(condition)) {                                                                 \
            std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: " #condition       \
                      << '\n';                                                              \
            ++failures;                                                                     \
        }                                                                                   \
    } while (false)

void check_close(double actual, double expected, double tolerance, const char* label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::cerr << label << ": got " << actual << ", expected " << expected
                  << " within " << tolerance << '\n';
        ++failures;
    }
}

std::vector<double> calibration(std::uint32_t qubits) {
    std::vector<double> values(static_cast<std::size_t>(qubits) * 4U);
    for (std::uint32_t q = 0; q < qubits; ++q) {
        const double error_01 = 0.01 + 0.003 * q;
        const double error_10 = 0.03 + 0.002 * q;
        values[static_cast<std::size_t>(q) * 4U + 0] = 1.0 - error_10;
        values[static_cast<std::size_t>(q) * 4U + 1] = error_01;
        values[static_cast<std::size_t>(q) * 4U + 2] = error_10;
        values[static_cast<std::size_t>(q) * 4U + 3] = 1.0 - error_01;
    }
    return values;
}

std::uint32_t popcount(std::uint64_t value) {
    return static_cast<std::uint32_t>(std::popcount(value));
}

double reference_element(
    std::uint64_t target,
    std::uint64_t source,
    std::uint32_t qubits,
    const std::vector<double>& cals) {
    double value = 1.0;
    for (std::uint32_t q = 0; q < qubits; ++q) {
        const std::uint32_t measured = (target >> q) & 1U;
        const std::uint32_t truth = (source >> q) & 1U;
        value *= cals[static_cast<std::size_t>(q) * 4U + measured * 2U + truth];
    }
    return value;
}

std::vector<double> reference_matrix(
    const std::vector<std::uint64_t>& states,
    std::uint32_t qubits,
    const std::vector<double>& cals,
    std::uint32_t distance) {
    const std::size_t count = states.size();
    std::vector<double> matrix(count * count, 0.0);
    for (std::size_t column = 0; column < count; ++column) {
        double norm = 0.0;
        for (std::size_t row = 0; row < count; ++row) {
            if (popcount(states[row] ^ states[column]) <= distance) {
                matrix[column * count + row] =
                    reference_element(states[row], states[column], qubits, cals);
                norm += matrix[column * count + row];
            }
        }
        for (std::size_t row = 0; row < count; ++row) {
            matrix[column * count + row] /= norm;
        }
    }
    return matrix;
}

std::vector<double> matvec(
    const std::vector<double>& matrix, const std::vector<double>& input) {
    const std::size_t count = input.size();
    std::vector<double> output(count, 0.0);
    for (std::size_t column = 0; column < count; ++column) {
        for (std::size_t row = 0; row < count; ++row) {
            output[row] += matrix[column * count + row] * input[column];
        }
    }
    return output;
}

void test_matrix_and_bit_order() {
    constexpr std::uint32_t qubits = 3;
    std::vector<std::uint64_t> states(1U << qubits);
    std::iota(states.begin(), states.end(), 0);
    const auto cals = calibration(qubits);
    for (std::uint32_t distance = 0; distance <= qubits; ++distance) {
        const auto actual = rem::build_reduced_matrix(
            states, states.size(), qubits, cals, distance);
        const auto expected = reference_matrix(states, qubits, cals, distance);
        CHECK(actual.size() == expected.size());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            check_close(actual[index], expected[index], 2e-14, "ratio matrix element");
        }
        for (std::size_t column = 0; column < states.size(); ++column) {
            double sum = 0.0;
            for (std::size_t row = 0; row < states.size(); ++row) {
                sum += actual[column * states.size() + row];
            }
            check_close(sum, 1.0, 2e-14, "column normalization");
        }
    }

    // q0 (LSB) has very different errors from q1.
    std::vector<double> asymmetric = {
        0.9, 0.2, 0.1, 0.8,
        0.6, 0.05, 0.4, 0.95,
    };
    std::vector<std::uint64_t> two_qubit_states = {0b00, 0b01, 0b10, 0b11};
    const auto actual =
        rem::build_reduced_matrix(two_qubit_states, 4, 2, asymmetric, 2);
    const auto expected = reference_matrix(two_qubit_states, 2, asymmetric, 2);
    check_close(actual[0 * 4 + 1], expected[0 * 4 + 1], 1e-14, "LSB q0 mapping");
    check_close(actual[0 * 4 + 2], expected[0 * 4 + 2], 1e-14, "MSB q1 mapping");
    CHECK(std::abs(actual[1] - actual[2]) > 1e-3);
}

void test_backends_and_solvers() {
    constexpr std::uint32_t qubits = 4;
    std::vector<std::uint64_t> states(1U << qubits);
    std::iota(states.begin(), states.end(), 0);
    const auto cals = calibration(qubits);
    constexpr std::uint32_t distance = 3;
    const auto matrix = reference_matrix(states, qubits, cals, distance);
    std::vector<double> ideal(states.size());
    for (std::size_t i = 0; i < ideal.size(); ++i) {
        ideal[i] = 1.0 + static_cast<double>((i * 7U) % 11U);
    }
    const double ideal_sum = std::accumulate(ideal.begin(), ideal.end(), 0.0);
    for (double& value : ideal) value /= ideal_sum;
    const auto noisy = matvec(matrix, ideal);

    struct Configuration {
        rem::Backend backend;
        rem::Solver solver;
        rem::Orthogonalization orthogonalization;
    };
    std::vector<Configuration> configurations = {
        {rem::Backend::Dense, rem::Solver::Direct, rem::Orthogonalization::Auto},
        {rem::Backend::ExplicitSparse, rem::Solver::GMRES,
         rem::Orthogonalization::ModifiedGramSchmidt},
        {rem::Backend::ExplicitSparse, rem::Solver::GMRES,
         rem::Orthogonalization::ClassicalGramSchmidt},
        {rem::Backend::MatrixFree, rem::Solver::GMRES,
         rem::Orthogonalization::ModifiedGramSchmidt},
        {rem::Backend::ExplicitSparse, rem::Solver::BiCGSTAB,
         rem::Orthogonalization::Auto},
        {rem::Backend::MatrixFree, rem::Solver::BiCGSTAB,
         rem::Orthogonalization::Auto},
    };
    for (const auto [backend, solver, orthogonalization] : configurations) {
        rem::Options options;
        options.backend = backend;
        options.solver = solver;
        options.max_hamming_distance = distance;
        options.relative_tolerance = 1e-11;
        options.gmres_restart = 12;
        options.max_iterations = 100;
        options.threads = 2;
        options.cache_values = true;
        options.orthogonalization = orthogonalization;
        const auto result =
            rem::mitigate(states, states.size(), qubits, noisy, cals, options);
        if (!result.diagnostics.converged) {
            std::cerr << "nonconverged backend=" << rem::to_string(backend)
                      << " solver=" << rem::to_string(solver)
                      << " iterations=" << result.diagnostics.iterations
                      << " residual=" << result.diagnostics.residual_norm
                      << " warning=" << result.diagnostics.warning << '\n';
        }
        CHECK(result.diagnostics.converged);
        CHECK(result.quasi_probabilities.size() == ideal.size());
        for (std::size_t i = 0; i < ideal.size(); ++i) {
            check_close(
                result.quasi_probabilities[i], ideal[i],
                solver == rem::Solver::Direct ? 2e-12 : 2e-9, "backend solve");
        }
        CHECK(result.diagnostics.residual_norm < 1e-9);
    }

    rem::Options forced_matrix_free;
    forced_matrix_free.backend = rem::Backend::MatrixFree;
    forced_matrix_free.solver = rem::Solver::GMRES;
    forced_matrix_free.graph_strategy = rem::GraphStrategy::BlockedPairwise;
    forced_matrix_free.max_hamming_distance = distance;
    forced_matrix_free.relative_tolerance = 1e-10;
    forced_matrix_free.threads = 2;
    const auto forced_scan =
        rem::mitigate(states, states.size(), qubits, noisy, cals, forced_matrix_free);
    CHECK(
        forced_scan.diagnostics.selected_graph_strategy ==
        rem::GraphStrategy::BlockedPairwise);
    CHECK(forced_scan.diagnostics.converged);
}

void test_duplicates_d0_and_projection() {
    std::vector<std::uint64_t> states = {0, 0, 1};
    std::vector<double> raw = {0.2, 0.3, 0.5};
    const auto cals = calibration(1);
    rem::Options options;
    options.backend = rem::Backend::Dense;
    options.solver = rem::Solver::Direct;
    options.max_hamming_distance = 0;
    const auto result = rem::mitigate(states, states.size(), 1, raw, cals, options);
    CHECK(result.diagnostics.unique_states == 2);
    CHECK(result.original_to_unique == std::vector<std::uint32_t>({0, 0, 1}));
    check_close(result.quasi_probabilities[0], 0.5, 1e-14, "duplicate aggregation 0");
    check_close(result.quasi_probabilities[1], 0.5, 1e-14, "duplicate aggregation 1");

    const std::vector<double> quasi = {-0.2, 0.3, 0.9};
    const auto projected = rem::project_probability_simplex(quasi);
    CHECK(std::all_of(projected.begin(), projected.end(), [](double x) { return x >= 0; }));
    check_close(
        std::accumulate(projected.begin(), projected.end(), 0.0), 1.0, 1e-14,
        "projection sum");
    check_close(projected[0], 0.0, 1e-14, "projection clipped");
}

void test_safe_ratio_fallback() {
    std::vector<std::uint64_t> states = {0, 1};
    // True 0 always reads as 1, so C[0][0][0] is zero.
    std::vector<double> cals = {0.0, 0.1, 1.0, 0.9};
    std::vector<double> raw = {0.1, 0.9};
    rem::Options options;
    options.backend = rem::Backend::Dense;
    options.solver = rem::Solver::Direct;
    options.max_hamming_distance = 1;
    const auto result = rem::mitigate(states, 2, 1, raw, cals, options);
    CHECK(result.diagnostics.used_safe_element_fallback);
    CHECK(result.diagnostics.jacobi_had_zero_diagonal);
    CHECK(result.diagnostics.converged);
    CHECK(std::all_of(
        result.quasi_probabilities.begin(), result.quasi_probabilities.end(),
        [](double value) { return std::isfinite(value); }));
}

void test_single_state_zero_raw_and_near_singular() {
    {
        const std::vector<std::uint64_t> states = {0};
        const std::vector<double> raw = {0.0};
        const auto cals = calibration(1);
        rem::Options options;
        options.backend = rem::Backend::Dense;
        options.solver = rem::Solver::Direct;
        const auto result = rem::mitigate(states, 1, 1, raw, cals, options);
        CHECK(result.diagnostics.converged);
        check_close(result.quasi_probabilities.front(), 0.0, 0.0, "single zero state");
    }

    const std::vector<std::uint64_t> states = {0, 1};
    const std::vector<double> cals = {
        0.5000001, 0.4999999,
        0.4999999, 0.5000001,
    };
    const auto matrix = reference_matrix(states, 1, cals, 1);
    const std::vector<double> ideal = {0.75, 0.25};
    const auto raw = matvec(matrix, ideal);
    rem::Options options;
    options.backend = rem::Backend::Dense;
    options.solver = rem::Solver::Direct;
    options.max_hamming_distance = 1;
    options.relative_tolerance = 1e-8;
    const auto result = rem::mitigate(states, 2, 1, raw, cals, options);
    CHECK(result.diagnostics.converged);
    check_close(result.quasi_probabilities[0], ideal[0], 2e-10, "near singular p0");
    check_close(result.quasi_probabilities[1], ideal[1], 2e-10, "near singular p1");
}

void test_generic_more_than_64_qubits() {
    constexpr std::uint32_t qubits = 65;
    constexpr std::size_t words = 2;
    std::vector<std::uint64_t> states = {
        0, 0,
        1, 0,
        0, 1,
        1, 1,
    };
    const auto cals = calibration(qubits);
    const auto matrix =
        rem::build_reduced_matrix(states, states.size() / words, qubits, cals, 1);
    CHECK(matrix.size() == 16);
    for (std::size_t column = 0; column < 4; ++column) {
        double sum = 0.0;
        for (std::size_t row = 0; row < 4; ++row) sum += matrix[column * 4 + row];
        check_close(sum, 1.0, 1e-13, "generic column sum");
    }
    check_close(matrix[0 * 4 + 3], 0.0, 0.0, "generic Hamming cutoff");

    // Exercise allocation-free generic Hamming-ball traversal in both storage modes.
    std::vector<std::uint64_t> ball_states;
    ball_states.reserve(128 * words);
    auto append_state = [&](std::uint64_t low, std::uint64_t high) {
        ball_states.push_back(low);
        ball_states.push_back(high);
    };
    append_state(0, 0);
    for (std::uint32_t q = 0; q < qubits; ++q) {
        append_state(q < 64 ? (1ULL << q) : 0, q == 64 ? 1 : 0);
    }
    std::mt19937_64 rng(65);
    while (ball_states.size() / words < 128) {
        const std::uint64_t low = rng();
        const std::uint64_t high = rng() & 1U;
        bool duplicate = false;
        for (std::size_t row = 0; row < ball_states.size() / words; ++row) {
            duplicate = duplicate ||
                        (ball_states[row * words] == low &&
                         ball_states[row * words + 1] == high);
        }
        if (!duplicate) append_state(low, high);
    }
    const std::size_t count = ball_states.size() / words;
    const auto ball_matrix =
        rem::build_reduced_matrix(ball_states, count, qubits, cals, 1);
    std::vector<double> ideal(count);
    for (std::size_t i = 0; i < count; ++i) ideal[i] = 1.0 + (i % 13U);
    const double sum = std::accumulate(ideal.begin(), ideal.end(), 0.0);
    for (double& value : ideal) value /= sum;
    const auto noisy = matvec(ball_matrix, ideal);
    for (const rem::Backend backend :
         {rem::Backend::ExplicitSparse, rem::Backend::MatrixFree}) {
        rem::Options generic_options;
        generic_options.backend = backend;
        generic_options.solver = rem::Solver::GMRES;
        generic_options.graph_strategy = rem::GraphStrategy::HammingBall;
        generic_options.sparse_storage = rem::SparseStorage::Valued;
        generic_options.max_hamming_distance = 1;
        generic_options.relative_tolerance = 1e-10;
        generic_options.max_iterations = 80;
        generic_options.threads = 2;
        const auto solved = rem::mitigate(
            ball_states, count, qubits, noisy, cals, generic_options);
        CHECK(solved.diagnostics.converged);
        for (std::size_t i = 0; i < count; ++i) {
            check_close(
                solved.quasi_probabilities[i], ideal[i], 2e-9,
                "generic iterative solve");
        }
    }
}

void test_batched_dense() {
    std::vector<std::uint64_t> states = {0, 1, 2, 3};
    const auto cals = calibration(2);
    const auto matrix = reference_matrix(states, 2, cals, 2);
    std::vector<double> p0 = {0.1, 0.2, 0.3, 0.4};
    std::vector<double> p1 = {0.4, 0.3, 0.2, 0.1};
    const auto q0 = matvec(matrix, p0);
    const auto q1 = matvec(matrix, p1);
    std::vector<double> rhs;
    rhs.insert(rhs.end(), q0.begin(), q0.end());
    rhs.insert(rhs.end(), q1.begin(), q1.end());
    rem::Options options;
    options.backend = rem::Backend::Dense;
    options.solver = rem::Solver::Direct;
    options.max_hamming_distance = 2;
    const auto results =
        rem::mitigate_many(states, 4, 2, rhs, 2, cals, options);
    CHECK(results.size() == 2);
    for (std::size_t i = 0; i < 4; ++i) {
        check_close(results[0].quasi_probabilities[i], p0[i], 1e-13, "batched rhs0");
        check_close(results[1].quasi_probabilities[i], p1[i], 1e-13, "batched rhs1");
    }
}

void test_memory_budget_selection() {
    constexpr std::uint32_t qubits = 16;
    constexpr std::size_t count = 512;
    std::vector<std::uint64_t> states(count);
    std::iota(states.begin(), states.end(), 0);
    std::vector<double> raw(count, 1.0 / static_cast<double>(count));
    const auto cals = calibration(qubits);
    rem::Options options;
    options.max_hamming_distance = 0;
    options.memory_budget_bytes = 1024U * 1024U;
    options.backend = rem::Backend::Auto;
    const auto result =
        rem::mitigate(states, count, qubits, raw, cals, options);
    CHECK(result.diagnostics.selected_backend != rem::Backend::Dense);
    CHECK(result.diagnostics.estimated_memory_bytes <= options.memory_budget_bytes);
    CHECK(result.diagnostics.converged);

    options.backend = rem::Backend::Dense;
    bool rejected = false;
    try {
        (void)rem::mitigate(states, count, qubits, raw, cals, options);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    CHECK(rejected);

    options.backend = rem::Backend::Auto;
    options.gmres_restart = static_cast<std::uint32_t>(count);
    bool no_backend_fits = false;
    try {
        (void)rem::mitigate(states, count, qubits, raw, cals, options);
    } catch (const std::runtime_error&) {
        no_backend_fits = true;
    }
    CHECK(no_backend_fits);
}

void test_ghz_and_determinism() {
    constexpr std::uint32_t qubits = 8;
    constexpr std::size_t count = 1U << qubits;
    std::vector<std::uint64_t> states(count);
    std::iota(states.begin(), states.end(), 0);
    const auto cals = calibration(qubits);
    const auto matrix = reference_matrix(states, qubits, cals, 3);
    std::vector<double> ideal(count, 0.0);
    ideal.front() = 0.5;
    ideal.back() = 0.5;
    const auto noisy = matvec(matrix, ideal);

    rem::Options options;
    options.backend = rem::Backend::ExplicitSparse;
    options.solver = rem::Solver::GMRES;
    options.sparse_storage = rem::SparseStorage::Valued;
    options.max_hamming_distance = 3;
    options.relative_tolerance = 1e-11;
    options.threads = 2;
    options.project_to_probability_simplex = true;
    const auto first =
        rem::mitigate(states, count, qubits, noisy, cals, options);
    const auto second =
        rem::mitigate(states, count, qubits, noisy, cals, options);
    CHECK(first.diagnostics.converged);
    CHECK(first.quasi_probabilities == second.quasi_probabilities);
    CHECK(first.projected_probabilities.has_value());
    check_close(first.quasi_probabilities.front(), 0.5, 2e-9, "GHZ zero state");
    check_close(first.quasi_probabilities.back(), 0.5, 2e-9, "GHZ one state");
    const double bhattacharyya =
        std::sqrt(first.projected_probabilities->front() * 0.5) +
        std::sqrt(first.projected_probabilities->back() * 0.5);
    check_close(
        bhattacharyya * bhattacharyya, 1.0, 2e-9,
        "GHZ Hellinger fidelity");
}

}  // namespace

int main() {
    try {
        test_matrix_and_bit_order();
        test_backends_and_solvers();
        test_duplicates_d0_and_projection();
        test_safe_ratio_fallback();
        test_single_state_zero_raw_and_near_singular();
        test_generic_more_than_64_qubits();
        test_batched_dense();
        test_memory_budget_selection();
        test_ghz_and_determinism();
    } catch (const std::exception& error) {
        std::cerr << "uncaught exception: " << error.what() << '\n';
        return 2;
    }
    if (failures != 0) {
        std::cerr << failures << " test checks failed\n";
        return 1;
    }
    std::cout << "all REM C++ tests passed\n";
    return 0;
}
