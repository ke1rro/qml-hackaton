#include "detail.hpp"

namespace rem::detail {
namespace {

constexpr std::uint32_t kMissing = std::numeric_limits<std::uint32_t>::max();

template <class Callback>
void visit_indexed_neighbors(
    const PackedStates& states,
    const StateIndex& index,
    std::size_t row,
    std::uint32_t distance,
    std::span<std::uint64_t> scratch,
    Callback&& callback) {
    if (states.words_per_state == 1 && distance <= 3) {
        const std::uint64_t value = states.words[row];
        for_each_ball_candidate_64(
            value, states.qubits, distance, [&](std::uint64_t candidate) {
                const std::span<const std::uint64_t> view(&candidate, 1);
                const std::uint32_t found = index.find(view);
                if (found != kMissing) {
                    callback(found);
                }
            });
        return;
    }
    for_each_ball_candidate_generic_scratch(
        states.state(row), states.qubits, distance, scratch,
        [&](std::span<const std::uint64_t> candidate) {
            const std::uint32_t found = index.find(candidate);
            if (found != kMissing) {
                callback(found);
            }
        });
}

std::pair<double, std::uint64_t> sample_density(
    const PackedStates& states, std::uint32_t distance) {
    const std::size_t count = states.size();
    if (count == 0) {
        return {0.0, 0};
    }
    const std::size_t samples = std::min<std::size_t>(count, 128);
    std::uint64_t retained = 0;
    std::uint64_t tested = 0;
    for (std::size_t sample = 0; sample < samples; ++sample) {
        const std::size_t row =
            samples == count ? sample : sample * (count - 1) / (samples - 1);
        for (std::size_t column = 0; column < count; ++column) {
            ++tested;
            if (hamming_distance(states.state(row), states.state(column), distance) <=
                distance) {
                ++retained;
            }
        }
    }
    const double density =
        tested == 0 ? 0.0 : static_cast<double>(retained) / static_cast<double>(tested);
    const long double projected =
        static_cast<long double>(density) * static_cast<long double>(count) *
        static_cast<long double>(count);
    const auto projected_edges = projected >=
                                         static_cast<long double>(
                                             std::numeric_limits<std::uint64_t>::max())
                                     ? std::numeric_limits<std::uint64_t>::max()
                                     : static_cast<std::uint64_t>(std::ceil(projected));
    return {density, std::max<std::uint64_t>(projected_edges, count)};
}

void compute_jacobi(AssignmentOperator& op, Diagnostics& diagnostics, double threshold) {
    const std::size_t count = op.size();
    op.jacobi_inverse.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        const double diagonal = op.calibration->base[index] / op.column_norms[index];
        if (!std::isfinite(diagonal) || std::abs(diagonal) <= threshold) {
            op.jacobi_inverse[index] = 1.0;
            diagnostics.jacobi_had_zero_diagonal = true;
        } else {
            op.jacobi_inverse[index] = 1.0 / diagonal;
        }
    }
    if (diagnostics.jacobi_had_zero_diagonal) {
        diagnostics.warning =
            "one or more normalized diagonal entries were near zero; "
            "identity scaling was used for those Jacobi components";
    }
}

}  // namespace

Calibration prepare_calibration(
    const PackedStates& states,
    std::span<const double> matrices,
    double near_zero,
    ThreadPool& pool,
    Diagnostics& diagnostics) {
    if (matrices.size() != static_cast<std::size_t>(states.qubits) * 4U) {
        throw std::invalid_argument("confusion matrix buffer must have shape [Q,2,2]");
    }
    if (!(near_zero > 0.0) || !std::isfinite(near_zero)) {
        throw std::invalid_argument("near_zero_threshold must be finite and positive");
    }
    Calibration calibration;
    calibration.qubits = states.qubits;
    calibration.values.assign(matrices.begin(), matrices.end());
    calibration.ratios.resize(static_cast<std::size_t>(states.qubits) * 2U);
    calibration.near_zero = near_zero;
    for (std::uint32_t q = 0; q < states.qubits; ++q) {
        for (std::uint32_t measured = 0; measured < 2; ++measured) {
            for (std::uint32_t truth = 0; truth < 2; ++truth) {
                const double value =
                    calibration.values[static_cast<std::size_t>(q) * 4U +
                                       measured * 2U + truth];
                if (!std::isfinite(value) || value < 0.0) {
                    throw std::invalid_argument(
                        "confusion matrix entries must be finite and nonnegative");
                }
            }
        }
        for (std::uint32_t truth = 0; truth < 2; ++truth) {
            const double diagonal =
                calibration.values[static_cast<std::size_t>(q) * 4U + truth * 2U +
                                   truth];
            const double flipped =
                calibration.values[static_cast<std::size_t>(q) * 4U +
                                   (1U - truth) * 2U + truth];
            calibration.ratios[static_cast<std::size_t>(q) * 2U + truth] =
                std::abs(diagonal) <= near_zero
                    ? std::numeric_limits<double>::quiet_NaN()
                    : flipped / diagonal;
        }
    }

    calibration.base.resize(states.size());
    calibration.ratio_safe.resize(states.size(), 1);
    std::atomic<bool> any_fallback = false;
    pool.parallel_for(states.size(), [&](std::size_t begin, std::size_t end, std::uint32_t) {
        for (std::size_t state_index = begin; state_index < end; ++state_index) {
            double base = 1.0;
            bool safe = true;
            for (std::uint32_t q = 0; q < states.qubits; ++q) {
                const std::uint32_t bit = states.bit(state_index, q) ? 1U : 0U;
                const double diagonal =
                    calibration.values[static_cast<std::size_t>(q) * 4U + bit * 2U +
                                       bit];
                base *= diagonal;
                const double ratio =
                    calibration.ratios[static_cast<std::size_t>(q) * 2U + bit];
                safe = safe && std::isfinite(ratio) && std::abs(diagonal) > near_zero;
            }
            calibration.base[state_index] = base;
            calibration.ratio_safe[state_index] = safe ? 1U : 0U;
            if (!safe) {
                any_fallback.store(true, std::memory_order_relaxed);
            }
        }
    });
    diagnostics.used_safe_element_fallback = any_fallback.load(std::memory_order_relaxed);
    return calibration;
}

double direct_element(
    const PackedStates& states,
    const Calibration& calibration,
    std::uint32_t target,
    std::uint32_t source) noexcept {
    double value = 1.0;
    for (std::uint32_t q = 0; q < states.qubits; ++q) {
        const std::uint32_t measured = states.bit(target, q) ? 1U : 0U;
        const std::uint32_t truth = states.bit(source, q) ? 1U : 0U;
        value *= calibration.values[static_cast<std::size_t>(q) * 4U +
                                    measured * 2U + truth];
    }
    return value;
}

double fast_element(
    const PackedStates& states,
    const Calibration& calibration,
    std::uint32_t target,
    std::uint32_t source) noexcept {
    if (calibration.ratio_safe[source] == 0) {
        return direct_element(states, calibration, target, source);
    }
    double value = calibration.base[source];
    const auto target_state = states.state(target);
    const auto source_state = states.state(source);
    for (std::size_t word_index = 0; word_index < states.words_per_state; ++word_index) {
        std::uint64_t changed = target_state[word_index] ^ source_state[word_index];
        while (changed != 0) {
            const auto local_q = static_cast<std::uint32_t>(std::countr_zero(changed));
            const auto q = static_cast<std::uint32_t>(word_index * 64U + local_q);
            const std::uint32_t truth = (source_state[word_index] >> local_q) & 1U;
            value *= calibration.ratios[static_cast<std::size_t>(q) * 2U + truth];
            changed &= changed - 1U;
        }
    }
    return value;
}

MemoryEstimate estimate_memory(
    const PackedStates& states,
    std::uint32_t distance,
    const Options& options,
    std::size_t right_hand_sides) {
    MemoryEstimate estimate;
    const std::uint64_t count = states.size();
    const auto [density, projected_edges] =
        sample_density(states, std::min(distance, states.qubits));
    estimate.sampled_density = density;
    estimate.projected_edges = projected_edges;
    const std::uint64_t effective_restart =
        std::min<std::uint64_t>(count, options.gmres_restart);
    estimate.krylov = saturating_mul(
        saturating_mul(count, saturating_add(effective_restart, 10U)),
        sizeof(double));
    const std::uint64_t extra_rhs = right_hand_sides > 0 ? right_hand_sides - 1U : 0U;
    const std::uint64_t batch_vectors = saturating_mul(
        saturating_mul(count, extra_rhs),
        (options.project_to_probability_simplex ? 4U : 3U) * sizeof(double));
    const std::uint64_t base_common = saturating_add(
        saturating_mul(
            count, states.words_per_state * sizeof(std::uint64_t) + 48U),
        batch_vectors);
    const std::uint64_t iterative_common =
        saturating_add(base_common, estimate.krylov);
    estimate.dense = saturating_add(
        saturating_add(
            base_common,
            saturating_mul(count, 3U * sizeof(double) + sizeof(std::int32_t))),
        saturating_mul(saturating_mul(count, count), sizeof(double)));
    const std::uint32_t estimated_threads =
        options.threads == 0
            ? std::max<std::uint32_t>(1, std::thread::hardware_concurrency())
            : options.threads;
    const std::uint64_t pairwise_setup_scratch = saturating_mul(
        saturating_mul(count, estimated_threads),
        sizeof(std::uint32_t) + sizeof(std::uint64_t));
    estimate.topology_csr = saturating_add(
        saturating_add(iterative_common, pairwise_setup_scratch),
        saturating_add(
            saturating_mul(projected_edges, sizeof(std::uint32_t)),
            saturating_mul(count + 1U, sizeof(std::uint64_t))));
    estimate.valued_csr = saturating_add(
        estimate.topology_csr, saturating_mul(projected_edges, sizeof(double)));
    estimate.matrix_free = iterative_common;
    return estimate;
}

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
    Diagnostics& diagnostics) {
    AssignmentOperator op;
    op.states = &states;
    op.state_index = &index;
    op.calibration = &calibration;
    op.pool = &pool;
    op.distance = std::min(options.max_hamming_distance, states.qubits);
    const std::size_t count = states.size();

    if (backend == Backend::Dense) {
        op.storage = OperatorStorage::Dense;
        if (count != 0 &&
            count > std::numeric_limits<std::size_t>::max() / count) {
            throw std::length_error("dense assignment matrix size overflows size_t");
        }
        op.dense_column_major.assign(count * count, 0.0);
        op.column_norms.resize(count);
        std::vector<std::uint64_t> retained_per_column(count, 0);
        constexpr std::size_t kColumnTile = 32;
        const std::size_t tiles = (count + kColumnTile - 1U) / kColumnTile;
        pool.parallel_for(tiles, [&](std::size_t begin, std::size_t end, std::uint32_t) {
            for (std::size_t tile = begin; tile < end; ++tile) {
                const std::size_t first_column = tile * kColumnTile;
                const std::size_t last_column =
                    std::min(count, first_column + kColumnTile);
                for (std::size_t column = first_column; column < last_column; ++column) {
                    double norm = 0.0;
                    std::uint64_t retained = 0;
                    double* values = op.dense_column_major.data() + column * count;
                    for (std::size_t row = 0; row < count; ++row) {
                        if (hamming_distance(
                                states.state(row), states.state(column), op.distance) <=
                            op.distance) {
                            const double value = fast_element(
                                states, calibration, static_cast<std::uint32_t>(row),
                                static_cast<std::uint32_t>(column));
                            values[row] = value;
                            norm += value;
                            ++retained;
                        }
                    }
                    op.column_norms[column] = norm;
                    retained_per_column[column] = retained;
                }
            }
        });
        for (const double norm : op.column_norms) {
            if (!(norm > 0.0) || !std::isfinite(norm)) {
                throw std::runtime_error(
                    "a reduced assignment column has a nonpositive norm");
            }
        }
        pool.parallel_for(count, [&](std::size_t begin, std::size_t end, std::uint32_t) {
            for (std::size_t column = begin; column < end; ++column) {
                double* values = op.dense_column_major.data() + column * count;
                const double norm = op.column_norms[column];
                for (std::size_t row = 0; row < count; ++row) {
                    values[row] /= norm;
                }
            }
        });
        diagnostics.retained_edges = 0;
        for (const auto value : retained_per_column) {
            diagnostics.retained_edges += value;
        }
    } else if (backend == Backend::ExplicitSparse) {
        op.graph = build_hamming_graph(
            states, index, op.distance, graph_strategy, pool,
            options.memory_budget_bytes, diagnostics);
        const bool valued = sparse_storage == SparseStorage::Valued;
        const std::uint64_t edges = op.graph.row_offsets.back();
        const std::uint64_t graph_bytes = saturating_add(
            saturating_mul(count + 1U, sizeof(std::uint64_t)),
            saturating_mul(
                edges, sizeof(std::uint32_t) + (valued ? sizeof(double) : 0U)));
        const std::uint64_t restart =
            std::min<std::uint64_t>(count, options.gmres_restart);
        const std::uint64_t solver_bytes = saturating_mul(
            saturating_mul(count, saturating_add(restart, 10U)),
            sizeof(double));
        std::uint64_t common_bytes = saturating_mul(
            count,
            states.words_per_state * sizeof(std::uint64_t) + 4U * sizeof(double) +
                sizeof(std::uint8_t));
        const std::uint64_t extra_rhs =
            right_hand_sides > 0 ? right_hand_sides - 1U : 0U;
        common_bytes = saturating_add(
            common_bytes,
            saturating_mul(
                saturating_mul(count, extra_rhs),
                (options.project_to_probability_simplex ? 4U : 3U) *
                    sizeof(double)));
        const std::uint64_t exact_minimum =
            saturating_add(saturating_add(graph_bytes, solver_bytes), common_bytes);
        if (exact_minimum > options.memory_budget_bytes) {
            throw std::runtime_error(
                "exact sparse operator and solver workspace exceed the "
                "configured memory budget");
        }
        diagnostics.estimated_memory_bytes =
            std::max(diagnostics.estimated_memory_bytes, exact_minimum);
        diagnostics.peak_working_memory_bytes =
            std::max(diagnostics.peak_working_memory_bytes, exact_minimum);
        op.column_norms.resize(count);
        pool.parallel_for(count, [&](std::size_t begin, std::size_t end, std::uint32_t) {
            for (std::size_t source = begin; source < end; ++source) {
                double norm = 0.0;
                for (std::uint64_t edge = op.graph.row_offsets[source];
                     edge < op.graph.row_offsets[source + 1]; ++edge) {
                    const std::uint32_t target =
                        op.graph.column_indices[static_cast<std::size_t>(edge)];
                    norm += fast_element(
                        states, calibration, target, static_cast<std::uint32_t>(source));
                }
                op.column_norms[source] = norm;
            }
        });
        for (const double norm : op.column_norms) {
            if (!(norm > 0.0) || !std::isfinite(norm)) {
                throw std::runtime_error(
                    "a reduced assignment column has a nonpositive norm");
            }
        }
        op.storage = valued ? OperatorStorage::ValuedCsr : OperatorStorage::TopologyCsr;
        if (valued) {
            op.csr_values.resize(op.graph.column_indices.size());
            pool.parallel_for(count, [&](std::size_t begin, std::size_t end, std::uint32_t) {
                for (std::size_t row = begin; row < end; ++row) {
                    for (std::uint64_t edge = op.graph.row_offsets[row];
                         edge < op.graph.row_offsets[row + 1]; ++edge) {
                        const std::uint32_t column =
                            op.graph.column_indices[static_cast<std::size_t>(edge)];
                        op.csr_values[static_cast<std::size_t>(edge)] =
                            fast_element(
                                states, calibration, static_cast<std::uint32_t>(row),
                                column) /
                            op.column_norms[column];
                    }
                }
            });
        }
    } else {
        const long double neighbor_work =
            static_cast<long double>(hamming_ball_size(states.qubits, op.distance)) *
            4.0L;
        const long double scan_work =
            static_cast<long double>(count) *
            static_cast<long double>(states.words_per_state);
        const bool use_ball =
            graph_strategy == GraphStrategy::HammingBall ||
            (graph_strategy == GraphStrategy::Auto && neighbor_work < scan_work);
        op.storage =
            use_ball ? OperatorStorage::MatrixFreeBall : OperatorStorage::MatrixFreeScan;
        diagnostics.selected_graph_strategy =
            use_ball ? GraphStrategy::HammingBall : GraphStrategy::BlockedPairwise;
        op.column_norms.resize(count);
        if (use_ball) {
            op.matrix_free_scratch.resize(
                static_cast<std::size_t>(pool.size()) * states.words_per_state);
        }
        std::vector<std::uint64_t> retained_per_source(count, 0);
        pool.parallel_for(count, [&](std::size_t begin, std::size_t end, std::uint32_t tid) {
            std::span<std::uint64_t> thread_scratch;
            if (use_ball) {
                thread_scratch = std::span<std::uint64_t>(
                    op.matrix_free_scratch.data() +
                        static_cast<std::size_t>(tid) * states.words_per_state,
                    states.words_per_state);
            }
            for (std::size_t source = begin; source < end; ++source) {
                double norm = 0.0;
                std::uint64_t retained = 0;
                if (use_ball) {
                    visit_indexed_neighbors(
                        states, index, source, op.distance, thread_scratch,
                        [&](std::uint32_t target) {
                            norm += fast_element(
                                states, calibration, target,
                                static_cast<std::uint32_t>(source));
                            ++retained;
                        });
                } else {
                    for (std::size_t target = 0; target < count; ++target) {
                        if (hamming_distance(
                                states.state(target), states.state(source), op.distance) <=
                            op.distance) {
                            norm += fast_element(
                                states, calibration, static_cast<std::uint32_t>(target),
                                static_cast<std::uint32_t>(source));
                            ++retained;
                        }
                    }
                }
                op.column_norms[source] = norm;
                retained_per_source[source] = retained;
            }
        });
        for (const double norm : op.column_norms) {
            if (!(norm > 0.0) || !std::isfinite(norm)) {
                throw std::runtime_error(
                    "a reduced assignment column has a nonpositive norm");
            }
        }
        diagnostics.retained_edges = 0;
        for (const auto value : retained_per_source) {
            diagnostics.retained_edges += value;
        }
    }
    compute_jacobi(op, diagnostics, options.near_zero_threshold);
    return op;
}

void AssignmentOperator::matvec(
    std::span<const double> input, std::span<double> output) const {
    const std::size_t count = size();
    if (input.size() != count || output.size() != count) {
        throw std::invalid_argument("matvec vector length mismatch");
    }
    if (storage == OperatorStorage::Dense) {
        pool->parallel_for(count, [&](std::size_t begin, std::size_t end, std::uint32_t) {
            for (std::size_t row = begin; row < end; ++row) {
                double sum = 0.0;
                for (std::size_t column = 0; column < count; ++column) {
                    sum += dense_column_major[column * count + row] * input[column];
                }
                output[row] = sum;
            }
        });
        return;
    }
    if (storage == OperatorStorage::TopologyCsr ||
        storage == OperatorStorage::ValuedCsr) {
        pool->parallel_for(count, [&](std::size_t begin, std::size_t end, std::uint32_t) {
            for (std::size_t row = begin; row < end; ++row) {
                double sum = 0.0;
                for (std::uint64_t edge = graph.row_offsets[row];
                     edge < graph.row_offsets[row + 1]; ++edge) {
                    const std::uint32_t column =
                        graph.column_indices[static_cast<std::size_t>(edge)];
                    const double value =
                        storage == OperatorStorage::ValuedCsr
                            ? csr_values[static_cast<std::size_t>(edge)]
                            : fast_element(
                                  *states, *calibration, static_cast<std::uint32_t>(row),
                                  column) /
                                  column_norms[column];
                    sum += value * input[column];
                }
                output[row] = sum;
            }
        });
        return;
    }
    if (storage == OperatorStorage::MatrixFreeBall) {
        pool->parallel_for(count, [&](std::size_t begin, std::size_t end, std::uint32_t tid) {
            std::span<std::uint64_t> thread_scratch(
                matrix_free_scratch.data() +
                    static_cast<std::size_t>(tid) * states->words_per_state,
                states->words_per_state);
            for (std::size_t row = begin; row < end; ++row) {
                double sum = 0.0;
                visit_indexed_neighbors(
                    *states, *state_index, row, distance, thread_scratch,
                    [&](std::uint32_t column) {
                        sum += fast_element(
                                   *states, *calibration, static_cast<std::uint32_t>(row),
                                   column) /
                               column_norms[column] * input[column];
                    });
                output[row] = sum;
            }
        });
        return;
    }
    pool->parallel_for(count, [&](std::size_t begin, std::size_t end, std::uint32_t) {
        for (std::size_t row = begin; row < end; ++row) {
            double sum = 0.0;
            for (std::size_t column = 0; column < count; ++column) {
                if (hamming_distance(
                        states->state(row), states->state(column), distance) <= distance) {
                    sum += fast_element(
                               *states, *calibration, static_cast<std::uint32_t>(row),
                               static_cast<std::uint32_t>(column)) /
                           column_norms[column] * input[column];
                }
            }
            output[row] = sum;
        }
    });
}

}  // namespace rem::detail
