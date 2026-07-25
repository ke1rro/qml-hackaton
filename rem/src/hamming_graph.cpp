#include "detail.hpp"

namespace rem::detail {
namespace {

constexpr std::uint32_t kMissing = std::numeric_limits<std::uint32_t>::max();

void ensure_topology_fits(
    std::size_t rows, std::uint64_t edges, std::uint64_t memory_budget) {
    const std::uint64_t offset_bytes = saturating_mul(
        static_cast<std::uint64_t>(rows) + 1U, sizeof(std::uint64_t));
    const std::uint64_t index_bytes =
        saturating_mul(edges, sizeof(std::uint32_t));
    if (saturating_add(offset_bytes, index_bytes) > memory_budget) {
        throw std::runtime_error(
            "exact CSR topology exceeds the configured memory budget");
    }
}

template <class Callback>
void visit_neighbors(
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

CsrGraph build_ball(
    const PackedStates& states,
    const StateIndex& index,
    std::uint32_t distance,
    ThreadPool& pool,
    std::uint64_t memory_budget) {
    const std::size_t count = states.size();
    std::vector<std::uint64_t> row_counts(count, 0);
    std::vector<std::uint64_t> scratch(
        static_cast<std::size_t>(pool.size()) * states.words_per_state);
    pool.parallel_for(count, [&](std::size_t begin, std::size_t end, std::uint32_t tid) {
        std::span<std::uint64_t> thread_scratch(
            scratch.data() + static_cast<std::size_t>(tid) * states.words_per_state,
            states.words_per_state);
        for (std::size_t row = begin; row < end; ++row) {
            std::uint64_t terms = 0;
            visit_neighbors(
                states, index, row, distance, thread_scratch,
                [&](std::uint32_t) { ++terms; });
            row_counts[row] = terms;
        }
    });

    CsrGraph graph;
    graph.row_offsets.resize(count + 1, 0);
    for (std::size_t row = 0; row < count; ++row) {
        graph.row_offsets[row + 1] =
            saturating_add(graph.row_offsets[row], row_counts[row]);
    }
    if (graph.row_offsets.back() >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::length_error("CSR edge count exceeds addressable memory");
    }
    ensure_topology_fits(count, graph.row_offsets.back(), memory_budget);
    graph.column_indices.resize(static_cast<std::size_t>(graph.row_offsets.back()));
    pool.parallel_for(count, [&](std::size_t begin, std::size_t end, std::uint32_t tid) {
        std::span<std::uint64_t> thread_scratch(
            scratch.data() + static_cast<std::size_t>(tid) * states.words_per_state,
            states.words_per_state);
        for (std::size_t row = begin; row < end; ++row) {
            std::uint64_t cursor = graph.row_offsets[row];
            visit_neighbors(
                states, index, row, distance, thread_scratch,
                [&](std::uint32_t column) {
                    graph.column_indices[static_cast<std::size_t>(cursor++)] = column;
                });
        }
    });
    return graph;
}

CsrGraph build_pairwise(
    const PackedStates& states,
    std::uint32_t distance,
    ThreadPool& pool,
    std::uint64_t memory_budget) {
    const std::size_t count = states.size();
    const std::uint32_t threads = pool.size();
    const std::uint64_t local_bytes = saturating_mul(
        saturating_mul(threads, static_cast<std::uint64_t>(count)),
        sizeof(std::uint32_t));
    constexpr std::size_t kColumnTile = 256;

    if (local_bytes > memory_budget / 8U) {
        std::vector<std::uint64_t> row_counts(count, 1);
        for (std::size_t row = 0; row < count; ++row) {
            for (std::size_t block = row + 1; block < count; block += kColumnTile) {
                const std::size_t stop = std::min(count, block + kColumnTile);
                for (std::size_t column = block; column < stop; ++column) {
                    if (hamming_distance(states.state(row), states.state(column), distance) <=
                        distance) {
                        ++row_counts[row];
                        ++row_counts[column];
                    }
                }
            }
        }
        CsrGraph graph;
        graph.row_offsets.resize(count + 1, 0);
        for (std::size_t row = 0; row < count; ++row) {
            graph.row_offsets[row + 1] = graph.row_offsets[row] + row_counts[row];
        }
        ensure_topology_fits(count, graph.row_offsets.back(), memory_budget);
        graph.column_indices.resize(static_cast<std::size_t>(graph.row_offsets.back()));
        std::vector<std::uint64_t> cursor = graph.row_offsets;
        for (std::size_t row = 0; row < count; ++row) {
            graph.column_indices[static_cast<std::size_t>(cursor[row]++)] =
                static_cast<std::uint32_t>(row);
        }
        for (std::size_t row = 0; row < count; ++row) {
            for (std::size_t block = row + 1; block < count; block += kColumnTile) {
                const std::size_t stop = std::min(count, block + kColumnTile);
                for (std::size_t column = block; column < stop; ++column) {
                    if (hamming_distance(states.state(row), states.state(column), distance) <=
                        distance) {
                        graph.column_indices[static_cast<std::size_t>(cursor[row]++)] =
                            static_cast<std::uint32_t>(column);
                        graph.column_indices[static_cast<std::size_t>(cursor[column]++)] =
                            static_cast<std::uint32_t>(row);
                    }
                }
            }
        }
        return graph;
    }

    std::vector<std::uint32_t> local_counts(
        static_cast<std::size_t>(threads) * count, 0);
    pool.parallel_for(count, [&](std::size_t begin, std::size_t end, std::uint32_t tid) {
        auto* counts = local_counts.data() + static_cast<std::size_t>(tid) * count;
        for (std::size_t row = begin; row < end; ++row) {
            for (std::size_t block = row + 1; block < count; block += kColumnTile) {
                const std::size_t stop = std::min(count, block + kColumnTile);
                for (std::size_t column = block; column < stop; ++column) {
                    if (hamming_distance(states.state(row), states.state(column), distance) <=
                        distance) {
                        ++counts[row];
                        ++counts[column];
                    }
                }
            }
        }
    });

    CsrGraph graph;
    graph.row_offsets.resize(count + 1, 0);
    for (std::size_t row = 0; row < count; ++row) {
        std::uint64_t row_count = 1;
        for (std::uint32_t tid = 0; tid < threads; ++tid) {
            row_count += local_counts[static_cast<std::size_t>(tid) * count + row];
        }
        graph.row_offsets[row + 1] = graph.row_offsets[row] + row_count;
    }
    ensure_topology_fits(count, graph.row_offsets.back(), memory_budget);
    graph.column_indices.resize(static_cast<std::size_t>(graph.row_offsets.back()));
    std::vector<std::uint64_t> thread_cursors(
        static_cast<std::size_t>(threads) * count, 0);
    for (std::size_t row = 0; row < count; ++row) {
        graph.column_indices[static_cast<std::size_t>(graph.row_offsets[row])] =
            static_cast<std::uint32_t>(row);
        std::uint64_t cursor = graph.row_offsets[row] + 1;
        for (std::uint32_t tid = 0; tid < threads; ++tid) {
            thread_cursors[static_cast<std::size_t>(tid) * count + row] = cursor;
            cursor += local_counts[static_cast<std::size_t>(tid) * count + row];
        }
    }
    pool.parallel_for(count, [&](std::size_t begin, std::size_t end, std::uint32_t tid) {
        auto* cursor = thread_cursors.data() + static_cast<std::size_t>(tid) * count;
        for (std::size_t row = begin; row < end; ++row) {
            for (std::size_t block = row + 1; block < count; block += kColumnTile) {
                const std::size_t stop = std::min(count, block + kColumnTile);
                for (std::size_t column = block; column < stop; ++column) {
                    if (hamming_distance(states.state(row), states.state(column), distance) <=
                        distance) {
                        graph.column_indices[static_cast<std::size_t>(cursor[row]++)] =
                            static_cast<std::uint32_t>(column);
                        graph.column_indices[static_cast<std::size_t>(cursor[column]++)] =
                            static_cast<std::uint32_t>(row);
                    }
                }
            }
        }
    });
    return graph;
}

}  // namespace

GraphStrategy choose_graph_strategy(
    std::size_t states,
    std::uint32_t qubits,
    std::uint32_t distance,
    std::size_t words_per_state) {
    const long double pairwise =
        static_cast<long double>(states) * static_cast<long double>(states - 1) * 0.5L *
        static_cast<long double>(words_per_state);
    const long double neighbor =
        static_cast<long double>(states) *
        static_cast<long double>(hamming_ball_size(qubits, distance)) * 4.0L;
    return neighbor < pairwise ? GraphStrategy::HammingBall
                               : GraphStrategy::BlockedPairwise;
}

CsrGraph build_hamming_graph(
    const PackedStates& states,
    const StateIndex& index,
    std::uint32_t distance,
    GraphStrategy strategy,
    ThreadPool& pool,
    std::uint64_t memory_budget,
    Diagnostics& diagnostics) {
    distance = std::min(distance, states.qubits);
    if (strategy == GraphStrategy::Auto) {
        strategy = choose_graph_strategy(
            states.size(), states.qubits, distance, states.words_per_state);
    }
    diagnostics.selected_graph_strategy = strategy;
    CsrGraph graph =
        strategy == GraphStrategy::HammingBall
            ? build_ball(states, index, distance, pool, memory_budget)
            : build_pairwise(states, distance, pool, memory_budget);
    diagnostics.retained_edges = graph.row_offsets.back();
    return graph;
}

}  // namespace rem::detail
