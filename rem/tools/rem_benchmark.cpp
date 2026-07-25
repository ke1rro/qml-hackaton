#include "rem/api.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct Arguments {
    std::uint32_t qubits = 16;
    std::size_t states = 1000;
    std::uint32_t distance = 3;
    std::uint32_t threads = 1;
    std::size_t repetitions = 5;
    std::string dataset = "random";
    rem::Backend backend = rem::Backend::Auto;
    rem::Solver solver = rem::Solver::Auto;
    rem::GraphStrategy graph = rem::GraphStrategy::Auto;
    rem::SparseStorage storage = rem::SparseStorage::Auto;
    rem::Orthogonalization orthogonalization = rem::Orthogonalization::Auto;
    bool json = false;
};

std::string next_value(int& index, int argc, char** argv) {
    if (++index >= argc) throw std::invalid_argument("missing command-line value");
    return argv[index];
}

Arguments parse(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--q") args.qubits = std::stoul(next_value(i, argc, argv));
        else if (key == "--k") args.states = std::stoull(next_value(i, argc, argv));
        else if (key == "--d") args.distance = std::stoul(next_value(i, argc, argv));
        else if (key == "--threads") args.threads = std::stoul(next_value(i, argc, argv));
        else if (key == "--repetitions")
            args.repetitions = std::stoull(next_value(i, argc, argv));
        else if (key == "--dataset") args.dataset = next_value(i, argc, argv);
        else if (key == "--json") args.json = true;
        else if (key == "--backend") {
            const auto value = next_value(i, argc, argv);
            if (value == "auto") args.backend = rem::Backend::Auto;
            else if (value == "dense") args.backend = rem::Backend::Dense;
            else if (value == "sparse") args.backend = rem::Backend::ExplicitSparse;
            else if (value == "matrix_free") args.backend = rem::Backend::MatrixFree;
            else throw std::invalid_argument("invalid backend");
        } else if (key == "--solver") {
            const auto value = next_value(i, argc, argv);
            if (value == "auto") args.solver = rem::Solver::Auto;
            else if (value == "direct") args.solver = rem::Solver::Direct;
            else if (value == "gmres") args.solver = rem::Solver::GMRES;
            else if (value == "bicgstab") args.solver = rem::Solver::BiCGSTAB;
            else throw std::invalid_argument("invalid solver");
        } else if (key == "--graph") {
            const auto value = next_value(i, argc, argv);
            if (value == "auto") args.graph = rem::GraphStrategy::Auto;
            else if (value == "ball") args.graph = rem::GraphStrategy::HammingBall;
            else if (value == "pairwise")
                args.graph = rem::GraphStrategy::BlockedPairwise;
            else throw std::invalid_argument("invalid graph strategy");
        } else if (key == "--storage") {
            const auto value = next_value(i, argc, argv);
            if (value == "auto") args.storage = rem::SparseStorage::Auto;
            else if (value == "topology")
                args.storage = rem::SparseStorage::TopologyOnly;
            else if (value == "valued") args.storage = rem::SparseStorage::Valued;
            else throw std::invalid_argument("invalid sparse storage");
        } else if (key == "--orthogonalization") {
            const auto value = next_value(i, argc, argv);
            if (value == "auto")
                args.orthogonalization = rem::Orthogonalization::Auto;
            else if (value == "mgs")
                args.orthogonalization =
                    rem::Orthogonalization::ModifiedGramSchmidt;
            else if (value == "cgs")
                args.orthogonalization =
                    rem::Orthogonalization::ClassicalGramSchmidt;
            else
                throw std::invalid_argument("invalid orthogonalization");
        } else if (key == "--help") {
            std::cout
                << "rem_benchmark [--q N] [--k N] [--d N] [--threads N]\n"
                   "  [--dataset random|ghz|clustered] [--backend auto|dense|sparse|matrix_free]\n"
                   "  [--solver auto|direct|gmres|bicgstab] [--graph auto|ball|pairwise]\n"
                   "  [--storage auto|topology|valued] [--orthogonalization auto|mgs|cgs]\n"
                   "  [--repetitions N] [--json]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    return args;
}

std::vector<std::uint64_t> make_states(const Arguments& args) {
    if (args.qubits == 0 || args.qubits > 64) {
        throw std::invalid_argument("benchmark generator currently supports 1 <= Q <= 64");
    }
    const std::uint64_t mask =
        args.qubits == 64 ? ~0ULL : ((1ULL << args.qubits) - 1ULL);
    if (args.qubits < 64 && args.states > (1ULL << args.qubits)) {
        throw std::invalid_argument("K exceeds the number of Q-bit states");
    }
    std::mt19937_64 rng(0x5eed1234ULL);
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(args.states * 2);
    std::vector<std::uint64_t> states;
    states.reserve(args.states);
    auto add = [&](std::uint64_t value) {
        value &= mask;
        if (seen.insert(value).second) states.push_back(value);
    };
    if (args.dataset == "ghz") {
        add(0);
        add(mask);
        for (std::uint32_t radius = 1; states.size() < args.states && radius <= 3; ++radius) {
            for (std::uint32_t a = 0; a < args.qubits && states.size() < args.states; ++a) {
                add(1ULL << a);
                add(mask ^ (1ULL << a));
                if (radius >= 2) {
                    for (std::uint32_t b = a + 1;
                         b < args.qubits && states.size() < args.states; ++b) {
                        add((1ULL << a) | (1ULL << b));
                        add(mask ^ ((1ULL << a) | (1ULL << b)));
                    }
                }
            }
        }
    } else if (args.dataset == "clustered") {
        std::vector<std::uint64_t> centers = {0, mask, rng() & mask, rng() & mask};
        while (states.size() < args.states) {
            std::uint64_t value = centers[states.size() % centers.size()];
            const std::uint32_t flips = 1U + static_cast<std::uint32_t>(rng() % 4U);
            for (std::uint32_t f = 0; f < flips; ++f) {
                value ^= 1ULL << (rng() % args.qubits);
            }
            add(value);
            if (seen.size() + 16 >= (args.qubits < 64 ? (1ULL << args.qubits) : ~0ULL))
                break;
        }
    } else if (args.dataset == "random") {
        while (states.size() < args.states) add(rng());
    } else {
        throw std::invalid_argument("invalid dataset");
    }
    while (states.size() < args.states) add(rng());
    return states;
}

std::vector<double> make_calibration(std::uint32_t qubits) {
    std::vector<double> cals(static_cast<std::size_t>(qubits) * 4U);
    for (std::uint32_t q = 0; q < qubits; ++q) {
        const double p10 = 0.015 + 0.0003 * (q % 11U);
        const double p01 = 0.025 + 0.0004 * (q % 7U);
        cals[static_cast<std::size_t>(q) * 4U + 0] = 1.0 - p10;
        cals[static_cast<std::size_t>(q) * 4U + 1] = p01;
        cals[static_cast<std::size_t>(q) * 4U + 2] = p10;
        cals[static_cast<std::size_t>(q) * 4U + 3] = 1.0 - p01;
    }
    return cals;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle])
                                  : values[middle];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse(argc, argv);
        const auto states = make_states(args);
        const auto cals = make_calibration(args.qubits);
        std::vector<double> raw(states.size());
        std::mt19937_64 rng(0xabcddcbaULL);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (double& value : raw) value = distribution(rng);
        const double sum = std::accumulate(raw.begin(), raw.end(), 0.0);
        for (double& value : raw) value /= sum;
        rem::Options options;
        options.backend = args.backend;
        options.solver = args.solver;
        options.graph_strategy = args.graph;
        options.sparse_storage = args.storage;
        options.orthogonalization = args.orthogonalization;
        options.max_hamming_distance = args.distance;
        options.threads = args.threads;
        options.relative_tolerance = 1e-7;
        options.max_iterations = 100;
        options.cache_values = args.storage == rem::SparseStorage::Valued;

        (void)rem::mitigate(
            states, states.size(), args.qubits, raw, cals, options);  // warm-up
        std::vector<double> totals;
        std::vector<double> setups;
        std::vector<double> solves;
        rem::Diagnostics diagnostics;
        for (std::size_t repetition = 0; repetition < args.repetitions; ++repetition) {
            const auto start = std::chrono::steady_clock::now();
            const auto result =
                rem::mitigate(states, states.size(), args.qubits, raw, cals, options);
            const double seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - start)
                                       .count();
            totals.push_back(seconds);
            setups.push_back(result.diagnostics.setup_seconds);
            solves.push_back(result.diagnostics.solve_seconds);
            diagnostics = result.diagnostics;
        }
        const auto [minimum, maximum] = std::minmax_element(totals.begin(), totals.end());
        if (args.json) {
            std::cout << std::setprecision(12)
                      << "{\"dataset\":\"" << args.dataset << "\",\"q\":" << args.qubits
                      << ",\"k\":" << states.size() << ",\"d\":" << args.distance
                      << ",\"threads\":" << diagnostics.threads << ",\"backend\":\""
                      << rem::to_string(diagnostics.selected_backend)
                      << "\",\"solver\":\"" << rem::to_string(diagnostics.selected_solver)
                      << "\",\"graph\":\""
                      << rem::to_string(diagnostics.selected_graph_strategy)
                      << "\",\"storage\":\""
                      << rem::to_string(diagnostics.selected_sparse_storage)
                      << "\",\"orthogonalization\":\""
                      << rem::to_string(diagnostics.selected_orthogonalization)
                      << "\",\"edges\":" << diagnostics.retained_edges
                      << ",\"iterations\":" << diagnostics.iterations
                      << ",\"setup_median_s\":" << median(setups)
                      << ",\"solve_median_s\":" << median(solves)
                      << ",\"total_min_s\":" << *minimum
                      << ",\"total_median_s\":" << median(totals)
                      << ",\"total_max_s\":" << *maximum
                      << ",\"residual\":" << diagnostics.residual_norm
                      << ",\"estimated_memory_bytes\":"
                      << diagnostics.estimated_memory_bytes << "}\n";
        } else {
            std::cout
                << "dataset,q,k,d,threads,backend,solver,graph,storage,orthogonalization,"
                   "edges,iterations,"
                   "setup_median_s,solve_median_s,total_min_s,total_median_s,total_max_s,"
                   "residual,estimated_memory_bytes\n"
                << std::setprecision(12) << args.dataset << ',' << args.qubits << ','
                << states.size() << ',' << args.distance << ',' << diagnostics.threads
                << ',' << rem::to_string(diagnostics.selected_backend) << ','
                << rem::to_string(diagnostics.selected_solver) << ','
                << rem::to_string(diagnostics.selected_graph_strategy) << ','
                << rem::to_string(diagnostics.selected_sparse_storage) << ','
                << rem::to_string(diagnostics.selected_orthogonalization) << ','
                << diagnostics.retained_edges << ',' << diagnostics.iterations << ','
                << median(setups) << ',' << median(solves) << ',' << *minimum << ','
                << median(totals) << ',' << *maximum << ','
                << diagnostics.residual_norm << ','
                << diagnostics.estimated_memory_bytes << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        return 1;
    }
}
