#include "detail.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const std::size_t pairs =
        argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 100000;
    const std::uint32_t qubits =
        argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 64;
    const std::size_t repetitions =
        argc > 3 ? static_cast<std::size_t>(std::stoull(argv[3])) : 20;
    if (qubits == 0 || qubits > 64) {
        std::cerr << "bench_elements currently expects 1 <= Q <= 64\n";
        return 1;
    }

    rem::detail::PackedStates states;
    states.qubits = qubits;
    states.words_per_state = 1;
    states.words.resize(pairs * 2U);
    std::mt19937_64 rng(0x13579bdfULL);
    const std::uint64_t mask =
        qubits == 64 ? ~0ULL : ((1ULL << qubits) - 1ULL);
    for (std::size_t pair = 0; pair < pairs; ++pair) {
        const std::uint64_t source = rng() & mask;
        std::uint64_t target = source;
        for (std::uint32_t flip = 0; flip < std::min<std::uint32_t>(3, qubits); ++flip) {
            target ^= 1ULL << ((pair * 17U + flip * 11U) % qubits);
        }
        states.words[2U * pair] = source;
        states.words[2U * pair + 1U] = target;
    }
    std::vector<double> cals(static_cast<std::size_t>(qubits) * 4U);
    for (std::uint32_t q = 0; q < qubits; ++q) {
        const double p10 = 0.01 + 0.0001 * q;
        const double p01 = 0.02 + 0.0001 * q;
        cals[static_cast<std::size_t>(q) * 4U + 0] = 1.0 - p10;
        cals[static_cast<std::size_t>(q) * 4U + 1] = p01;
        cals[static_cast<std::size_t>(q) * 4U + 2] = p10;
        cals[static_cast<std::size_t>(q) * 4U + 3] = 1.0 - p01;
    }
    rem::detail::ThreadPool pool(1);
    rem::Diagnostics diagnostics;
    const auto calibration =
        rem::detail::prepare_calibration(states, cals, 1e-15, pool, diagnostics);

    auto run = [&](bool fast) {
        double checksum = 0.0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
            for (std::size_t pair = 0; pair < pairs; ++pair) {
                const auto source = static_cast<std::uint32_t>(2U * pair);
                const auto target = source + 1U;
                checksum +=
                    fast ? rem::detail::fast_element(
                               states, calibration, target, source)
                         : rem::detail::direct_element(
                               states, calibration, target, source);
            }
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        return std::pair{seconds, checksum};
    };
    (void)run(true);
    (void)run(false);
    const auto [direct_seconds, direct_checksum] = run(false);
    const auto [fast_seconds, fast_checksum] = run(true);
    const double evaluations = static_cast<double>(pairs * repetitions);
    const double relative_error =
        std::abs(direct_checksum - fast_checksum) /
        std::max(std::abs(direct_checksum), 1e-300);
    std::cout << std::setprecision(12)
              << "{\"q\":" << qubits << ",\"pairs\":" << pairs
              << ",\"repetitions\":" << repetitions
              << ",\"direct_ns_per_element\":" << direct_seconds * 1e9 / evaluations
              << ",\"ratio_ns_per_element\":" << fast_seconds * 1e9 / evaluations
              << ",\"speedup\":" << direct_seconds / fast_seconds
              << ",\"checksum_relative_error\":" << relative_error << "}\n";
    return relative_error <= 1e-12 ? 0 : 2;
}
