// Development benchmark retained to document why ankerl::unordered_dense was
// adopted. LegacyIndex is the project's former task-specific open-addressed
// uint64_t table; StateIndex is intentionally not used because its fast path now
// uses ankerl::unordered_dense.
#include "detail.hpp"
#include <ankerl/unordered_dense.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>

namespace {

std::uint64_t mix(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

class LegacyIndex {
public:
    explicit LegacyIndex(std::span<const std::uint64_t> states) {
        std::size_t capacity = 8;
        while (capacity < states.size() * 2U) {
            capacity *= 2U;
        }
        keys_.assign(capacity, 0);
        values_.assign(capacity, missing());
        mask_ = capacity - 1U;
        for (std::size_t index = 0; index < states.size(); ++index) {
            std::size_t slot = static_cast<std::size_t>(mix(states[index])) & mask_;
            while (values_[slot] != missing()) {
                slot = (slot + 1U) & mask_;
            }
            keys_[slot] = states[index];
            values_[slot] = static_cast<std::uint32_t>(index);
        }
    }

    [[nodiscard]] bool contains(std::uint64_t value) const noexcept {
        std::size_t slot = static_cast<std::size_t>(mix(value)) & mask_;
        while (values_[slot] != missing()) {
            if (keys_[slot] == value) {
                return true;
            }
            slot = (slot + 1U) & mask_;
        }
        return false;
    }

private:
    static constexpr std::uint32_t missing() noexcept {
        return std::numeric_limits<std::uint32_t>::max();
    }

    std::vector<std::uint64_t> keys_;
    std::vector<std::uint32_t> values_;
    std::size_t mask_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    const std::size_t count =
        argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 20000;
    const std::size_t repetitions =
        argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 5;
    constexpr std::uint32_t qubits = 32;
    rem::detail::PackedStates states;
    states.qubits = qubits;
    states.words_per_state = 1;
    states.words.resize(count);
    std::mt19937_64 rng(0x10203040ULL);
    ankerl::unordered_dense::map<std::uint64_t, std::uint32_t> external;
    external.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint64_t value;
        do {
            value = rng() & 0xffffffffULL;
        } while (external.contains(value));
        states.words[i] = value;
        external.emplace(value, static_cast<std::uint32_t>(i));
    }
    LegacyIndex custom(states.words);

    const std::uint64_t queries_per_repetition =
        count * (1U + qubits + qubits * (qubits - 1U) / 2U);
    auto run_custom = [&] {
        std::uint64_t checksum = 0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
            for (const std::uint64_t state : states.words) {
                rem::detail::for_each_ball_candidate_64(
                    state, qubits, 2, [&](std::uint64_t candidate) {
                        checksum += custom.contains(candidate);
                    });
            }
        }
        return std::pair{
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                .count(),
            checksum};
    };
    auto run_external = [&] {
        std::uint64_t checksum = 0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
            for (const std::uint64_t state : states.words) {
                rem::detail::for_each_ball_candidate_64(
                    state, qubits, 2, [&](std::uint64_t candidate) {
                        checksum += external.contains(candidate);
                    });
            }
        }
        return std::pair{
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                .count(),
            checksum};
    };
    (void)run_custom();
    (void)run_external();
    const auto [custom_seconds, custom_checksum] = run_custom();
    const auto [external_seconds, external_checksum] = run_external();
    const double queries =
        static_cast<double>(queries_per_repetition * repetitions);
    std::cout << std::setprecision(12)
              << "{\"k\":" << count << ",\"queries\":" << queries
              << ",\"custom_ns_per_query\":" << custom_seconds * 1e9 / queries
              << ",\"ankerl_ns_per_query\":" << external_seconds * 1e9 / queries
              << ",\"custom_over_ankerl\":" << custom_seconds / external_seconds
              << ",\"checksums_equal\":"
              << (custom_checksum == external_checksum ? "true" : "false") << "}\n";
    return custom_checksum == external_checksum ? 0 : 2;
}
