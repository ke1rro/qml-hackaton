#include "detail.hpp"

#include <cstring>
#include <unordered_set>

namespace rem::detail {
namespace {

constexpr std::uint32_t kMissing = std::numeric_limits<std::uint32_t>::max();

std::uint64_t mix64(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

}  // namespace

StateIndex::StateIndex(const PackedStates* states, std::size_t expected_size) : states_(states) {
    if (states_ != nullptr && states_->words_per_state == 1) {
        fast64_.reserve(expected_size);
        if (states_->size() >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::length_error("state index exceeds uint32 range");
        }
        for (std::size_t index = 0; index < states_->size(); ++index) {
            fast64_.emplace(
                states_->words[index], static_cast<std::uint32_t>(index));
        }
        return;
    }
    if (expected_size >
        (std::numeric_limits<std::size_t>::max() - 1U) / 2U) {
        throw std::length_error("state index capacity overflows size_t");
    }
    const std::size_t required_capacity = expected_size * 2U + 1U;
    std::size_t capacity = 8;
    while (capacity < required_capacity) {
        if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::length_error("state index capacity overflow");
        }
        capacity *= 2;
    }
    buckets_.assign(capacity, 0);
    mask_ = capacity - 1;
    if (states_ != nullptr) {
        if (states_->size() >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::length_error("state index exceeds uint32 range");
        }
        for (std::size_t index = 0; index < states_->size(); ++index) {
            insert_index(static_cast<std::uint32_t>(index));
        }
    }
}

std::uint64_t StateIndex::hash_words(std::span<const std::uint64_t> state) noexcept {
    if (state.size() == 1) {
        return mix64(state.front());
    }
    std::uint64_t hash = 0x243f6a8885a308d3ULL ^ static_cast<std::uint64_t>(state.size());
    for (const std::uint64_t word : state) {
        hash = mix64(hash ^ mix64(word));
    }
    return hash;
}

bool StateIndex::equal(
    std::uint32_t index, std::span<const std::uint64_t> state) const noexcept {
    const auto candidate = states_->state(index);
    return std::equal(candidate.begin(), candidate.end(), state.begin(), state.end());
}

std::uint32_t StateIndex::find(std::span<const std::uint64_t> state) const noexcept {
    if (states_->words_per_state == 1) {
        const auto found = fast64_.find(state.front());
        return found == fast64_.end() ? kMissing : found->second;
    }
    if (buckets_.empty()) {
        return kMissing;
    }
    std::size_t slot = static_cast<std::size_t>(hash_words(state)) & mask_;
    for (;;) {
        const std::uint32_t encoded = buckets_[slot];
        if (encoded == 0) {
            return kMissing;
        }
        const std::uint32_t index = encoded - 1U;
        if (equal(index, state)) {
            return index;
        }
        slot = (slot + 1U) & mask_;
    }
}

std::pair<std::uint32_t, bool> StateIndex::insert_or_find(
    std::span<const std::uint64_t> state) {
    if (states_->size() == 0 ||
        states_->size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::length_error("state index exceeds uint32 range");
    }
    const auto insertion_index = static_cast<std::uint32_t>(states_->size() - 1);
    if (states_->words_per_state == 1) {
        const auto [iterator, inserted] =
            fast64_.emplace(state.front(), insertion_index);
        return {iterator->second, inserted};
    }
    std::size_t slot = static_cast<std::size_t>(hash_words(state)) & mask_;
    for (;;) {
        const std::uint32_t encoded = buckets_[slot];
        if (encoded == 0) {
            buckets_[slot] = insertion_index + 1U;
            return {insertion_index, true};
        }
        const std::uint32_t index = encoded - 1U;
        if (equal(index, state)) {
            return {index, false};
        }
        slot = (slot + 1U) & mask_;
    }
}

void StateIndex::insert_index(std::uint32_t index) {
    const auto state = states_->state(index);
    std::size_t slot = static_cast<std::size_t>(hash_words(state)) & mask_;
    while (buckets_[slot] != 0) {
        if (equal(buckets_[slot] - 1U, state)) {
            return;
        }
        slot = (slot + 1U) & mask_;
    }
    buckets_[slot] = index + 1U;
}

PreparedInput prepare_input(
    std::span<const std::uint64_t> packed,
    std::size_t count,
    std::uint32_t qubits,
    std::span<const double> probabilities,
    std::size_t right_hand_sides) {
    if (qubits == 0) {
        throw std::invalid_argument("measured_qubits must be positive");
    }
    if (count == 0) {
        throw std::invalid_argument("at least one bitstring is required");
    }
    if (right_hand_sides == 0) {
        throw std::invalid_argument("right_hand_sides must be positive");
    }
    const std::size_t word_count = words_for(qubits);
    if (count > std::numeric_limits<std::size_t>::max() / word_count) {
        throw std::length_error("packed bitstring buffer size overflows size_t");
    }
    const std::size_t packed_size = count * word_count;
    if (right_hand_sides > std::numeric_limits<std::size_t>::max() / count) {
        throw std::length_error("probability buffer size overflows size_t");
    }
    const std::size_t probability_size = count * right_hand_sides;
    if (packed.size() != packed_size) {
        throw std::invalid_argument("packed bitstring buffer has the wrong size");
    }
    if (probabilities.size() != probability_size) {
        throw std::invalid_argument("probability buffer has the wrong size");
    }
    for (const double value : probabilities) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("raw probabilities must be finite");
        }
    }

    PreparedInput output;
    output.states.qubits = qubits;
    output.states.words_per_state = word_count;
    output.states.words.reserve(packed_size);
    output.original_to_unique.resize(count);
    output.right_hand_sides = right_hand_sides;
    StateIndex index(&output.states, count);
    std::vector<std::uint64_t> scratch(word_count);
    const std::uint32_t tail_bits = qubits % 64U;
    const std::uint64_t tail_mask =
        tail_bits == 0 ? std::numeric_limits<std::uint64_t>::max()
                       : ((1ULL << tail_bits) - 1ULL);

    for (std::size_t row = 0; row < count; ++row) {
        std::copy_n(packed.data() + row * word_count, word_count, scratch.data());
        scratch.back() &= tail_mask;
        std::uint32_t unique_index = index.find(scratch);
        if (unique_index == kMissing) {
            output.states.words.insert(
                output.states.words.end(), scratch.begin(), scratch.end());
            auto [inserted_index, inserted] =
                index.insert_or_find(output.states.state(output.states.size() - 1));
            if (!inserted) {
                throw std::logic_error("state index insertion failed");
            }
            unique_index = inserted_index;
        }
        output.original_to_unique[row] = unique_index;
    }

    const std::size_t unique_count = output.states.size();
    output.probabilities.assign(right_hand_sides * unique_count, 0.0);
    for (std::size_t rhs = 0; rhs < right_hand_sides; ++rhs) {
        for (std::size_t row = 0; row < count; ++row) {
            output.probabilities[rhs * unique_count + output.original_to_unique[row]] +=
                probabilities[rhs * count + row];
        }
    }
    return output;
}

std::uint32_t hamming_distance(
    std::span<const std::uint64_t> lhs,
    std::span<const std::uint64_t> rhs,
    std::uint32_t stop_after) noexcept {
    std::uint32_t distance = 0;
    for (std::size_t word = 0; word < lhs.size(); ++word) {
        distance += static_cast<std::uint32_t>(std::popcount(lhs[word] ^ rhs[word]));
        if (distance > stop_after) {
            break;
        }
    }
    return distance;
}

std::uint64_t hamming_ball_size(std::uint32_t qubits, std::uint32_t distance) noexcept {
    distance = std::min(distance, qubits);
    std::uint64_t sum = 1;
    std::uint64_t binomial = 1;
    for (std::uint32_t rank = 1; rank <= distance; ++rank) {
        const std::uint64_t numerator = qubits - rank + 1U;
        if (binomial > std::numeric_limits<std::uint64_t>::max() / numerator) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        binomial = binomial * numerator / rank;
        sum = saturating_add(sum, binomial);
    }
    return sum;
}

void for_each_ball_candidate_generic(
    std::span<const std::uint64_t> state,
    std::uint32_t qubits,
    std::uint32_t distance,
    const std::function<void(std::span<const std::uint64_t>)>& callback) {
    std::vector<std::uint64_t> scratch(state.size());
    for_each_ball_candidate_generic_scratch(
        state, qubits, distance, scratch, callback);
}

}  // namespace rem::detail
