#include "rem/api.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace rem {

std::vector<double> project_probability_simplex(std::span<const double> values) {
    if (values.empty()) {
        return {};
    }
    for (const double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("simplex projection requires finite values");
        }
    }
    std::vector<double> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    double prefix = 0.0;
    std::size_t active = 0;
    for (std::size_t index = 0; index < sorted.size(); ++index) {
        prefix += sorted[index];
        const double threshold = (prefix - 1.0) / static_cast<double>(index + 1U);
        if (sorted[index] > threshold) {
            active = index + 1U;
        }
    }
    prefix = std::accumulate(
        sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(active), 0.0);
    const double theta = (prefix - 1.0) / static_cast<double>(active);
    std::vector<double> projected(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        projected[index] = std::max(values[index] - theta, 0.0);
    }
    return projected;
}

}  // namespace rem
