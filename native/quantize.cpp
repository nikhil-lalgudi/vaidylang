#include "quantize.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>

namespace vaidy {
namespace {

constexpr std::array fp4_magnitudes{0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

const auto fp8_magnitudes = [] {
    std::array<float, 127> values{};
    for (int bits = 0; bits < static_cast<int>(values.size()); ++bits) {
        const int exponent = bits >> 3;
        const int mantissa = bits & 7;
        const float significand = (exponent > 0 ? 1.f : 0.f) + mantissa / 8.f;
        values[bits] = std::ldexp(significand, std::max(exponent, 1) - 7);
    }
    return values;
}();

float round_magnitude(float value, std::span<const float> table) {
    const auto upper = std::lower_bound(table.begin(), table.end(), value);
    if (upper == table.begin()) {
        return table.front();
    }
    if (upper == table.end()) {
        return table.back();
    }
    const auto lower = upper - 1;
    const float from_lower = value - *lower;
    const float from_upper = *upper - value;
    const bool even_lower = (lower - table.begin()) % 2 == 0;
    return (from_lower < from_upper || (from_lower == from_upper && even_lower)) ? *lower : *upper;
}

float power_of_two_scale(float maximum, float minimum, float inverse_limit) {
    const float relative = std::max(maximum, minimum) * inverse_limit;
    int exponent;
    const float fraction = std::frexp(relative, &exponent);
    return std::ldexp(1.f, exponent - (fraction == .5f));
}

}  // namespace

void quantize_activation(const float* input, float* output, std::size_t rows,
    std::size_t width, std::size_t group, bool fp4, bool fp8_scale) {
    if (width == 0 || group == 0 || (fp8_scale && !fp4)) {
        throw std::invalid_argument("Invalid activation quantization dimensions or encoding");
    }
    const auto table = fp4 ? std::span<const float>(fp4_magnitudes) : std::span<const float>(fp8_magnitudes);
    const float minimum = fp4 ? 6.f * 0x1p-126f : 1e-4f;
    const float inverse_limit = 1.f / table.back();
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < width;) {
            const auto count = std::min(group, width - column);
            const auto values = std::span(input + column, count);
            float maximum = 0;
            for (const float value : values) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("Found a nonfinite activation");
                }
                maximum = std::max(maximum, std::abs(value));
            }
            const float scale = fp8_scale
                ? round_magnitude(std::max(maximum, 6.f * 0x1p-9f) / 6.f, fp8_magnitudes)
                : power_of_two_scale(maximum, minimum, inverse_limit);
            for (std::size_t i = 0; i < count; ++i) {
                const float magnitude = round_magnitude(std::abs(values[i]) / scale, table);
                output[column + i] = std::copysign(magnitude, values[i]) * scale;
            }
            column += count;
        }
        input += width;
        output += width;
    }
}

}  // namespace vaidy
