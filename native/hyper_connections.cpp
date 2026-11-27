#include "hyper_connections.hpp"

#include <algorithm>
#include <cmath>

namespace vaidy {
namespace {

float sigmoid(float value) {
    if (value >= 0) {
        return 1.f / (1.f + std::exp(-value));
    }
    const float exp = std::exp(value);
    return exp / (1.f + exp);
}

void normalize(float* matrix, std::size_t count, std::size_t outer_stride,
               std::size_t inner_stride, float eps) {
    for (std::size_t outer = 0; outer < count; ++outer) {
        float* start = matrix + outer * outer_stride;
        float sum = 0;
        for (std::size_t i = 0; i < count; ++i) {
            sum += start[i * inner_stride];
        }
        const float denominator = sum + eps;
        for (std::size_t i = 0; i < count; ++i) {
            start[i * inner_stride] /= denominator;
        }
    }
}

}  // namespace

void split_hyper_connections(const float* mixes, const float* scale, const float* base,
    float* output, std::size_t rows, std::size_t count, int iterations, float eps) {
    const auto offset = 2 * count;
    const auto width = count * (count + 2);
    for (std::size_t sample = 0; sample < rows; ++sample) {
        for (std::size_t i = 0; i < count; ++i) {
            output[i] = sigmoid(mixes[i] * scale[0] + base[i]) + eps;
            output[count + i] = 2.f * sigmoid(mixes[count + i] * scale[1] + base[count + i]);
        }
        float* comb = output + offset;
        for (std::size_t i = offset; i < width; ++i) {
            output[i] = mixes[i] * scale[2] + base[i];
        }
        for (std::size_t row = 0; row < count; ++row) {
            float* start = comb + row * count;
            const float max_value = *std::max_element(start, start + count);
            float sum = 0;
            for (std::size_t i = 0; i < count; ++i) {
                start[i] = std::exp(start[i] - max_value);
                sum += start[i];
            }
            for (std::size_t i = 0; i < count; ++i) {
                start[i] = start[i] / sum + eps;
            }
        }
        normalize(comb, count, 1, count, eps);
        for (int i = 1; i < iterations; ++i) {
            normalize(comb, count, count, 1, eps);
            normalize(comb, count, 1, count, eps);
        }
        mixes += width;
        output += width;
    }
}

}  // namespace vaidy
