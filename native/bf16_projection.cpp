#include "bf16_projection.hpp"

#if defined(__APPLE__) && defined(__aarch64__)

#include <arm_neon.h>
#include <dispatch/dispatch.h>

#include <bit>
#include <cstdint>
#include <cstring>

namespace vaidy {
namespace {

float dot_bf16(const std::byte* weights, const float* input, int columns) {
    float32x4_t a = vdupq_n_f32(0), b = a, c = a, d = a;
    int column = 0;
    for (; column <= columns - 16; column += 16) {
        const auto* data = reinterpret_cast<const std::uint8_t*>(weights) + 2 * std::size_t(column);
        const auto low = vreinterpretq_u16_u8(vld1q_u8(data));
        const auto high = vreinterpretq_u16_u8(vld1q_u8(data + 16));
        a = vfmaq_f32(a, vld1q_f32(input + column), vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(low), 16)));
        b = vfmaq_f32(b, vld1q_f32(input + column + 4), vreinterpretq_f32_u32(vshll_high_n_u16(low, 16)));
        c = vfmaq_f32(c, vld1q_f32(input + column + 8), vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(high), 16)));
        d = vfmaq_f32(d, vld1q_f32(input + column + 12), vreinterpretq_f32_u32(vshll_high_n_u16(high, 16)));
    }
    float result = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
    for (; column < columns; ++column) {
        std::uint16_t bits;
        std::memcpy(&bits, weights + std::size_t(column) * sizeof(bits), sizeof(bits));
        result += std::bit_cast<float>(std::uint32_t(bits) << 16) * input[column];
    }
    return result;
}

struct Projection {
    const std::byte* weights;
    const float* input;
    float* output;
    int rows;
    int columns;
};

constexpr std::size_t workers = 4;

void project_rows(void* opaque, std::size_t index) {
    const auto& projection = *static_cast<const Projection*>(opaque);
    const std::size_t start = std::size_t(projection.rows) * index / workers;
    const std::size_t end = std::size_t(projection.rows) * (index + 1) / workers;
    const std::size_t stride = 2 * std::size_t(projection.columns);
    for (auto row = start; row < end; ++row) {
        projection.output[row] = dot_bf16(projection.weights + row * stride, projection.input, projection.columns);
    }
}

}  // namespace

void project_bf16_vocabulary(const std::byte* weights, int rows, int columns,
    const float* input, float* output) {
    Projection projection{weights, input, output, rows, columns};
    dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &projection, project_rows);
}

}  // namespace vaidy

#endif
