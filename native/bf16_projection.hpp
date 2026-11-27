#pragma once

#include <cstddef>

namespace vaidy {

// The source rows use little-endian BF16. Products and accumulation use FP32.
void project_bf16_rows(const std::byte* weights, int rows, int columns,
    const float* input, float* output, float beta);

// Large Apple Silicon vocabulary projection using the same row kernel.
void project_bf16_vocabulary(const std::byte* weights, int rows, int columns,
    const float* input, float* output);

}  // namespace vaidy
