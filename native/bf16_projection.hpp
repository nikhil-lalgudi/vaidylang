#pragma once

#include <cstddef>

namespace vaidy {

// Large Apple Silicon vocabulary projection. The checkpoint weights remain
// in BF16 while all products and accumulators use FP32.
void project_bf16_vocabulary(const std::byte* weights, int rows, int columns,
    const float* input, float* output);

}  // namespace vaidy
