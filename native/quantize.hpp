#pragma once

#include <cstddef>

namespace vaidy {

void quantize_activation(const float* input, float* output, std::size_t rows,
    std::size_t width, std::size_t group, bool fp4, bool fp8_scale);

}  // namespace vaidy
