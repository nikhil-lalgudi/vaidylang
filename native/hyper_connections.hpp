#pragma once

#include <cstddef>

namespace vaidy {

void split_hyper_connections(const float* mixes, const float* scale, const float* base,
    float* output, std::size_t rows, std::size_t count, int iterations, float eps);

}  // namespace vaidy
