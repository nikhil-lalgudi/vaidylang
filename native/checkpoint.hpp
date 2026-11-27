#pragma once

#include <cstddef>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>

namespace vaidy {

struct Qwen3Config;
Qwen3Config load_qwen3_config(const std::filesystem::path& directory);

std::size_t elements(std::initializer_list<int> dimensions);

class MappedTensor {
    struct Mapping;
    friend class Checkpoint;

public:
    enum class Dtype { fp32, bf16, fp16 };

    void read(std::size_t offset, std::span<float> output) const;

private:
    std::shared_ptr<const Mapping> mapping_;
    std::size_t start_ = 0;
    std::size_t count_ = 0;
    Dtype dtype_ = Dtype::fp32;
};

class Checkpoint {
    struct Files;

public:
    explicit Checkpoint(const std::filesystem::path& directory);
    ~Checkpoint();
    void read(const std::string& name, std::initializer_list<int> shape, std::span<float> output);
    MappedTensor map(const std::string& name, std::initializer_list<int> shape);

private:
    std::unique_ptr<Files> files_;
};

}  // namespace vaidy
