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

class Checkpoint {
    struct Files;

public:
    explicit Checkpoint(const std::filesystem::path& directory);
    ~Checkpoint();
    void read(const std::string& name, std::initializer_list<int> shape, std::span<float> output);

private:
    std::unique_ptr<Files> files_;
};

}  // namespace vaidy
