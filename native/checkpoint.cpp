#include "checkpoint.hpp"
#include "qwen3.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace vaidy {
namespace {

using Json = nlohmann::json;
namespace fs = std::filesystem;

Json read_json(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    return Json::parse(stream);
}

std::uint64_t unsigned_integer(const Json& value) {
    if (value.is_number_unsigned()) {
        return value.get<std::uint64_t>();
    }
    if (value.is_number_integer() && value.get<std::int64_t>() >= 0) {
        return value.get<std::uint64_t>();
    }
    throw std::runtime_error("Expected a nonnegative integer");
}

int dimension(const Json& config, const char* name) {
    const auto value = unsigned_integer(config.at(name));
    if (value == 0 || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("Invalid model dimension ") + name);
    }
    return static_cast<int>(value);
}

bool is_set(const Json& config, const char* name) {
    return config.contains(name) && !config[name].is_null();
}

float half_to_float(std::uint16_t bits) {
    const auto sign = std::uint32_t(bits & 0x8000) << 16;
    const auto exponent = std::uint32_t(bits >> 10 & 31);
    const auto mantissa = std::uint32_t(bits & 1023);
    if (exponent == 0) {
        const float value = std::ldexp(static_cast<float>(mantissa), -24);
        return sign ? -value : value;
    }
    const auto expanded = exponent == 31 ? 255 : exponent + 112;
    return std::bit_cast<float>(sign | (expanded << 23) | (mantissa << 13));
}

class Shard {
public:
    explicit Shard(const fs::path& path) : stream_(path, std::ios::binary | std::ios::ate) {
        static_assert(std::endian::native == std::endian::little);
        if (!stream_) {
            throw std::runtime_error("Cannot open checkpoint shard " + path.string());
        }
        const auto file_size = stream_.tellg();
        stream_.seekg(0);
        std::uint64_t header_size = 0;
        if (file_size < static_cast<std::streamoff>(sizeof(header_size)) ||
            !stream_.read(reinterpret_cast<char*>(&header_size), sizeof(header_size)) ||
            header_size == 0 || header_size > 100'000'000 ||
            header_size > static_cast<std::uint64_t>(file_size) - sizeof(header_size)) {
            throw std::runtime_error("Invalid safetensors header in " + path.string());
        }
        std::string data(static_cast<std::size_t>(header_size), '\0');
        if (!stream_.read(data.data(), static_cast<std::streamsize>(data.size()))) {
            throw std::runtime_error("Cannot read safetensors header in " + path.string());
        }
        header_ = Json::parse(data);
        data_start_ = sizeof(header_size) + header_size;
        data_size_ = static_cast<std::uint64_t>(file_size) - data_start_;
    }

    void read(const std::string& name, std::initializer_list<int> shape, std::span<float> output) {
        const auto& info = header_.at(name);
        const auto& dtype = info.at("dtype");
        const auto& dimensions = info.at("shape");
        const auto& offsets = info.at("data_offsets");
        if (!dimensions.is_array() || dimensions.size() != shape.size() ||
            !std::equal(shape.begin(), shape.end(), dimensions.begin(),
                [](int expected, const Json& actual) {
                    return static_cast<std::uint64_t>(expected) == unsigned_integer(actual);
                }) ||
            (dtype != "F32" && dtype != "BF16" && dtype != "F16") ||
            !offsets.is_array() || offsets.size() != 2 ||
            output.size() != elements(shape)) {
            throw std::runtime_error("Incompatible weight " + name);
        }
        const auto start = unsigned_integer(offsets[0]);
        const auto end = unsigned_integer(offsets[1]);
        const std::size_t value_size = dtype == "F32" ? sizeof(float) : sizeof(std::uint16_t);
        if (start > end || end > data_size_ || (end - start) % value_size != 0 ||
            (end - start) / value_size != output.size()) {
            throw std::runtime_error("Invalid data offsets for " + name);
        }
        stream_.seekg(static_cast<std::streamoff>(data_start_ + start));
        if (dtype == "F32") {
            read_bytes(output.data(), static_cast<std::size_t>(end - start), name);
            return;
        }
        const bool bf16 = dtype == "BF16";
        while (!output.empty()) {
            const auto count = std::min(output.size(), storage_.size());
            read_bytes(storage_.data(), count * sizeof(std::uint16_t), name);
            if (bf16) {
                for (std::size_t i = 0; i < count; ++i) {
                    output[i] = std::bit_cast<float>(std::uint32_t(storage_[i]) << 16);
                }
            } else {
                std::transform(storage_.begin(), storage_.begin() + count, output.begin(), half_to_float);
            }
            output = output.subspan(count);
        }
    }

private:
    void read_bytes(void* destination, std::size_t count, const std::string& name) {
        if (!stream_.read(static_cast<char*>(destination), static_cast<std::streamsize>(count))) {
            throw std::runtime_error("Cannot read tensor " + name);
        }
    }

    std::ifstream stream_;
    Json header_;
    std::uint64_t data_start_;
    std::uint64_t data_size_;
    std::array<std::uint16_t, 32768> storage_;
};

}  // namespace

Qwen3Config load_qwen3_config(const fs::path& directory) {
    const auto config = read_json(directory / "config.json");
    if (config.value("model_type", "") != "qwen3" || config.value("hidden_act", "") != "silu" ||
        config.value("attention_bias", false) || config.value("use_sliding_window", false) ||
        is_set(config, "rope_scaling") || is_set(config, "rope_parameters")) {
        throw std::runtime_error("Use a dense Qwen3 checkpoint with standard RoPE and full attention");
    }
    Qwen3Config result{
        dimension(config, "vocab_size"), dimension(config, "hidden_size"),
        dimension(config, "intermediate_size"), dimension(config, "num_hidden_layers"),
        dimension(config, "num_attention_heads"), dimension(config, "num_key_value_heads"),
        dimension(config, "head_dim"), dimension(config, "max_position_embeddings"),
        config.value("rms_norm_eps", 1e-6f), config.value("rope_theta", 1'000'000.f),
        config.value("tie_word_embeddings", true),
    };
    if (result.head_dim % 2 || result.query_heads % result.kv_heads ||
        elements({result.query_heads, result.head_dim}) + 2 * elements({result.kv_heads, result.head_dim}) >
            std::numeric_limits<int>::max() ||
        result.intermediate > std::numeric_limits<int>::max() / 2 ||
        !std::isfinite(result.norm_eps) || result.norm_eps <= 0 ||
        !std::isfinite(result.rope_theta) || result.rope_theta <= 0) {
        throw std::runtime_error("Invalid Qwen3 attention or normalization dimensions");
    }
    return result;
}

std::size_t elements(std::initializer_list<int> dimensions) {
    std::size_t result = 1;
    for (const int dimension : dimensions) {
        if (dimension <= 0 || result > std::vector<float>().max_size() / static_cast<std::size_t>(dimension)) {
            throw std::length_error("Tensor dimensions are too large");
        }
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

struct Checkpoint::Files {
    fs::path directory;
    Json weight_map;
    std::map<std::string, Shard> shards;
};

Checkpoint::Checkpoint(const fs::path& directory) : files_(std::make_unique<Files>()) {
    files_->directory = directory;
    if (!fs::is_regular_file(directory / "model.safetensors") &&
        fs::exists(directory / "model.safetensors.index.json")) {
        const auto index = read_json(directory / "model.safetensors.index.json");
        files_->weight_map = index.at("weight_map");
        if (!files_->weight_map.is_object() || files_->weight_map.empty()) {
            throw std::runtime_error("Safetensors index requires a nonempty weight_map");
        }
        for (const auto& [name, value] : files_->weight_map.items()) {
            const auto& filename = value.get_ref<const std::string&>();
            if (name.empty() || !filename.ends_with(".safetensors") ||
                filename.find_first_of("/\\") != std::string::npos ||
                filename.find('\0') != std::string::npos ||
                !fs::is_regular_file(directory / filename)) {
                throw std::runtime_error("Invalid or missing checkpoint shard: " + filename);
            }
        }
    }
}

Checkpoint::~Checkpoint() = default;

void Checkpoint::read(const std::string& name, std::initializer_list<int> shape, std::span<float> output) {
    const auto file = files_->weight_map.is_null() ? "model.safetensors" :
        files_->weight_map.at(name).get<std::string>();
    files_->shards.try_emplace(file, files_->directory / file).first->second.read(name, shape, output);
}

}  // namespace vaidy
