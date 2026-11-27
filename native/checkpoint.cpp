#include "checkpoint.hpp"
#include "inkling.hpp"
#include "qwen3.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
    struct Layout {
        std::size_t start;
        std::size_t count;
        MappedTensor::Dtype dtype;
    };

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

    Layout layout(const std::string& name, std::initializer_list<int> shape) const {
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
            !offsets.is_array() || offsets.size() != 2) {
            throw std::runtime_error("Incompatible weight " + name);
        }
        const auto start = unsigned_integer(offsets[0]);
        const auto end = unsigned_integer(offsets[1]);
        const auto count = elements(shape);
        const std::size_t value_size = dtype == "F32" ? sizeof(float) : sizeof(std::uint16_t);
        if (start > end || end > data_size_ || (end - start) % value_size != 0 ||
            (end - start) / value_size != count) {
            throw std::runtime_error("Invalid data offsets for " + name);
        }
        using Dtype = MappedTensor::Dtype;
        const auto storage = dtype == "F32" ? Dtype::fp32 : dtype == "BF16" ? Dtype::bf16 : Dtype::fp16;
        return {static_cast<std::size_t>(data_start_ + start), count, storage};
    }

    void read(const std::string& name, std::initializer_list<int> shape, std::span<float> output) {
        const auto tensor = layout(name, shape);
        if (output.size() != tensor.count) {
            throw std::runtime_error("Incompatible output size for " + name);
        }
        stream_.seekg(static_cast<std::streamoff>(tensor.start));
        if (tensor.dtype == MappedTensor::Dtype::fp32) {
            read_bytes(output.data(), output.size_bytes(), name);
            return;
        }
        const bool bf16 = tensor.dtype == MappedTensor::Dtype::bf16;
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

struct MappedTensor::Mapping {
    explicit Mapping(const fs::path& path) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            throw std::runtime_error("Cannot open checkpoint shard " + path.string());
        }
        struct stat info {};
        if (::fstat(fd, &info) != 0 || info.st_size <= 0) {
            ::close(fd);
            throw std::runtime_error("Cannot size checkpoint shard " + path.string());
        }
        size = static_cast<std::size_t>(info.st_size);
        void* region = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (region == MAP_FAILED) {
            throw std::runtime_error("Cannot map checkpoint shard " + path.string());
        }
        data = static_cast<const std::byte*>(region);
    }

    ~Mapping() { ::munmap(const_cast<std::byte*>(data), size); }

    const std::byte* data;
    std::size_t size;
};

void MappedTensor::read(std::size_t offset, std::span<float> output) const {
    if (!mapping_ || offset > count_ || output.size() > count_ - offset) {
        throw std::out_of_range("Mapped tensor read is outside its shape");
    }
    if (dtype_ == Dtype::fp32) {
        std::memcpy(output.data(), mapping_->data + start_ + offset * sizeof(float), output.size_bytes());
        return;
    }
    const auto* data = mapping_->data + start_ + offset * sizeof(std::uint16_t);
    for (std::size_t i = 0; i < output.size(); ++i) {
        std::uint16_t bits;
        std::memcpy(&bits, data + i * sizeof(bits), sizeof(bits));
        output[i] = dtype_ == Dtype::bf16 ? std::bit_cast<float>(std::uint32_t(bits) << 16) : half_to_float(bits);
    }
}

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

InklingConfig load_inkling_config(const fs::path& directory) {
    const auto config = read_json(directory / "config.json");
    const auto& text = config.at("text_config");
    if (config.value("model_type", "") != "inkling_mm_model" ||
        is_set(config, "quantization_config") || fs::exists(directory / "hf_quant_config.json") ||
        text.value("q_bias", false) || text.value("o_bias", false) ||
        !text.value("use_embed_norm", true) || !text.value("use_sconv", true) ||
        !text.value("shared_expert_sink", true) || !text.value("use_gate_bias", true) ||
        !text.value("norm_after_topk", true) || !text.value("use_global_scale", true) ||
        text.value("gate_activation", "sigmoid") != "sigmoid" || is_set(text, "final_logit_softcapping")) {
        throw std::runtime_error("Use an unquantized checkpoint in the original Inkling architecture");
    }
    const int vocab = dimension(text, "vocab_size");
    const int layers = dimension(text, "num_hidden_layers");
    const int window = dimension(text, "sliding_window_size");
    const auto dense = unsigned_integer(text.at("dense_mlp_idx"));
    if (dense > static_cast<std::uint64_t>(layers)) {
        throw std::runtime_error("Invalid Inkling dense layer count");
    }
    InklingConfig result{
        vocab, is_set(text, "unpadded_vocab_size") ? dimension(text, "unpadded_vocab_size") : vocab,
        dimension(text, "hidden_size"), dimension(text, text.contains("max_position_embeddings") ?
            "max_position_embeddings" : "model_max_length"), dimension(text, "d_rel"),
        dimension(text, "sconv_kernel_size"), window,
        {dimension(text, "num_attention_heads"), dimension(text, "num_key_value_heads"),
            dimension(text, "head_dim"), dimension(text, "rel_extent")},
        {dimension(text, "swa_num_attention_heads"), dimension(text, "swa_num_key_value_heads"),
            dimension(text, "swa_head_dim"), window},
        std::vector<bool>(layers, false), static_cast<int>(dense), dimension(text, "dense_intermediate_size"),
        dimension(text, "intermediate_size"), dimension(text, "n_routed_experts"),
        dimension(text, "n_shared_experts"), dimension(text, "num_experts_per_tok"),
        is_set(text, "log_scaling_n_floor") ? dimension(text, "log_scaling_n_floor") : 0,
        text.value("log_scaling_alpha", 0.1f), text.value("rms_norm_eps", 1e-6f),
        text.at("route_scale").get<float>(), text.at("logits_mup_width_multiplier").get<float>(),
    };
    const auto& local = text.at("local_layer_ids");
    if (!local.is_array()) {
        throw std::runtime_error("Expected Inkling sliding layer indices");
    }
    for (const auto& entry : local) {
        const auto layer = unsigned_integer(entry);
        if (layer >= result.sliding_layers.size() || result.sliding_layers[layer]) {
            throw std::runtime_error("Invalid Inkling sliding layer index");
        }
        result.sliding_layers[layer] = true;
    }
    const auto integer_max = std::numeric_limits<int>::max();
    for (const auto& attention : {result.full, result.sliding}) {
        if (attention.heads % attention.kv_heads || attention.heads > integer_max / attention.head_dim ||
            attention.heads > integer_max / result.relative_dim) {
            throw std::runtime_error("Invalid Inkling grouped-query attention dimensions");
        }
    }
    if (result.output_vocab > result.vocab || result.experts_per_token > result.routed_experts ||
        result.routed_experts > integer_max - result.shared_experts || result.dense_width > integer_max / 2 ||
        result.expert_width > integer_max / 2 || !std::isfinite(result.log_scale_alpha) || result.log_scale_alpha < 0) {
        throw std::runtime_error("Invalid Inkling expert dimensions, vocabulary or attention scale");
    }
    for (float scale : {result.route_scale, result.output_scale, result.eps}) {
        if (!std::isfinite(scale) || scale <= 0) {
            throw std::runtime_error("Invalid Inkling normalization or projection scale");
        }
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
    std::map<std::string, std::shared_ptr<const MappedTensor::Mapping>> mappings;
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

MappedTensor Checkpoint::map(const std::string& name, std::initializer_list<int> shape) {
    const auto file = files_->weight_map.is_null() ? "model.safetensors" :
        files_->weight_map.at(name).get<std::string>();
    const auto path = files_->directory / file;
    const auto layout = files_->shards.try_emplace(file, path).first->second.layout(name, shape);
    auto& mapping = files_->mappings[file];
    if (!mapping) {
        mapping = std::make_shared<const MappedTensor::Mapping>(path);
    }
    const auto value_size = layout.dtype == MappedTensor::Dtype::fp32 ? sizeof(float) : sizeof(std::uint16_t);
    if (layout.start > mapping->size || layout.count > (mapping->size - layout.start) / value_size) {
        throw std::runtime_error("Invalid mapped data offsets for " + name);
    }
    MappedTensor tensor;
    tensor.mapping_ = mapping;
    tensor.start_ = layout.start;
    tensor.count_ = layout.count;
    tensor.dtype_ = layout.dtype;
    return tensor;
}

}  // namespace vaidy
