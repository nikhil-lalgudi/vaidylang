#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace vaidy {

struct InklingAttentionConfig {
    int heads;
    int kv_heads;
    int head_dim;
    int relative_extent;
};

struct InklingConfig {
    int vocab;
    int output_vocab;
    int hidden;
    int context;
    int relative_dim;
    int kernel;
    int window;
    InklingAttentionConfig full;
    InklingAttentionConfig sliding;
    std::vector<bool> sliding_layers;
    int dense_layers;
    int dense_width;
    int expert_width;
    int routed_experts;
    int shared_experts;
    int experts_per_token;
    int log_scale_floor;
    float log_scale_alpha;
    float eps;
    float route_scale;
    float output_scale;
};

InklingConfig load_inkling_config(const std::filesystem::path& directory);

class Inkling {
    struct Weights;

public:
    class Request {
        struct Buffers;
        friend class Inkling;

    public:
        ~Request();
        int length() const noexcept;

    private:
        Request(std::shared_ptr<const Weights> weights, int capacity);
        std::unique_ptr<Buffers> buffers_;
    };

    explicit Inkling(const std::filesystem::path& checkpoint);
    std::unique_ptr<Request> new_request(int capacity) const;
    void append(Request& request, std::span<const int> tokens, std::span<float> logits) const;

private:
    std::shared_ptr<const Weights> weights_;
};

}  // namespace vaidy
