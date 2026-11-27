#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace vaidy {

struct Qwen3Config {
    int vocab;
    int hidden;
    int intermediate;
    int layers;
    int query_heads;
    int kv_heads;
    int head_dim;
    int context;
    float norm_eps;
    float rope_theta;
    bool tied;
};

class Qwen3 {
    struct Weights;

public:
    class Request {
        struct Buffers;
        friend class Qwen3;

    public:
        ~Request();
        int length() const noexcept;

    private:
        Request(std::shared_ptr<const Weights> weights, int capacity);
        std::unique_ptr<Buffers> buffers_;
    };

    explicit Qwen3(const std::filesystem::path& checkpoint);
    const Qwen3Config& config() const noexcept;
    std::unique_ptr<Request> new_request(int capacity) const;

    // Writes the last input token's next-token logits. Input and output must not overlap.
    void append(Request& request, std::span<const int> tokens, std::span<float> logits) const;

private:
    Qwen3Config config_;
    std::shared_ptr<const Weights> weights_;
};

}  // namespace vaidy
