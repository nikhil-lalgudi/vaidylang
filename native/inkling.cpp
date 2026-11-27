#include "inkling.hpp"

#include "checkpoint.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

namespace vaidy {
namespace {

using Vector = std::vector<float>;

Vector read_vector(Checkpoint& checkpoint, const std::string& name, std::initializer_list<int> shape) {
    Vector values(elements(shape));
    checkpoint.read(name, shape, values);
    return values;
}

struct Linear {
    MappedTensor storage;
    int rows;
    int columns;

    void project(std::span<const float> input, std::span<float> output, std::span<float> scratch,
                 int expert = 0, float alpha = 1.f, float beta = 0.f) const {
        const auto offset = static_cast<std::size_t>(expert) * rows * columns;
        const auto tile_rows = scratch.size() / columns;
        for (std::size_t first = 0; first < output.size(); first += tile_rows) {
            const auto count = std::min(tile_rows, output.size() - first);
            storage.read(offset + first * columns, scratch.first(count * columns));
            cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(count), columns, alpha, scratch.data(), columns,
                input.data(), 1, beta, output.data() + first, 1);
        }
    }
};

Linear linear(Checkpoint& checkpoint, const std::string& name, int rows, int columns, int experts = 0) {
    auto tensor = experts ? checkpoint.map(name, {experts, rows, columns}) : checkpoint.map(name, {rows, columns});
    return {std::move(tensor), rows, columns};
}

struct Dense {
    Linear gate_up;
    Linear down;
    float scale;
};

struct Experts {
    Linear gate_up;
    Linear down;
};

struct MoE {
    Linear router;
    Vector correction;
    float scale;
    Experts routed;
    Experts shared;
};

struct Layer {
    InklingAttentionConfig attention;
    bool sliding;
    Vector input_norm, post_norm, query_norm, key_norm;
    Vector key_conv, value_conv, attention_conv, mlp_conv;
    Vector relative_bias;
    Linear query, key, value, relative, attention_output;
    std::unique_ptr<Dense> dense;
    std::unique_ptr<MoE> moe;
};

Layer load_layer(Checkpoint& checkpoint, const InklingConfig& config, int index) {
    const bool sliding = config.sliding_layers[index];
    const auto attention = sliding ? config.sliding : config.full;
    const int query = attention.heads * attention.head_dim;
    const int kv = attention.kv_heads * attention.head_dim;
    const int hidden = config.hidden;
    const auto prefix = "model.llm.layers." + std::to_string(index) + ".";
    Layer layer{
        attention, sliding,
        read_vector(checkpoint, prefix + "attn_norm.weight", {hidden}),
        read_vector(checkpoint, prefix + "mlp_norm.weight", {hidden}),
        read_vector(checkpoint, prefix + "attn.q_norm.weight", {attention.head_dim}),
        read_vector(checkpoint, prefix + "attn.k_norm.weight", {attention.head_dim}),
        read_vector(checkpoint, prefix + "attn.k_sconv.weight", {kv, 1, config.kernel}),
        read_vector(checkpoint, prefix + "attn.v_sconv.weight", {kv, 1, config.kernel}),
        read_vector(checkpoint, prefix + "attn_sconv.weight", {hidden, 1, config.kernel}),
        read_vector(checkpoint, prefix + "mlp_sconv.weight", {hidden, 1, config.kernel}),
        read_vector(checkpoint, prefix + "attn.rel_logits_proj.proj", {config.relative_dim, attention.relative_extent}),
        linear(checkpoint, prefix + "attn.wq_du.weight", query, hidden),
        linear(checkpoint, prefix + "attn.wk_dv.weight", kv, hidden),
        linear(checkpoint, prefix + "attn.wv_dv.weight", kv, hidden),
        linear(checkpoint, prefix + "attn.wr_du.weight", attention.heads * config.relative_dim, hidden),
        linear(checkpoint, prefix + "attn.wo_ud.weight", hidden, query),
        nullptr, nullptr,
    };
    if (index < config.dense_layers) {
        const int width = config.dense_width;
        layer.dense = std::make_unique<Dense>(Dense{
            linear(checkpoint, prefix + "mlp.w13_dn.weight", 2 * width, hidden),
            linear(checkpoint, prefix + "mlp.w2_md.weight", hidden, width),
            read_vector(checkpoint, prefix + "mlp.global_scale", {1})[0],
        });
    } else {
        const int routed = config.routed_experts;
        const int shared = config.shared_experts;
        const int width = config.expert_width;
        layer.moe = std::make_unique<MoE>(MoE{
            linear(checkpoint, prefix + "mlp.gate.weight", routed + shared, hidden),
            read_vector(checkpoint, prefix + "mlp.gate.bias", {routed}),
            read_vector(checkpoint, prefix + "mlp.gate.global_scale", {1})[0] * config.route_scale,
            {linear(checkpoint, prefix + "mlp.experts.w13_weight", 2 * width, hidden, routed),
             linear(checkpoint, prefix + "mlp.experts.w2_weight", hidden, width, routed)},
            {linear(checkpoint, prefix + "mlp.shared_experts.shared_w13_weight", 2 * width, hidden, shared),
             linear(checkpoint, prefix + "mlp.shared_experts.shared_w2_weight", hidden, width, shared)},
        });
    }
    return layer;
}

void normalize(std::span<const float> input, std::span<float> output, std::span<const float> weights, float eps) {
    const float squares = std::inner_product(input.begin(), input.end(), input.begin(), 0.f);
    const float factor = 1.f / std::sqrt(squares / static_cast<float>(input.size()) + eps);
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * factor * weights[i];
    }
}

void normalize_heads(std::span<float> values, std::span<const float> weights, float eps) {
    for (std::size_t first = 0; first < values.size(); first += weights.size()) {
        const auto head = values.subspan(first, weights.size());
        normalize(head, head, weights, eps);
    }
}

float log_sigmoid(float value) {
    return std::min(value, 0.f) - std::log1p(std::exp(-std::abs(value)));
}

void swiglu(std::span<float> interleaved, float scale = 1.f) {
    for (std::size_t i = 0; i < interleaved.size() / 2; ++i) {
        const float gate = interleaved[2 * i];
        interleaved[i] = scale * gate * std::exp(log_sigmoid(gate)) * interleaved[2 * i + 1];
    }
}

// Convolution kernels are stored oldest input first. Preserve each pre-convolution input.
void convolve(std::span<float> values, Vector& history, const Vector& weights, int position, int kernel) {
    const auto width = values.size();
    std::copy(values.begin(), values.end(), history.begin() + (position % kernel) * width);
    const int valid = std::min(position + 1, kernel);
    for (std::size_t channel = 0; channel < width; ++channel) {
        for (int lag = 0; lag < valid; ++lag) {
            values[channel] += history[((position - lag) % kernel) * width + channel] *
                weights[channel * kernel + kernel - 1 - lag];
        }
    }
}

struct LayerCache {
    LayerCache(const Layer& layer, const InklingConfig& config, int capacity)
        : slots(layer.sliding ? std::min(config.window, capacity) : capacity),
          kv_width(layer.attention.kv_heads * layer.attention.head_dim),
          key(elements({slots, kv_width})), value(key.size()),
          key_conv(elements({config.kernel, kv_width})), value_conv(key_conv.size()),
          attention_conv(elements({config.kernel, config.hidden})), mlp_conv(attention_conv.size()) {}

    int slots;
    int kv_width;
    Vector key, value, key_conv, value_conv, attention_conv, mlp_conv;
};

struct Workspace {
    Workspace(const InklingConfig& config, int capacity)
        : hidden(config.hidden), normalized(config.hidden), delta(config.hidden),
          query(std::max(config.full.heads * config.full.head_dim, config.sliding.heads * config.sliding.head_dim)),
          relative(std::max(config.full.heads, config.sliding.heads) * config.relative_dim),
          attention(query.size()), scores(capacity),
          activations(2 * std::max(config.dense_width, config.expert_width)),
          router(config.routed_experts + config.shared_experts), choices(config.routed_experts),
          factors(config.experts_per_token + config.shared_experts), order(config.routed_experts),
          tile(std::max<std::size_t>({1'048'576, query.size(), hidden.size(), activations.size() / 2})) {}

    Vector hidden, normalized, delta, query, relative, attention, scores, activations, router, choices, factors;
    std::vector<int> order;
    Vector tile;
};

void attend(const InklingConfig& config, const Layer& layer, LayerCache& cache, Workspace& work, int position) {
    const auto& dims = layer.attention;
    const auto offset = static_cast<std::size_t>(position % cache.slots) * cache.kv_width;
    auto key = std::span(cache.key).subspan(offset, cache.kv_width);
    auto value = std::span(cache.value).subspan(offset, cache.kv_width);
    auto query = std::span(work.query).first(dims.heads * dims.head_dim);
    auto relative = std::span(work.relative).first(dims.heads * config.relative_dim);
    layer.query.project(work.normalized, query, work.tile);
    layer.key.project(work.normalized, key, work.tile);
    layer.value.project(work.normalized, value, work.tile);
    layer.relative.project(work.normalized, relative, work.tile);
    convolve(key, cache.key_conv, layer.key_conv, position, config.kernel);
    convolve(value, cache.value_conv, layer.value_conv, position, config.kernel);
    normalize_heads(query, layer.query_norm, config.eps);
    normalize_heads(key, layer.key_norm, config.eps);

    const int groups = dims.heads / dims.kv_heads;
    const int first = std::max(0, position + 1 - cache.slots);
    const auto scores = std::span(work.scores).first(position - first + 1);
    float scale = 1.f;
    if (!layer.sliding && config.log_scale_floor > 0 && position + 1 > config.log_scale_floor) {
        scale += config.log_scale_alpha * std::log(static_cast<float>(position + 1) / config.log_scale_floor);
    }
    for (int head = 0; head < dims.heads; ++head) {
        const auto q = query.subspan(head * dims.head_dim, dims.head_dim);
        const auto r = relative.subspan(head * config.relative_dim, config.relative_dim);
        const int kv_offset = head / groups * dims.head_dim;
        for (int token = first; token <= position; ++token) {
            const auto slot = static_cast<std::size_t>(token % cache.slots) * cache.kv_width + kv_offset;
            float score = std::inner_product(q.begin(), q.end(), cache.key.begin() + slot, 0.f) / dims.head_dim;
            const int distance = position - token;
            if (distance < dims.relative_extent) {
                for (int d = 0; d < config.relative_dim; ++d) {
                    score += r[d] * layer.relative_bias[static_cast<std::size_t>(d) * dims.relative_extent + distance];
                }
            }
            scores[token - first] = score * scale;
        }
        const float maximum = *std::max_element(scores.begin(), scores.end());
        float denominator = 0.f;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        auto output = std::span(work.attention).subspan(head * dims.head_dim, dims.head_dim);
        std::fill(output.begin(), output.end(), 0.f);
        for (int token = first; token <= position; ++token) {
            const auto slot = static_cast<std::size_t>(token % cache.slots) * cache.kv_width + kv_offset;
            const float probability = scores[token - first] / denominator;
            for (int d = 0; d < dims.head_dim; ++d) {
                output[d] += probability * cache.value[slot + d];
            }
        }
    }
    layer.attention_output.project(std::span(work.attention).first(query.size()), work.delta, work.tile);
}

void feed_forward(const InklingConfig& config, const Layer& layer, Workspace& work) {
    if (layer.dense) {
        const auto& dense = *layer.dense;
        const auto activations = std::span(work.activations).first(2 * config.dense_width);
        dense.gate_up.project(work.normalized, activations, work.tile);
        swiglu(activations);
        dense.down.project(activations.first(config.dense_width), work.delta, work.tile, 0, dense.scale);
        return;
    }
    const auto& moe = *layer.moe;
    moe.router.project(work.normalized, work.router, work.tile);
    for (float& score : work.router) {
        score = log_sigmoid(score);
    }
    for (int i = 0; i < config.routed_experts; ++i) {
        work.choices[i] = std::exp(work.router[i]) + moe.correction[i];
    }
    std::iota(work.order.begin(), work.order.end(), 0);
    const auto end = work.order.begin() + config.experts_per_token;
    std::partial_sort(work.order.begin(), end, work.order.end(), [&](int lhs, int rhs) {
        return work.choices[lhs] == work.choices[rhs] ? lhs < rhs : work.choices[lhs] > work.choices[rhs];
    });
    std::sort(work.order.begin(), end);
    for (int i = 0; i < config.experts_per_token; ++i) {
        work.factors[i] = work.router[work.order[i]];
    }
    std::copy(work.router.begin() + config.routed_experts, work.router.end(),
        work.factors.begin() + config.experts_per_token);
    const float maximum = *std::max_element(work.factors.begin(), work.factors.end());
    float denominator = 0.f;
    for (float& factor : work.factors) {
        factor = std::exp(factor - maximum);
        denominator += factor;
    }
    for (float& factor : work.factors) {
        factor *= moe.scale / denominator;
    }
    std::fill(work.delta.begin(), work.delta.end(), 0.f);
    const auto activations = std::span(work.activations).first(2 * config.expert_width);
    const auto inner = activations.first(config.expert_width);
    for (int i = 0; i < config.experts_per_token; ++i) {
        const int expert = work.order[i];
        moe.routed.gate_up.project(work.normalized, activations, work.tile, expert);
        swiglu(activations);
        moe.routed.down.project(inner, work.delta, work.tile, expert, work.factors[i], 1.f);
    }
    for (int expert = 0; expert < config.shared_experts; ++expert) {
        moe.shared.gate_up.project(work.normalized, activations, work.tile, expert);
        swiglu(activations, work.factors[config.experts_per_token + expert]);
        moe.shared.down.project(inner, work.delta, work.tile, expert, 1.f, 1.f);
    }
}

}  // namespace

struct Inkling::Weights {
    explicit Weights(const std::filesystem::path& directory) : config(load_inkling_config(directory)) {
        Checkpoint checkpoint(directory);
        embedding = checkpoint.map("model.llm.embed.weight", {config.vocab, config.hidden});
        embedding_norm = read_vector(checkpoint, "model.llm.embed_norm.weight", {config.hidden});
        output = linear(checkpoint, "model.llm.unembed.weight", config.vocab, config.hidden);
        output_norm = read_vector(checkpoint, "model.llm.norm.weight", {config.hidden});
        for (std::size_t i = 0; i < config.sliding_layers.size(); ++i) {
            layers.push_back(load_layer(checkpoint, config, static_cast<int>(i)));
        }
    }

    InklingConfig config;
    MappedTensor embedding;
    Vector embedding_norm, output_norm;
    Linear output{};
    std::vector<Layer> layers;
};

struct Inkling::Request::Buffers {
    Buffers(std::shared_ptr<const Weights> model, int limit)
        : weights(std::move(model)), capacity(limit), work(weights->config, capacity) {
        for (const auto& layer : weights->layers) {
            caches.emplace_back(layer, weights->config, capacity);
        }
    }

    void forward(int token) {
        const auto& config = weights->config;
        weights->embedding.read(static_cast<std::size_t>(token) * config.hidden, work.hidden);
        normalize(work.hidden, work.hidden, weights->embedding_norm, config.eps);
        for (std::size_t index = 0; index < caches.size(); ++index) {
            const auto& layer = weights->layers[index];
            auto& cache = caches[index];
            normalize(work.hidden, work.normalized, layer.input_norm, config.eps);
            attend(config, layer, cache, work, length);
            convolve(work.delta, cache.attention_conv, layer.attention_conv, length, config.kernel);
            std::transform(work.hidden.begin(), work.hidden.end(), work.delta.begin(), work.hidden.begin(), std::plus{});
            normalize(work.hidden, work.normalized, layer.post_norm, config.eps);
            feed_forward(config, layer, work);
            convolve(work.delta, cache.mlp_conv, layer.mlp_conv, length, config.kernel);
            std::transform(work.hidden.begin(), work.hidden.end(), work.delta.begin(), work.hidden.begin(), std::plus{});
        }
        ++length;
    }

    std::shared_ptr<const Weights> weights;
    int capacity;
    int length = 0;
    Workspace work;
    std::vector<LayerCache> caches;
};

Inkling::Request::Request(std::shared_ptr<const Weights> weights, int capacity)
    : buffers_(std::make_unique<Buffers>(std::move(weights), capacity)) {}

Inkling::Request::~Request() = default;

int Inkling::Request::length() const noexcept { return buffers_->length; }

Inkling::Inkling(const std::filesystem::path& checkpoint) : weights_(std::make_shared<Weights>(checkpoint)) {}

std::unique_ptr<Inkling::Request> Inkling::new_request(int capacity) const {
    if (capacity < 1 || capacity > weights_->config.context) {
        throw std::invalid_argument("KV capacity is outside the model context limit");
    }
    return std::unique_ptr<Request>(new Request(weights_, capacity));
}

void Inkling::append(Request& request, std::span<const int> tokens, std::span<float> logits) const {
    auto& state = *request.buffers_;
    const auto& config = weights_->config;
    if (state.weights != weights_) {
        throw std::invalid_argument("Request belongs to a different model");
    }
    if (tokens.empty() || logits.size() != static_cast<std::size_t>(config.output_vocab)) {
        throw std::invalid_argument("Expected input tokens and an output matching Inkling's unpadded vocabulary");
    }
    if (tokens.size() > static_cast<std::size_t>(state.capacity - state.length)) {
        throw std::invalid_argument("KV cache capacity exceeded");
    }
    if (std::any_of(tokens.begin(), tokens.end(), [&](int id) { return id < 0 || id >= config.vocab; })) {
        throw std::invalid_argument("Token ID is outside the vocabulary");
    }
    for (int token : tokens) {
        state.forward(token);
    }
    auto& work = state.work;
    normalize(work.hidden, work.normalized, weights_->output_norm, config.eps);
    for (float& value : work.normalized) {
        value /= config.output_scale;
    }
    weights_->output.project(work.normalized, logits, work.tile);
}

}  // namespace vaidy
