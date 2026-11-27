#include "qwen3.hpp"

#include "bf16_projection.hpp"
#include "checkpoint.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#if defined(__aarch64__)
#include <dispatch/dispatch.h>
#endif
#else
#include <cblas.h>
#endif

namespace vaidy {
namespace {

constexpr int prefill_block = 128;
[[maybe_unused]] constexpr std::size_t wide_matrix_elements = 1 << 20;

struct Matrix {
    Matrix(int rows, int columns) : rows(rows), columns(columns), weight_count(elements({rows, columns})) {}

    void read(Checkpoint& checkpoint, const std::string& name, bool retain_bf16 = true) {
        read_rows(checkpoint, name, 0, rows, retain_bf16);
    }

    // Append projection segments in row order so they share one [out, in] matrix.
    void read_rows(Checkpoint& checkpoint, const std::string& name, int start, int count,
                   [[maybe_unused]] bool retain_bf16 = true) {
        const auto offset = static_cast<std::size_t>(start) * columns;
        const auto length = elements({count, columns});
#if defined(__APPLE__) && defined(__aarch64__) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (retain_bf16 && data.empty() && rows >= 512 && weight_count >= wide_matrix_elements) {
            if (__builtin_available(macOS 15.0, *)) {
                if (!packed) {
                    packed = std::make_unique_for_overwrite<std::uint16_t[]>(weight_count);
                }
                if (checkpoint.read_bf16(name, {count, columns}, {packed.get() + offset, length})) {
                    return;
                }
                // A mixed matrix falls back to FP32, including its previously loaded rows.
                data.resize(weight_count);
                expand_bf16(packed.get(), data.data(), offset);
                packed.reset();
            }
        }
#endif
        if (data.empty()) {
            data.resize(weight_count);
        }
        checkpoint.read(name, {count, columns}, std::span(data).subspan(offset, length));
    }

    std::size_t workspace_size() const noexcept {
        return packed ? weight_count : 0;
    }

    void project(int steps, const float* input, float* output, float beta = 0, float* workspace = nullptr) const {
#if defined(__APPLE__) && defined(__aarch64__) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        // Larger prefill blocks already benefit from the BLAS scheduler.
        if (steps <= 16 && rows >= 512 && weight_count >= wide_matrix_elements) {
            if (__builtin_available(macOS 15.0, *)) {
                ProjectionJob job{*this, steps, input, output, beta, workspace};
                dispatch_apply_f(projection_workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                    &job, project_job);
                return;
            }
        }
#endif
        project_rows(steps, input, output, beta, workspace, 0, rows);
    }

    int rows;
    int columns;
    std::vector<float> data;

private:
    std::size_t weight_count;
    std::unique_ptr<std::uint16_t[]> packed;

#if defined(__APPLE__) && defined(__aarch64__)
    static void expand_bf16(const std::uint16_t* source, float* output, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            output[i] = std::bit_cast<float>(std::uint32_t(source[i]) << 16);
        }
    }
#endif

    void project_rows(int steps, const float* input, float* output, float beta,
                      [[maybe_unused]] float* workspace, int first, int count) const {
        const auto offset = std::size_t(first) * columns;
        const float* weights;
#if defined(__APPLE__) && defined(__aarch64__)
        if (packed) {
            if (steps == 1) {
                project_bf16_rows(reinterpret_cast<const std::byte*>(packed.get() + offset),
                    count, columns, input, output + first, beta);
                return;
            }
            expand_bf16(packed.get() + offset, workspace + offset, std::size_t(count) * columns);
            weights = workspace + offset;
        } else
#endif
        {
            weights = data.data() + offset;
        }
        if (steps == 1) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, count, columns, 1.f, weights, columns,
                input, 1, beta, output + first, 1);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, steps, count, columns, 1.f,
                input, columns, weights, columns, beta, output + first, rows);
        }
    }

#if defined(__APPLE__) && defined(__aarch64__) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    static constexpr std::size_t projection_workers = 4;

    struct ProjectionJob {
        const Matrix& matrix;
        int steps;
        const float* input;
        float* output;
        float beta;
        float* workspace;
    };

    static void project_job(void* context, std::size_t index) API_AVAILABLE(macos(15.0)) {
        const auto& job = *static_cast<const ProjectionJob*>(context);
        const auto row_count = static_cast<std::size_t>(job.matrix.rows);
        const int first = static_cast<int>(row_count * index / projection_workers);
        const int end = static_cast<int>(row_count * (index + 1) / projection_workers);
        // Each worker owns disjoint output columns and runs a single BLAS job.
        const auto saved = BLASGetThreading();
        const bool changed = saved != BLAS_THREADING_SINGLE_THREADED
            && BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) == 0;
        job.matrix.project_rows(job.steps, job.input, job.output, job.beta, job.workspace, first, end - first);
        if (changed) {
            BLASSetThreading(saved);
        }
    }
#endif
};

struct TokenTable {
    TokenTable(int rows, int columns) : rows(rows), columns(columns) {}

    void read(Checkpoint& checkpoint, const std::string& name) {
#if defined(__APPLE__) && defined(__aarch64__)
        // Small tables fit in cache and run faster with the ordinary BLAS path.
        constexpr std::size_t min_packed_elements = 1 << 24;
        if (elements({rows, columns}) >= min_packed_elements) {
            auto tensor = checkpoint.map(name, {rows, columns});
            if (!tensor.bf16_bytes().empty()) {
                packed = std::move(tensor);
                return;
            }
        }
#endif
        dense = std::make_unique<Matrix>(rows, columns);
        dense->read(checkpoint, name, false);
    }

    void embed(int token, float* output) const {
        const auto offset = static_cast<std::size_t>(token) * columns;
        if (dense) {
            std::copy_n(dense->data.data() + offset, columns, output);
        } else {
            packed.read(offset, {output, static_cast<std::size_t>(columns)});
        }
    }

    void project(const float* input, float* output) const {
#if defined(__APPLE__) && defined(__aarch64__)
        if (!dense) {
            project_bf16_vocabulary(packed.bf16_bytes().data(), rows, columns, input, output);
            return;
        }
#endif
        dense->project(1, input, output);
    }

    int rows;
    int columns;
    std::unique_ptr<Matrix> dense;
    MappedTensor packed;
};

struct Layer {
    Layer(Checkpoint& checkpoint, const Qwen3Config& c, int index)
        : input_norm(c.hidden), query_norm(c.head_dim), key_norm(c.head_dim), post_norm(c.hidden),
          qkv((c.query_heads + 2 * c.kv_heads) * c.head_dim, c.hidden),
          attention_output(c.hidden, c.query_heads * c.head_dim),
          gate_up(2 * c.intermediate, c.hidden), down(c.hidden, c.intermediate) {
        const auto prefix = "model.layers." + std::to_string(index) + ".";
        const int query = c.query_heads * c.head_dim;
        const int kv = c.kv_heads * c.head_dim;
        checkpoint.read(prefix + "input_layernorm.weight", {c.hidden}, input_norm);
        checkpoint.read(prefix + "self_attn.q_norm.weight", {c.head_dim}, query_norm);
        checkpoint.read(prefix + "self_attn.k_norm.weight", {c.head_dim}, key_norm);
        checkpoint.read(prefix + "post_attention_layernorm.weight", {c.hidden}, post_norm);
        qkv.read_rows(checkpoint, prefix + "self_attn.q_proj.weight", 0, query);
        qkv.read_rows(checkpoint, prefix + "self_attn.k_proj.weight", query, kv);
        qkv.read_rows(checkpoint, prefix + "self_attn.v_proj.weight", query + kv, kv);
        attention_output.read(checkpoint, prefix + "self_attn.o_proj.weight");
        gate_up.read_rows(checkpoint, prefix + "mlp.gate_proj.weight", 0, c.intermediate);
        gate_up.read_rows(checkpoint, prefix + "mlp.up_proj.weight", c.intermediate, c.intermediate);
        down.read(checkpoint, prefix + "mlp.down_proj.weight");
    }

    std::vector<float> input_norm, query_norm, key_norm, post_norm;
    Matrix qkv, attention_output, gate_up, down;
};

float rms_factor(const float* x, int width, float eps) {
    float sum = 0;
    for (int i = 0; i < width; ++i) {
        sum += x[i] * x[i];
    }
    return 1.f / std::sqrt(sum / static_cast<float>(width) + eps);
}

void normalize(const float* input, float* output, const std::vector<float>& weights, float eps) {
    const int width = static_cast<int>(weights.size());
    const float factor = rms_factor(input, width, eps);
    for (int i = 0; i < width; ++i) {
        output[i] = input[i] * factor * weights[i];
    }
}

void normalize_rotate(float* vector, const std::vector<float>& weights, const float* cos,
                      const float* sin, float eps) {
    const int width = static_cast<int>(weights.size());
    const int half = width / 2;
    const float factor = rms_factor(vector, width, eps);
    for (int i = 0; i < half; ++i) {
        const float a = vector[i] * factor * weights[i];
        const float b = vector[i + half] * factor * weights[i + half];
        vector[i] = a * cos[i] - b * sin[i];
        vector[i + half] = b * cos[i] + a * sin[i];
    }
}

void softmax(float* scores, int visible, int total) {
#ifdef __APPLE__
    // Long rows amortize the vector-library calls.
    if (visible >= 128) {
        const auto count = static_cast<vDSP_Length>(visible);
        float maximum, sum;
        vDSP_maxv(scores, 1, &maximum, count);
        const float shift = -maximum;
        vDSP_vsadd(scores, 1, &shift, scores, 1, count);
        vvexpf(scores, scores, &visible);
        vDSP_sve(scores, 1, &sum, count);
        const float inverse = 1.f / sum;
        vDSP_vsmul(scores, 1, &inverse, scores, 1, count);
        std::fill(scores + visible, scores + total, 0.f);
        return;
    }
#endif
    const float max_score = *std::max_element(scores, scores + visible);
    float sum = 0;
    for (int i = 0; i < visible; ++i) {
        scores[i] = std::exp(scores[i] - max_score);
        sum += scores[i];
    }
    const float inverse = 1.f / sum;
    for (int i = 0; i < visible; ++i) {
        scores[i] *= inverse;
    }
    std::fill(scores + visible, scores + total, 0.f);
}

float* row(std::vector<float>& data, int index, int width) {
    return data.data() + static_cast<std::size_t>(index) * width;
}

}  // namespace

struct Qwen3::Weights {
    Weights(const Qwen3Config& c, const std::filesystem::path& directory)
        : embeddings(c.vocab, c.hidden), final_norm(c.hidden), frequencies(c.head_dim / 2) {
        Checkpoint checkpoint(directory);
        embeddings.read(checkpoint, "model.embed_tokens.weight");
        checkpoint.read("model.norm.weight", {c.hidden}, final_norm);
        if (!c.tied) {
            head = std::make_unique<TokenTable>(c.vocab, c.hidden);
            head->read(checkpoint, "lm_head.weight");
        }
        layers.reserve(c.layers);
        for (int i = 0; i < c.layers; ++i) {
            const auto& layer = layers.emplace_back(checkpoint, c, i);
            workspace_size = std::max({workspace_size, layer.qkv.workspace_size(), layer.attention_output.workspace_size(),
                layer.gate_up.workspace_size(), layer.down.workspace_size()});
        }
        for (int i = 0; i < c.head_dim / 2; ++i) {
            frequencies[i] = std::pow(c.rope_theta, -2.f * i / c.head_dim);
        }
    }

    TokenTable embeddings;
    std::unique_ptr<TokenTable> head;
    std::vector<float> final_norm, frequencies;
    std::vector<Layer> layers;
    std::size_t workspace_size = 0;
};

struct Qwen3::Request::Buffers {
    Buffers(std::shared_ptr<const Weights> model, int maximum)
        : weights(std::move(model)), capacity(maximum) {}

    void forward(const Qwen3Config& c, std::span<const int> ids, std::span<float> logits) {
        const int steps = static_cast<int>(ids.size());
        const int end = length + steps;
        const int query = c.query_heads * c.head_dim;
        const int kv = c.kv_heads * c.head_dim;
        const int qkv_width = query + 2 * kv;
        const int half = c.head_dim / 2;
        const auto cache_elements = elements({c.layers, c.kv_heads, capacity, c.head_dim});
        // Pages are touched as tokens arrive, so a large capacity does not clear the whole cache.
        if (!keys) {
            auto new_keys = std::make_unique_for_overwrite<float[]>(cache_elements);
            values = std::make_unique_for_overwrite<float[]>(cache_elements);
            keys = std::move(new_keys);
        }
        // Prefill reuses one FP32 expansion across layers. Decode reads BF16 directly.
        if (steps > 1 && weights->workspace_size && !projection_workspace) {
            projection_workspace = std::make_unique_for_overwrite<float[]>(weights->workspace_size);
        }
        float* workspace = projection_workspace.get();
        hidden.resize(elements({steps, c.hidden}));
        normalized.resize(hidden.size());
        projections.resize(elements({steps, qkv_width}));
        joined.resize(elements({steps, query}));
        gate_up.resize(elements({steps, 2, c.intermediate}));
        gated.resize(elements({steps, c.intermediate}));
        cosine.resize(elements({steps, half}));
        sine.resize(cosine.size());
        scores.resize(elements({steps, end}));
        for (int t = 0; t < steps; ++t) {
            weights->embeddings.embed(ids[t], row(hidden, t, c.hidden));
            for (int d = 0; d < half; ++d) {
                const float angle = static_cast<float>(length + t) * weights->frequencies[d];
                row(cosine, t, half)[d] = std::cos(angle);
                row(sine, t, half)[d] = std::sin(angle);
            }
        }

        for (int layer_index = 0; layer_index < c.layers; ++layer_index) {
            const auto& layer = weights->layers[layer_index];
            for (int t = 0; t < steps; ++t) {
                normalize(row(hidden, t, c.hidden), row(normalized, t, c.hidden), layer.input_norm, c.norm_eps);
            }
            layer.qkv.project(steps, normalized.data(), projections.data(), 0.f, workspace);
            const auto cache_stride = elements({capacity, c.head_dim});
            const auto layer_offset = static_cast<std::size_t>(layer_index) * c.kv_heads * cache_stride;
            for (int t = 0; t < steps; ++t) {
                float* q = row(projections, t, qkv_width);
                float* k = q + query;
                const float* v = k + kv;
                const float* cos = row(cosine, t, half);
                const float* sin = row(sine, t, half);
                for (int head = 0; head < c.query_heads; ++head) {
                    normalize_rotate(q + head * c.head_dim, layer.query_norm, cos, sin, c.norm_eps);
                }
                for (int head = 0; head < c.kv_heads; ++head) {
                    const auto offset = layer_offset + head * cache_stride +
                        static_cast<std::size_t>(length + t) * c.head_dim;
                    normalize_rotate(k + head * c.head_dim, layer.key_norm, cos, sin, c.norm_eps);
                    std::copy_n(k + head * c.head_dim, c.head_dim, keys.get() + offset);
                    std::copy_n(v + head * c.head_dim, c.head_dim, values.get() + offset);
                }
            }

            const int heads_per_group = c.query_heads / c.kv_heads;
            const float scale = 1.f / std::sqrt(static_cast<float>(c.head_dim));
            // Reuse a single score tile for every head. The only masked cells are
            // future tokens within the current block, since the prefix is cached.
            for (int head = 0; head < c.query_heads; ++head) {
                const auto offset = layer_offset + (head / heads_per_group) * cache_stride;
                const float* k = keys.get() + offset;
                const float* v = values.get() + offset;
                const float* q = projections.data() + head * c.head_dim;
                float* attended = joined.data() + head * c.head_dim;
                if (steps == 1) {
                    cblas_sgemv(CblasRowMajor, CblasNoTrans, end, c.head_dim, scale, k, c.head_dim,
                        q, 1, 0.f, scores.data(), 1);
                    softmax(scores.data(), end, end);
                    cblas_sgemv(CblasRowMajor, CblasTrans, end, c.head_dim, 1.f, v, c.head_dim,
                        scores.data(), 1, 0.f, attended, 1);
                } else {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, steps, end, c.head_dim, scale,
                        q, qkv_width, k, c.head_dim, 0.f, scores.data(), end);
                    for (int t = 0; t < steps; ++t) {
                        softmax(row(scores, t, end), length + t + 1, end);
                    }
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, steps, c.head_dim, end, 1.f,
                        scores.data(), end, v, c.head_dim, 0.f, attended, query);
                }
            }
            layer.attention_output.project(steps, joined.data(), hidden.data(), 1.f, workspace);
            for (int t = 0; t < steps; ++t) {
                normalize(row(hidden, t, c.hidden), row(normalized, t, c.hidden), layer.post_norm, c.norm_eps);
            }
            layer.gate_up.project(steps, normalized.data(), gate_up.data(), 0.f, workspace);
            for (int t = 0; t < steps; ++t) {
                const float* gate = row(gate_up, t, 2 * c.intermediate);
                const float* up = gate + c.intermediate;
                float* output = row(gated, t, c.intermediate);
                for (int i = 0; i < c.intermediate; ++i) {
                    output[i] = (gate[i] / (1.f + std::exp(-gate[i]))) * up[i];
                }
            }
            layer.down.project(steps, gated.data(), hidden.data(), 1.f, workspace);
        }
        if (!logits.empty()) {
            normalize(row(hidden, steps - 1, c.hidden), normalized.data(), weights->final_norm, c.norm_eps);
            const auto& head = weights->head ? *weights->head : weights->embeddings;
            head.project(normalized.data(), logits.data());
        }
        length = end;
    }

    std::shared_ptr<const Weights> weights;
    int capacity;
    int length = 0;
    std::unique_ptr<float[]> keys, values, projection_workspace;
    std::vector<float> hidden, normalized, projections, joined, gate_up, gated, cosine, sine, scores;
};

Qwen3::Request::Request(std::shared_ptr<const Weights> weights, int capacity)
    : buffers_(std::make_unique<Buffers>(std::move(weights), capacity)) {}

Qwen3::Request::~Request() = default;

int Qwen3::Request::length() const noexcept {
    return buffers_->length;
}

Qwen3::Qwen3(const std::filesystem::path& directory)
    : config_(load_qwen3_config(directory)), weights_(std::make_shared<Weights>(config_, directory)) {}

const Qwen3Config& Qwen3::config() const noexcept {
    return config_;
}

std::unique_ptr<Qwen3::Request> Qwen3::new_request(int capacity) const {
    if (capacity < 1 || capacity > config_.context) {
        throw std::invalid_argument("KV capacity is outside the model context limit");
    }
    return std::unique_ptr<Request>(new Request(weights_, capacity));
}

void Qwen3::append(Request& request, std::span<const int> tokens, std::span<float> logits) const {
    auto& state = *request.buffers_;
    if (state.weights != weights_) {
        throw std::invalid_argument("Request belongs to a different model");
    }
    if (tokens.empty() || logits.size() != static_cast<std::size_t>(config_.vocab)) {
        throw std::invalid_argument("Expected a nonempty sequence of token IDs and a vocabulary-sized output");
    }
    if (tokens.size() > static_cast<std::size_t>(state.capacity - state.length)) {
        throw std::invalid_argument("KV cache capacity exceeded");
    }
    if (std::any_of(tokens.begin(), tokens.end(), [this](int token) {
            return token < 0 || token >= config_.vocab;
        })) {
        throw std::invalid_argument("Token ID is outside the vocabulary");
    }
    const int start = state.length;
    try {
        while (tokens.size() > prefill_block) {
            state.forward(config_, tokens.first(prefill_block), {});
            tokens = tokens.subspan(prefill_block);
        }
        state.forward(config_, tokens, logits);
    } catch (...) {
        state.length = start;
        throw;
    }
}

}  // namespace vaidy
