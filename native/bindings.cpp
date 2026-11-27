#include "qwen3.hpp"

#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>

namespace {

thread_local std::array<char, 1024> last_error{};

template <typename Function>
auto checked(Function function) noexcept -> decltype(function()) {
    try {
        return function();
    } catch (const std::exception& error) {
        std::snprintf(last_error.data(), last_error.size(), "%s", error.what());
    } catch (...) {
        std::snprintf(last_error.data(), last_error.size(), "Unknown native inference error");
    }
    return {};
}

}  // namespace

extern "C" {

#define VAIDY_EXPORT __attribute__((visibility("default")))

VAIDY_EXPORT const char* vaidy_last_error() noexcept {
    return last_error.data();
}

VAIDY_EXPORT vaidy::Qwen3* vaidy_qwen3_load(const char* directory) noexcept {
    return checked([&] { return new vaidy::Qwen3(directory); });
}

VAIDY_EXPORT void vaidy_qwen3_free(vaidy::Qwen3* model) noexcept {
    delete model;
}

VAIDY_EXPORT vaidy::Qwen3::Request* vaidy_qwen3_request(const vaidy::Qwen3* model, int capacity) noexcept {
    return checked([&] { return model->new_request(capacity).release(); });
}

VAIDY_EXPORT void vaidy_qwen3_request_free(vaidy::Qwen3::Request* request) noexcept {
    delete request;
}

VAIDY_EXPORT int vaidy_qwen3_length(const vaidy::Qwen3::Request* request) noexcept {
    return request->length();
}

VAIDY_EXPORT bool vaidy_qwen3_append(const vaidy::Qwen3* model, vaidy::Qwen3::Request* request,
    const int* tokens, std::size_t steps, float* logits, std::size_t vocab) noexcept {
    return checked([&] {
        model->append(*request, {tokens, steps}, {logits, vocab});
        return true;
    });
}

}  // extern "C"
