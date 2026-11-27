#include "qwen3.hpp"

#include <algorithm>
#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include <nlohmann/json.hpp>

namespace {

int parse_integer(std::string_view text) {
    int value;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument("Invalid integer: " + std::string(text));
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: vaidylang-qwen3 CHECKPOINT NEW_TOKENS ID [ID ...]\n";
        return 2;
    }
    try {
        const int count = parse_integer(argv[2]);
        const int prompt_size = argc - 3;
        if (count < 1) {
            throw std::invalid_argument("Use at least one output token");
        }
        std::vector<int> prompt;
        prompt.reserve(prompt_size);
        for (int i = 3; i < argc; ++i) {
            prompt.push_back(parse_integer(argv[i]));
        }
        const vaidy::Qwen3 model(argv[1]);
        if (prompt_size > model.config().context || count - 1 > model.config().context - prompt_size) {
            throw std::invalid_argument("Prompt and generated tokens exceed the model context limit");
        }
        auto request = model.new_request(prompt_size + count - 1);
        std::vector<float> logits(model.config().vocab);
        std::vector<int> output;
        output.reserve(count);
        model.append(*request, prompt, logits);
        for (int i = 0; i < count; ++i) {
            output.push_back(static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin()));
            if (i + 1 < count) {
                model.append(*request, std::span<const int>(&output.back(), 1), logits);
            }
        }
        std::cout << nlohmann::json(output).dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
