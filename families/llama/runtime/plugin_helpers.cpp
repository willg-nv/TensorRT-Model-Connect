/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/llama/runtime/plugin_helpers.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::llama {

std::vector<char> require_section(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::string require_text_section(const BundleReader& bundle, std::string_view name) {
    const auto& data = require_section(bundle, name);
    return {data.begin(), data.end()};
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* label) {
    auto engine = backend.create_module(plan.data(), plan.size(), {});
    if (engine == nullptr || !engine->ok())
        throw std::runtime_error(std::string("llama failed to load ") + label);
    engine->set_timing_label(label);
    return engine;
}

std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle) {
    const auto& data = require_section(bundle, "tokenizer.json");
    std::string serialized(data.begin(), data.end());
    if (bundle.find_section("tokenizer_config.json") != nullptr) {
        const auto config =
            nlohmann::json::parse(require_text_section(bundle, "tokenizer_config.json"));
        if (!config.is_object())
            throw std::runtime_error("llama tokenizer_config.json must be an object");
        const auto tokenizer_class = config.value("tokenizer_class", std::string{});
        if ((tokenizer_class == "LlamaTokenizer" || tokenizer_class == "LlamaTokenizerFast") &&
            !config.value("legacy", true)) {
            // LlamaTokenizer's nonlegacy wrapper replaces the serialized legacy
            // normalizer with Metaspace(first). In particular, text after an
            // added token must not acquire a new synthetic prefix space.
            auto document = nlohmann::json::parse(serialized);
            const auto prefix = config.find("add_prefix_space");
            const bool add_prefix_space =
                prefix == config.end() || prefix->is_null() ? true : prefix->get<bool>();
            document["normalizer"] = nullptr;
            document["pre_tokenizer"] = {{"type", "Metaspace"},
                                         {"replacement", "▁"},
                                         {"prepend_scheme", add_prefix_space ? "first" : "never"},
                                         {"split", false}};
            serialized = document.dump();
        }
    }
    auto tokenizer = CreateBpeTokenizer(serialized.data(), serialized.size(), true);
    if (tokenizer == nullptr)
        throw std::runtime_error("llama BPE tokenizer construction failed");
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}

} // namespace trtmc::llama
