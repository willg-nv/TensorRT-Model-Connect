/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A real public C++ consumer: load once, then issue independent requests.
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc < 10 || (argc - 4) % 6 != 0)
            throw std::runtime_error("expected bundle, runtime, output directory, and request sextuples");
        auto task = trtmc::load_task(argv[1], argv[2]);
        auto* text = dynamic_cast<trtmc::ITextGeneration*>(task.get());
        if (!text || std::string(task->task()) != "text_generation")
            throw std::runtime_error("bundle does not implement text_generation");
        const std::filesystem::path output(argv[3]);
        for (int i = 4; i < argc; i += 6) {
            const auto base = output / argv[i];
            const auto trace = base.string() + ".logits.jsonl";
            if (setenv("TRTMC_LLAMA_LOGITS_TRACE", trace.c_str(), 1) != 0)
                throw std::runtime_error("cannot configure logits capture");
            trtmc::TextGenerationConfig config;
            config.max_new_tokens = std::stoi(argv[i + 2]);
            config.use_chat_template = std::string(argv[i + 3]) == "true";
            config.enable_thinking = false;
            config.text_generation_mode = argv[i + 4];
            // argv[i + 5] is the expected error substring, checked by Python.
            // Keep all outcomes so a failure does not erase subsequent reuse evidence.
            try {
                const auto result = text->generate(argv[i + 1], config);
                std::ofstream ids(base.string() + ".ids");
                for (const auto id : result.token_ids)
                    ids << id << '\n';
                std::ofstream(base.string() + ".text", std::ios::binary) << result.text;
                std::ofstream(base.string() + ".timing") << std::setprecision(17)
                    << result.setup_ms << ' ' << result.prefill_ms << ' ' << result.decode_ms;
            } catch (const std::exception& error) {
                std::ofstream(base.string() + ".error") << error.what();
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
