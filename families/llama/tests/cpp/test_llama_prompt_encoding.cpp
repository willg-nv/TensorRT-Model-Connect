/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/llama/runtime/chat_templates.h"

#include <algorithm>
#include <iostream>

int main() {
    const std::string json = R"({
      "model": {"type":"BPE", "vocab": {
        "<unk>":0,"<s>":1,"</s>":2,"▁":3,"h":4,"e":5,"\n":6,
        "<|user|>":7,"<|assistant|>":8,"<|begin_of_text|>":9},"merges":[]},
      "added_tokens": [
        {"id":1,"content":"<s>","special":true},
        {"id":2,"content":"</s>","special":true},
        {"id":7,"content":"<|user|>","special":true},
        {"id":8,"content":"<|assistant|>","special":true},
        {"id":9,"content":"<|begin_of_text|>","special":true}],
      "normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},
      "post_processor":{"type":"TemplateProcessing","single":[
        {"SpecialToken":{"id":"<s>","type_id":0}},
        {"Sequence":{"id":"A","type_id":0}}]}
    })";
    auto tokenizer = trtmc::CreateBpeTokenizer(json.data(), json.size(), true);
    int failures = 0;
    auto check = [&](bool ok, const char* name) {
        if (!ok) {
            std::cerr << "FAIL: " << name << '\n';
            ++failures;
        }
    };
    const std::vector<std::int32_t> raw{1, 4, 5};
    const std::vector<std::int32_t> chat{7, 6, 4, 5, 2, 6, 8, 6};
    check(trtmc::llama_encode_prompt(*tokenizer, "zephyr", "he", false, false) == raw,
          "raw prompt retains checkpoint special-token framing");
    check(trtmc::llama_encode_prompt(*tokenizer, "zephyr", "he", true, false) == chat,
          "chat prompt uses complete template without tokenizer BOS");
    check(trtmc::llama_encode_prompt(*tokenizer, "zephyr", "he", false, false) == raw,
          "chat encoding does not mutate subsequent raw encoding");
    check(trtmc::llama_encode_prompt(*tokenizer, "zephyr", "", false, false) ==
              std::vector<std::int32_t>{1},
          "empty raw input retains the checkpoint BOS frame");
    check(tokenizer->encode("", false).empty(),
          "empty input without special tokens has no synthetic whitespace");
    const auto embedded = trtmc::llama_encode_prompt(*tokenizer, "llama3", "he", true, true);
    check(!embedded.empty() && embedded.front() == 9 &&
              std::count(embedded.begin(), embedded.end(), 9) == 1 &&
              std::find(embedded.begin(), embedded.end(), 1) == embedded.end(),
          "template-owned BOS is preserved exactly once");
    return failures == 0 ? 0 : 1;
}
