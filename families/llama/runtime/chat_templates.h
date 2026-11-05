/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/llama/runtime/tokenizer.h"

#include <string>

namespace trtmc {

std::string llama_detect_chat_template_format(const std::string& jinja_template);
std::string llama_apply_chat_template(const std::string& format, const std::string& prompt,
                                      bool enable_thinking = true);
std::vector<std::int32_t> llama_encode_prompt(const ITokenizer& tokenizer,
                                              const std::string& format, const std::string& prompt,
                                              bool use_chat_template, bool enable_thinking);

} // namespace trtmc
