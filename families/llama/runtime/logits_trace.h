/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

namespace trtmc {

// Optional diagnostic at the sampler boundary. Never changes scores or tokens.
// The caller supplies a private destination; disabled tracing performs no I/O.
class LlamaLogitsTrace {
public:
    explicit LlamaLogitsTrace(const char* path) {
        if (path && *path) {
            stream_.open(path, std::ios::out | std::ios::trunc);
            if (!stream_)
                throw std::runtime_error("Cannot open Llama logits diagnostic");
            stream_ << std::setprecision(std::numeric_limits<float>::max_digits10);
        }
    }

    void record(const std::vector<int32_t>& prefix, const std::vector<float>& logits) {
        if (!stream_.is_open())
            return;
        stream_ << "{\"prefix_token_ids\":[";
        for (std::size_t i = 0; i < prefix.size(); ++i) {
            if (i)
                stream_ << ',';
            stream_ << prefix[i];
        }
        stream_ << "],\"logits\":[";
        for (std::size_t i = 0; i < logits.size(); ++i) {
            if (i)
                stream_ << ',';
            if (std::isfinite(logits[i]))
                stream_ << logits[i];
            else
                stream_ << "null"; // Preserve failure evidence as valid JSON.
        }
        stream_ << "]}\n";
        stream_.flush();
        if (!stream_)
            throw std::runtime_error("Cannot write Llama logits diagnostic");
    }

private:
    std::ofstream stream_;
};

} // namespace trtmc
