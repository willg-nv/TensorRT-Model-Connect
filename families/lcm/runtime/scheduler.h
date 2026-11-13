/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc::lcm {
class Scheduler {
  public:
    Scheduler(std::vector<float> alphas, int32_t original_steps, float timestep_scaling)
        : alphas_(std::move(alphas)), original_steps_(original_steps), scaling_(timestep_scaling) {
        if (original_steps_ <= 0 || original_steps_ > static_cast<int32_t>(alphas_.size()) ||
            !std::isfinite(scaling_) || scaling_ <= 0)
            throw std::invalid_argument("Invalid LCM scheduler configuration");
        for (const auto alpha : alphas_)
            if (!std::isfinite(alpha) || alpha <= 0 || alpha > 1)
                throw std::invalid_argument("LCM cumulative alphas must be in (0, 1]");
    }
    std::vector<int32_t> timesteps(int32_t steps) const {
        if (steps <= 0 || steps > original_steps_)
            throw std::invalid_argument("LCM inference steps exceed the distillation schedule");
        const int32_t interval = static_cast<int32_t>(alphas_.size()) / original_steps_;
        std::vector<int32_t> schedule;
        for (int32_t i = 0; i < steps; ++i) {
            const auto selected = static_cast<int32_t>(static_cast<int64_t>(i) * original_steps_ / steps);
            schedule.push_back((original_steps_ - selected) * interval - 1);
        }
        return schedule;
    }
    void step(const float* epsilon, float* sample, std::size_t count, int32_t timestep,
              int32_t previous, const float* noise) const {
        const float alpha = alphas_.at(timestep);
        const float scaled = timestep * scaling_;
        const float c_skip = 0.25F / (scaled * scaled + 0.25F);
        const float c_out = scaled / std::sqrt(scaled * scaled + 0.25F);
        const float current_alpha = std::sqrt(alpha), current_beta = std::sqrt(1 - alpha);
        const float next_alpha = previous >= 0 ? std::sqrt(alphas_.at(previous)) : 1;
        const float next_beta = previous >= 0 ? std::sqrt(1 - alphas_.at(previous)) : 0;
        if (previous >= 0 && !noise) throw std::invalid_argument("LCM intermediate step needs noise");
        for (std::size_t i = 0; i < count; ++i) {
            const float original = (sample[i] - current_beta * epsilon[i]) / current_alpha;
            const float denoised = c_out * original + c_skip * sample[i];
            sample[i] = previous >= 0 ? next_alpha * denoised + next_beta * noise[i] : denoised;
        }
    }
  private:
    std::vector<float> alphas_;
    int32_t original_steps_;
    float scaling_;
};
} // namespace trtmc::lcm
