/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lcm/runtime/scheduler.h"
#include <iostream>
#include <nlohmann/json.hpp>

int main() {
    nlohmann::json input;
    std::cin >> input;
    trtmc::lcm::Scheduler scheduler(input.at("alphas").get<std::vector<float>>(),
        input.at("original_steps"), input.at("scaling"));
    const auto timesteps = scheduler.timesteps(input.at("steps"));
    auto sample = input.at("sample").get<std::vector<float>>();
    nlohmann::json states = nlohmann::json::array();
    for (std::size_t i = 0; i < timesteps.size(); ++i) {
        const auto epsilon = input.at("epsilon").at(i).get<std::vector<float>>();
        const auto noise = input.at("noise").at(i).get<std::vector<float>>();
        if (epsilon.size() != sample.size() || noise.size() != sample.size()) return 2;
        scheduler.step(epsilon.data(), sample.data(), sample.size(), timesteps[i],
            i + 1 < timesteps.size() ? timesteps[i + 1] : -1, noise.data());
        states.push_back(sample);
    }
    std::cout << nlohmann::json{{"timesteps", timesteps}, {"states", states}}.dump() << '\n';
}
