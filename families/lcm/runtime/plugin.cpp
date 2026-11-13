/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lcm/runtime/scheduler.h"
#include "families/lcm/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>

namespace trtmc::lcm {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0) throw std::runtime_error(std::string("LCM bundle lacks ") + name);
    return reader.read_section(name);
}
const Tensor& output(const TensorMap& values, const char* name, std::size_t count) {
    const auto found = values.find(name);
    if (found == values.end() || !found->second.data || found->second.dtype != DType::kFloat32 ||
        found->second.numel() != count) throw std::runtime_error(std::string("Invalid LCM output ") + name);
    return found->second;
}
Tensor tensor(std::vector<float>& values, std::initializer_list<int64_t> shape) {
    Tensor result;
    result.data = values.data(); result.shape = shape; result.dtype = DType::kFloat32;
    return result;
}

class Pipeline final : public IImageGeneration {
  public:
    Pipeline(std::unique_ptr<ITrtModule> text, std::unique_ptr<ITrtModule> unet,
             std::unique_ptr<ITrtModule> vae, std::unique_ptr<ITokenizer> tokenizer,
             nlohmann::json config)
        : text_(std::move(text)), unet_(std::move(unet)), vae_(std::move(vae)),
          tokenizer_(std::move(tokenizer)), cfg_(std::move(config)),
          scheduler_(cfg_.at("alphas_cumprod").get<std::vector<float>>(),
              cfg_.at("original_inference_steps"), cfg_.at("timestep_scaling")) {
        bos_ = tokenizer_->id_for_token("<|startoftext|>");
        eos_ = tokenizer_->id_for_token("<|endoftext|>");
        if (bos_ < 0 || eos_ < 0) throw std::runtime_error("LCM tokenizer lacks CLIP special tokens");
    }
    ImageResult generate_image(const std::string& prompt, const ImageGenerationConfig& request) override {
        const int32_t size = cfg_.at("image_size"), side = cfg_.at("latent_size");
        const int32_t channels = cfg_.at("latent_channels"), length = cfg_.at("context_length");
        const int32_t width = cfg_.at("context_width"), guidance_width = cfg_.at("guidance_width");
        if (request.num_samples != 1 || request.cfg_scale >= 0 || request.sde_gamma >= 0 ||
            !request.negative_prompt.empty() || (request.height && request.height != size) ||
            (request.width && request.width != size))
            throw std::invalid_argument("LCM uses one image, learned guidance and the built image size");
        const float guidance = request.guidance_scale < 0 ? 8.0F : request.guidance_scale;
        if (!std::isfinite(guidance)) throw std::invalid_argument("LCM guidance must be finite");
        const auto schedule = scheduler_.timesteps(request.num_steps < 0 ? 4 : request.num_steps);
        auto ids = tokenizer_->encode(prompt);
        if (ids.size() > static_cast<std::size_t>(length - 2))
            throw std::invalid_argument("LCM prompt exceeds the CLIP context length");
        ids.insert(ids.begin(), bos_); ids.push_back(eos_); ids.resize(length, eos_);
        Tensor input;
        input.data = ids.data(); input.shape = {1, length}; input.dtype = DType::kInt32;
        const auto encoded = text_->forward({{"input_ids", input}});
        const auto& hidden = output(encoded, "last_hidden_state", length * width);
        const auto* hidden_values = static_cast<const float*>(hidden.data);
        std::vector<float> context(hidden_values, hidden_values + hidden.numel());
        std::vector<float> guidance_embedding(guidance_width, 0);
        const int32_t half = guidance_width / 2;
        const float exponent = std::log(10000.0F) / (half - 1);
        for (int32_t i = 0; i < half; ++i) {
            const float angle = ((guidance - 1) * 1000) * std::exp(-float(i) * exponent);
            guidance_embedding[i] = std::sin(angle);
            guidance_embedding[half + i] = std::cos(angle);
        }
        const auto count = static_cast<std::size_t>(channels) * side * side;
        std::mt19937 random(request.seed < 0 ? 0 : static_cast<uint32_t>(request.seed));
        std::normal_distribution<float> normal(0, 1);
        auto latents = request.initial_latents;
        if (latents.empty()) {
            latents.resize(count);
            for (auto& value : latents) value = normal(random);
        } else if (latents.size() != count) throw std::invalid_argument("LCM initial latents have the wrong size");
        for (const auto value : latents)
            if (!std::isfinite(value)) throw std::invalid_argument("LCM initial latents must be finite");
        std::vector<float> gaussian(count);
        for (std::size_t i = 0; i < schedule.size(); ++i) {
            std::vector<float> timestep{static_cast<float>(schedule[i])};
            const auto predicted = unet_->forward({{"sample", tensor(latents, {1, channels, side, side})},
                {"timestep", tensor(timestep, {1, 1})},
                {"encoder_hidden_states", tensor(context, {1, length, width})},
                {"timestep_cond", tensor(guidance_embedding, {1, guidance_width})}});
            const auto& epsilon = output(predicted, "out_sample", count);
            const int32_t previous = i + 1 < schedule.size() ? schedule[i + 1] : -1;
            if (previous >= 0) for (auto& value : gaussian) value = normal(random);
            scheduler_.step(static_cast<const float*>(epsilon.data), latents.data(), count,
                schedule[i], previous, previous >= 0 ? gaussian.data() : nullptr);
        }
        const float scaling_factor = cfg_.at("scaling_factor");
        for (auto& value : latents) value /= scaling_factor;
        const auto decoded = vae_->forward({{"latents", tensor(latents, {1, channels, side, side})}});
        const auto plane = static_cast<std::size_t>(size) * size;
        const auto& image = output(decoded, "image", 3 * plane);
        const auto* values = static_cast<const float*>(image.data);
        ImageResult result;
        result.height = size; result.width = size; result.channels = 3; result.num_frames = 1;
        result.pixels.resize(3 * plane);
        for (std::size_t p = 0; p < plane; ++p)
            for (int32_t c = 0; c < 3; ++c) {
                const float value = values[c * plane + p] * 0.5F + 0.5F;
                if (!std::isfinite(value)) throw std::runtime_error("LCM produced a nonfinite pixel");
                result.pixels[p * 3 + c] = std::clamp(value, 0.0F, 1.0F);
            }
        return result;
    }
  private:
    std::unique_ptr<ITrtModule> text_, unet_, vae_;
    std::unique_ptr<ITokenizer> tokenizer_;
    nlohmann::json cfg_;
    Scheduler scheduler_;
    int32_t bos_, eos_;
};
} // namespace
} // namespace trtmc::lcm

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes || context.reader.info().task != "image_generation")
        throw std::invalid_argument("LCM requires image generation without a KV cache");
    const auto raw = lcm::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int32_t side = cfg.at("latent_size"), size = cfg.at("image_size");
    const int32_t channels = cfg.at("latent_channels"), length = cfg.at("context_length");
    const int32_t width = cfg.at("context_width"), guidance = cfg.at("guidance_width");
    const float scaling_factor = cfg.at("scaling_factor");
    if (side <= 0 || size != side * 8 || channels != 4 || length < 2 || width <= 0 ||
        guidance < 4 || !std::isfinite(scaling_factor) || scaling_factor <= 0)
        throw std::runtime_error("Invalid LCM runtime dimensions");
    auto load = [&](const char* name) {
        const auto plan = lcm::section(context.reader, name);
        auto engine = context.backend.create_module(plan.data(), plan.size(), {});
        if (!engine || !engine->ok()) throw std::runtime_error(std::string("LCM failed to load ") + name);
        return engine;
    };
    const auto tokenizer_json = lcm::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    return new lcm::Pipeline(load("text_encoder.plan"), load("unet.plan"), load("vae.plan"),
        std::move(tokenizer), cfg);
}
