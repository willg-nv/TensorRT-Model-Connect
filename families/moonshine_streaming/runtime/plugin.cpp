/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/moonshine_streaming/runtime/tokenizer.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::moonshine_streaming {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0)
        throw std::runtime_error(std::string("Moonshine bundle lacks ") + name);
    return reader.read_section(name);
}
Tensor tensor(void* data, std::initializer_list<int64_t> shape, DType dtype) {
    Tensor result;
    result.data = data; result.shape = shape; result.dtype = dtype;
    return result;
}
const Tensor& output(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || !found->second.data || found->second.dtype != DType::kFloat32 ||
        found->second.numel() != count)
        throw std::runtime_error(std::string("Invalid Moonshine output ") + name);
    return found->second;
}

class Pipeline final : public internal::IModel, public internal::ISpeechTranscription {
  public:
    Pipeline(std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> decoder,
             std::unique_ptr<ITokenizer> tokenizer, nlohmann::json cfg)
        : encoder_(std::move(encoder)), decoder_(std::move(decoder)), tokenizer_(std::move(tokenizer)),
          cfg_(std::move(cfg)), fields_{
            {"max_output_tokens", internal::ConfigKind::I64, internal::ConfigValue{int64_t{128}},
             "Maximum generated tokens; greedy decoding"}} {}
    const char* task() const noexcept override { return "speech_transcription"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ISpeechTranscription>(*this, {fields_.data(), fields_.size()})};
    }
    internal::TextResult run(const internal::SpeechTranscriptionRequest& request,
                            internal::ConfigView config) override {
        const Span<const internal::ConfigField> fields{fields_.data(), fields_.size()};
        internal::validate_config(fields, config);
        const auto max_new = *internal::config_get<int64_t>(config, fields, "max_output_tokens");
        const int32_t length = cfg_.at("decoder_length"), vocab = cfg_.at("vocab_size");
        const int32_t samples = cfg_.at("max_samples"), frame = cfg_.at("frame_length");
        const int32_t frames = cfg_.at("encoded_frames"), width = cfg_.at("source_width");
        const auto& audio = request.audio;
        if (audio.channels != 1 || audio.sample_rate.value_or(cfg_.at("sample_rate").get<uint32_t>()) !=
            cfg_.at("sample_rate").get<uint32_t>() || !audio.samples.data() ||
            audio.samples.size() < static_cast<std::size_t>(frame) ||
            audio.samples.size() > static_cast<std::size_t>(samples))
            throw std::invalid_argument("Moonshine requires mono 16 kHz PCM within its built audio profile");
        if (request.source_language && *request.source_language != cfg_.at("source_language").get<std::string>())
            throw std::invalid_argument("This Moonshine checkpoint supports English transcription");
        if (max_new <= 0 || max_new >= length)
            throw std::invalid_argument("Moonshine generation exceeds its built decoder profile");
        for (const auto sample : audio.samples)
            if (!std::isfinite(sample)) throw std::invalid_argument("Moonshine audio contains a nonfinite sample");
        std::vector<float> waveform(samples, 0), valid_frames(samples / frame, 0);
        std::copy(audio.samples.begin(), audio.samples.end(), waveform.begin());
        std::fill_n(valid_frames.begin(), audio.samples.size() / frame, 1);
        const auto encoded = encoder_->forward({
            {"input_values", tensor(waveform.data(), {samples}, DType::kFloat32)},
            {"frame_mask", tensor(valid_frames.data(), {samples / frame}, DType::kFloat32)}});
        const auto& memory = output(encoded, "encoder_hidden", static_cast<std::size_t>(frames) * width);
        const auto& mask = output(encoded, "encoder_mask", frames);
        const auto* memory_data = static_cast<const float*>(memory.data);
        const auto* mask_data = static_cast<const float*>(mask.data);
        std::vector<float> source(memory_data, memory_data + memory.numel());
        std::vector<float> source_mask(mask_data, mask_data + mask.numel());
        std::vector<int32_t> ids(length, cfg_.at("pad_id").get<int32_t>()), valid(length, 0);
        ids[0] = cfg_.at("bos_id"); valid[0] = 1;
        const int32_t eos = cfg_.at("eos_id");
        internal::TextResult result;
        for (int32_t step = 0; step < max_new; ++step) {
            int32_t last = step;
            const auto decoded = decoder_->forward({
                {"input_ids", tensor(ids.data(), {length}, DType::kInt32)},
                {"attention_mask", tensor(valid.data(), {length}, DType::kInt32)},
                {"encoder_hidden", tensor(source.data(), {frames, width}, DType::kFloat32)},
                {"encoder_mask", tensor(source_mask.data(), {frames}, DType::kFloat32)},
                {"last_index", tensor(&last, {1}, DType::kInt32)}});
            const auto& logits = output(decoded, "logits", vocab);
            const auto* values = static_cast<const float*>(logits.data);
            for (int32_t j = 0; j < vocab; ++j)
                if (!std::isfinite(values[j])) throw std::runtime_error("Moonshine logit is nonfinite");
            const auto next = static_cast<int32_t>(std::max_element(values, values + vocab) - values);
            result.token_ids.push_back(next);
            if (next == eos) break;
            ids[step + 1] = next; valid[step + 1] = 1;
        }
        result.text = tokenizer_->decode(result.token_ids);
        return result;
    }
  private:
    std::unique_ptr<ITrtModule> encoder_, decoder_;
    std::unique_ptr<ITokenizer> tokenizer_;
    nlohmann::json cfg_;
    std::vector<internal::ConfigField> fields_;
};
} // namespace
} // namespace trtmc::moonshine_streaming

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes || context.reader.info().task != "speech_transcription")
        throw std::invalid_argument("Moonshine requires speech transcription without an external KV cache");
    const auto raw = moonshine_streaming::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int32_t length = cfg.at("decoder_length"), width = cfg.at("source_width");
    const int32_t vocab = cfg.at("vocab_size"), samples = cfg.at("max_samples"), frame = cfg.at("frame_length");
    const int32_t frames = cfg.at("encoded_frames"), bos = cfg.at("bos_id"), eos = cfg.at("eos_id"), pad = cfg.at("pad_id");
    if (length < 2 || width <= 0 || vocab <= 0 || samples <= 0 || frame <= 0 || samples % frame ||
        frames != (samples / frame + 3) / 4 || bos < 0 || bos >= vocab || eos < 0 || eos >= vocab ||
        pad < 0 || pad >= vocab || cfg.at("sample_rate") != 16000 || cfg.at("source_language") != "en")
        throw std::runtime_error("Invalid Moonshine runtime dimensions");
    auto load = [&](const char* name) {
        const auto plan = moonshine_streaming::section(context.reader, name);
        auto engine = context.backend.create_module(plan.data(), plan.size(), {});
        if (!engine || !engine->ok()) throw std::runtime_error(std::string("Moonshine failed to load ") + name);
        return engine;
    };
    const auto raw_tokenizer = moonshine_streaming::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(raw_tokenizer.data(), raw_tokenizer.size(), false);
    return new moonshine_streaming::Pipeline(load("encoder.plan"), load("decoder.plan"), std::move(tokenizer), cfg);
}
