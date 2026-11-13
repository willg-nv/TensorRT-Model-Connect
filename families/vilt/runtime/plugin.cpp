/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vilt/runtime/tokenizer.h"
#include "families/vilt/runtime/image_preprocess_seam.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::vilt {
namespace {
constexpr int32_t kCanvas = 640, kPatch = 32, kGrid = kCanvas / kPatch;
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0)
        throw std::runtime_error(std::string("ViLT bundle is missing ") + name);
    return reader.read_section(name);
}

class Pipeline final : public internal::IModel, public internal::IImagesTextToText {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, std::unique_ptr<ITokenizer> tokenizer,
             int32_t length, int32_t width, int32_t original_grid, std::vector<float> positions,
             std::vector<std::string> labels)
        : engine_(std::move(engine)), tokenizer_(std::move(tokenizer)), length_(length),
          width_(width), original_grid_(original_grid), positions_(std::move(positions)),
          labels_(std::move(labels)) {}
    const char* task() const noexcept override { return "images_text_to_text"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IImagesTextToText>(*this)};
    }
    internal::TextResult run(const internal::ImagesTextToTextRequest& request,
                            internal::ConfigView config) override {
        if (!config.empty()) throw internal::ConfigError("ViLT VQA has no generation config");
        if (request.messages.size() != 1 || !request.tools.empty() ||
            request.messages[0].role != internal::MessageRole::User)
            throw std::invalid_argument("ViLT VQA requires one user image/question without tools");
        const internal::ImageView* image = nullptr;
        const internal::TextPartView* question = nullptr;
        for (const auto& part : request.messages[0].parts) {
            if (const auto* value = std::get_if<internal::ImageView>(&part)) {
                if (image) throw std::invalid_argument("ViLT VQA requires exactly one image");
                image = value;
            } else if (const auto* value = std::get_if<internal::TextPartView>(&part)) {
                if (question) throw std::invalid_argument("ViLT VQA requires exactly one question");
                question = value;
            } else throw std::invalid_argument("Unsupported ViLT VQA message part");
        }
        if (!image || !question || question->text.empty())
            throw std::invalid_argument("ViLT VQA requires an image and a nonempty question");
        if (!image->data || image->format != internal::ImageFormat::Float32 || image->channels != 3 ||
            image->height == 0 || image->width == 0 ||
            image->height > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
            image->width > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
            static_cast<uint64_t>(image->height) > std::numeric_limits<std::size_t>::max() / image->width / 3 / sizeof(float) ||
            image->byte_size != static_cast<std::size_t>(image->height) * image->width * 3 * sizeof(float))
            throw std::invalid_argument("ViLT requires contiguous float32 RGB input");
        auto ids = tokenizer_->encode(std::string(question->text));
        if (ids.empty() || ids.size() > static_cast<std::size_t>(length_))
            throw std::invalid_argument("ViLT question exceeds the built text profile");
        std::vector<int32_t> valid(length_, 0);
        std::fill(valid.begin(), valid.begin() + ids.size(), 1);
        ids.resize(length_, 0);

        // Checkpoint resize: shortest edge 384, longest edge at most 639,
        // rounded to the nearest pixel and then down to a multiple of 32.
        double target_h = image->height, target_w = image->width;
        double scale = 384.0 / std::min(target_h, target_w);
        target_h *= scale; target_w *= scale;
        if (std::max(target_h, target_w) > 639) {
            scale = 639.0 / std::max(target_h, target_w);
            target_h *= scale; target_w *= scale;
        }
        const int32_t height = static_cast<int32_t>(target_h + 0.5) / kPatch * kPatch;
        const int32_t width = static_cast<int32_t>(target_w + 0.5) / kPatch * kPatch;
        if (height <= 0 || width <= 0 || height > kCanvas || width > kCanvas)
            throw std::invalid_argument("ViLT image aspect ratio produces an empty patch axis");
        ViltPreprocessConfig preprocessing;
        preprocessing.input_image_h = height; preprocessing.input_image_w = width;
        preprocessing.image_mean = {0.5F, 0.5F, 0.5F}; preprocessing.image_std = {0.5F, 0.5F, 0.5F};
        const auto resized = preprocess_vilt_image(static_cast<const float*>(image->data),
            image->height, image->width, preprocessing);
        std::vector<float> pixels(3 * kCanvas * kCanvas, 0);
        for (int32_t c = 0; c < 3; ++c)
            for (int32_t y = 0; y < height; ++y)
                std::copy_n(resized.data() + (c * height + y) * width, width,
                    pixels.data() + (c * kCanvas + y) * kCanvas);
        const int32_t grid_h = height / kPatch, grid_w = width / kPatch;
        std::vector<float> positions((1 + kGrid * kGrid) * width_, 0);
        std::copy_n(positions_.data(), width_, positions.data());
        std::vector<int32_t> image_mask(1 + kGrid * kGrid, 0);
        image_mask[0] = 1;
        // Bilinear position interpolation with align_corners=true. Padding
        // positions are zero and remain excluded from every attention layer.
        for (int32_t y = 0; y < grid_h; ++y) {
            const float sy = grid_h == 1 ? 0 : float(y) * (original_grid_ - 1) / (grid_h - 1);
            const int32_t y0 = static_cast<int32_t>(sy), y1 = std::min(y0 + 1, original_grid_ - 1);
            for (int32_t x = 0; x < grid_w; ++x) {
                const float sx = grid_w == 1 ? 0 : float(x) * (original_grid_ - 1) / (grid_w - 1);
                const int32_t x0 = static_cast<int32_t>(sx), x1 = std::min(x0 + 1, original_grid_ - 1);
                const auto target = (1 + y * kGrid + x) * width_;
                image_mask[1 + y * kGrid + x] = 1;
                for (int32_t c = 0; c < width_; ++c) {
                    const float a = positions_[(1 + y0 * original_grid_ + x0) * width_ + c];
                    const float b = positions_[(1 + y0 * original_grid_ + x1) * width_ + c];
                    const float d = positions_[(1 + y1 * original_grid_ + x0) * width_ + c];
                    const float e = positions_[(1 + y1 * original_grid_ + x1) * width_ + c];
                    positions[target + c] = ((1 - (sx - x0)) * a + (sx - x0) * b) * (1 - (sy - y0)) +
                        ((1 - (sx - x0)) * d + (sx - x0) * e) * (sy - y0);
                }
            }
        }
        Tensor input, mask, rgb, position, patches;
        input.data = ids.data(); input.shape = {length_}; input.dtype = DType::kInt32;
        mask.data = valid.data(); mask.shape = {length_}; mask.dtype = DType::kInt32;
        rgb.data = pixels.data(); rgb.shape = {1, 3, kCanvas, kCanvas}; rgb.dtype = DType::kFloat32;
        position.data = positions.data(); position.shape = {1 + kGrid * kGrid, width_}; position.dtype = DType::kFloat32;
        patches.data = image_mask.data(); patches.shape = {1 + kGrid * kGrid}; patches.dtype = DType::kInt32;
        auto outputs = engine_->forward({{"input_ids", input}, {"attention_mask", mask},
            {"pixel_values", rgb}, {"image_positions", position}, {"image_mask", patches}});
        const auto found = outputs.find("logits");
        if (found == outputs.end() || !found->second.data || found->second.dtype != DType::kFloat32 ||
            found->second.numel() != labels_.size())
            throw std::runtime_error("ViLT engine returned invalid answer scores");
        const auto* values = static_cast<const float*>(found->second.data);
        for (std::size_t i = 0; i < labels_.size(); ++i)
            if (!std::isfinite(values[i])) throw std::runtime_error("ViLT answer score is nonfinite");
        const auto answer = std::max_element(values, values + labels_.size()) - values;
        // The head selects a learned answer label; it does not emit decoder tokens.
        return {labels_[answer], {}};
    }
  private:
    std::unique_ptr<ITrtModule> engine_;
    std::unique_ptr<ITokenizer> tokenizer_;
    int32_t length_, width_, original_grid_;
    std::vector<float> positions_;
    std::vector<std::string> labels_;
};
} // namespace
} // namespace trtmc::vilt

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes != 0 || context.reader.info().task != "images_text_to_text")
        throw std::invalid_argument("ViLT requires image/question answering without a KV cache");
    const auto data = vilt::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(data.begin(), data.end());
    const int32_t length = cfg.at("sequence_length"), width = cfg.at("hidden_size");
    const int32_t image_size = cfg.at("image_size"), grid = image_size / vilt::kPatch;
    auto labels = cfg.at("labels").get<std::vector<std::string>>();
    if (length <= 0 || length > 40 || width <= 0 || image_size <= 0 || grid > 20 ||
        image_size % vilt::kPatch != 0 || labels.empty())
        throw std::runtime_error("Invalid ViLT runtime dimensions");
    const auto raw_positions = vilt::section(context.reader, "image_positions.f32");
    if (raw_positions.size() != static_cast<std::size_t>(1 + grid * grid) * width * sizeof(float))
        throw std::runtime_error("Invalid ViLT image position table");
    std::vector<float> positions(raw_positions.size() / sizeof(float));
    std::memcpy(positions.data(), raw_positions.data(), raw_positions.size());
    const auto plan = vilt::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok()) throw std::runtime_error("ViLT engine failed to load");
    const auto tokenizer_json = vilt::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateWordPieceTokenizer(tokenizer_json.data(), tokenizer_json.size(), true);
    return new vilt::Pipeline(std::move(engine), std::move(tokenizer), length, width, grid,
        std::move(positions), std::move(labels));
}
