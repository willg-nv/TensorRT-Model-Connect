# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an OWL-ViT bundle.

Two engines, because the prompt must stay out of the graph. The image tower
runs once and produces everything that depends only on pixels - the boxes and
the three per-patch class terms. The text tower turns one phrase into one query
embedding. The runtime multiplies them.

Baking the phrases into the graph would have meant rebuilding the engine for
every prompt, which is the mistake Grounding DINO's first implementation made
here; the division of labour above is the fix, and it also means the boxes are
computed once no matter how many phrases are asked for.

"""

from __future__ import annotations

from pathlib import Path

import ml_dtypes  # noqa: F401
import numpy as np
import tensorrt as trt

from . import config as config_module, head_builder, text_builder, vision_builder

_SUPPORTED_TASKS = ("image_text_to_boxes",)


def _load(model_dir: Path):
    import torch
    weights = {key: value.float().numpy() for key,value in torch.load(
        model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True).items()}
    return weights, {key: value.shape for key,value in weights.items()}


def _require_weights(weights: dict) -> None:
    required = (
        "layer_norm.weight",
        "box_head.dense2.weight",
        "class_head.dense0.weight",
        "class_head.logit_shift.weight",
        "class_head.logit_scale.weight",
        "owlvit.text_model.embeddings.token_embedding.weight",
        "owlvit.text_projection.weight",
        "owlvit.vision_model.embeddings.class_embedding",
        "owlvit.vision_model.pre_layernorm.weight",
        "owlvit.vision_model.post_layernorm.weight",
    )
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError("owlvit checkpoint is missing expected tensors: " + ", ".join(missing))


def _tokenizer_json(model_dir: Path) -> bytes:
    """Materialise the checkpoint's tokenizer as a tokenizer.json.

    This checkpoint ships ``vocab.json`` and ``merges.txt`` but no
    ``tokenizer.json``. Rather than hand-assemble one, the reference library
    builds it from those same files, so the bundle carries the exact tokenizer
    the reference uses instead of a reconstruction of it.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("This family requires transformers in the build environment") from error

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if not tokenizer.is_fast:
        raise RuntimeError("owlvit needs the fast tokenizer to materialise tokenizer.json")
    return tokenizer.backend_tokenizer.to_str().encode("utf-8")


def _builder(precision, verbose, quant_ctx=None):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    if precision == "fp32":
        # TF32 caps a conv-and-matmul graph near 0.99998 cosine by default,
        # which hides real graph error behind a plausible-looking number.
        config.clear_flag(trt.BuilderFlag.TF32)
    if quant_ctx is not None:
        # Keep layer names so the plan inspector shows which kernels run.
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    return builder, network, config


def _serialize(builder, network, config, what):
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"owlvit {what} engine build failed")
    return bytes(plan)


def _as_fp32(network, tensor, name):
    out = (tensor if tensor.dtype == trt.float32
           else network.add_cast(tensor, trt.float32).get_output(0))
    out.name = name
    network.mark_output(out)
    return out


def build_vision_engine(weights, cfg, *, precision, verbose=False, quant_ctx=None) -> bytes:
    builder, network, config = _builder(precision, verbose, quant_ctx)
    work_np, work_trt = ((np.float16, trt.float16) if precision == "fp16"
                         else (np.float32, trt.float32))
    vision = cfg["vision"]
    size = vision["image_size"]
    pixels = network.add_input("pixel_values", trt.float32, (1, 3, size, size))
    source = (pixels if pixels.dtype == work_trt
              else network.add_cast(pixels, work_trt).get_output(0))

    merged = vision_builder.build_features(network, source, weights, vision, dtype=work_np,
                                           quant=quant_ctx)
    feats = vision_builder.merge_norm(network, merged, weights, vision, dtype=work_np)

    boxes = head_builder.build_boxes(network, feats, weights, grid=vision["grid"], dtype=work_np,
                                     quant=quant_ctx)
    embeds, shift, scale = head_builder.build_class_terms(network, feats, weights, dtype=work_np,
                                                          quant=quant_ctx)
    _as_fp32(network, boxes, "pred_boxes")
    _as_fp32(network, embeds, "class_embeds")
    _as_fp32(network, shift, "logit_shift")
    _as_fp32(network, scale, "logit_scale")
    return _serialize(builder, network, config, "vision")


def build_text_engine(weights, cfg, *, max_text_len, precision, verbose=False,
                      quant_ctx=None) -> bytes:
    builder, network, config = _builder(precision, verbose, quant_ctx)
    work_np = np.float16 if precision == "fp16" else np.float32
    input_ids = network.add_input("input_ids", trt.int32, (max_text_len,))
    eos_index = network.add_input("eos_index", trt.int32, (1,))
    query = text_builder.build_query(network, input_ids, eos_index, weights, cfg["text"],
                                     dtype=work_np, quant=quant_ctx)
    _as_fp32(network, query, "query_embeds")
    return _serialize(builder, network, config, "text")


def build(request, writer) -> None:
    """Build one OWL-ViT bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("owlvit supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("owlvit supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("owlvit does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("owlvit does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("owlvit does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("owlvit does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("owlvit does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError(
            f"owlvit supports unquantized weights only; got {request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("owlvit does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("owlvit does not support MTP speculative decoding")

    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported owlvit precision: {precision}")

    model_dir = Path(request.model_dir)
    weights, shapes = _load(model_dir)
    _require_weights(weights)
    cfg = config_module.resolve(model_dir, shapes)
    vision = cfg["vision"]

    image_size = int(request.image_height or 0) or vision["image_size"]
    if request.image_width is not None and int(request.image_width) != image_size:
        raise NotImplementedError("owlvit needs a square input size")
    if image_size != vision["image_size"]:
        # One position row per patch of the trained grid, and the box bias is
        # that grid too. A different size needs both rebuilt, not just resized.
        raise NotImplementedError(
            f"owlvit input size is fixed at the checkpoint's {vision['image_size']}")

    max_text_len = cfg["text"]["max_position_embeddings"]
    if request.max_sequence_length is not None:
        if int(request.max_sequence_length) != max_text_len:
            raise NotImplementedError(
                "owlvit phrase length is fixed at the checkpoint's position table")

    quant_ctx = None
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("owlvit requires unquantized weights")

    writer.set_header(family="owlvit", task=request.task, backend=request.backend)
    writer.add_bytes("vision.plan", build_vision_engine(
        weights, cfg, precision=precision, verbose=bool(request.verbose),
        quant_ctx=quant_ctx))
    writer.add_bytes("text.plan", build_text_engine(
        weights, cfg, max_text_len=max_text_len, precision=precision,
        verbose=bool(request.verbose), quant_ctx=quant_ctx))
    writer.add_bytes("tokenizer.json", _tokenizer_json(model_dir))
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": image_size,
            "input_image_w": image_size,
            "image_mean": cfg["image_mean"],
            "image_std": cfg["image_std"],
            "max_text_len": max_text_len,
            "num_patches": vision["num_patches"],
            "grid": vision["grid"],
            "query_dim": cfg["query_dim"],
            # The pad id is 0 here, not the end marker CLIP pads with.
            "pad_token_id": 0,
        },
    )
