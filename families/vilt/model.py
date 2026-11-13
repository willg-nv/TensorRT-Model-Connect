# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ViLT VQA with masked full-image patches and a learned answer classifier."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt

CANVAS = 640
PATCH = 32


def build_engine(weights, cfg, length, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    width, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    dim = width // heads
    patches = (CANVAS // PATCH) ** 2
    count = length + patches + 1

    def const(value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return net.add_constant(value.shape, value).get_output(0)

    def op(a, b, kind):
        return net.add_elementwise(a, b, kind).get_output(0)

    def add(a, b):
        return op(a, b, trt.ElementWiseOperation.SUM)

    def linear(x, key):
        output = net.add_matrix_multiply(
            x,
            trt.MatrixOperation.NONE,
            const(weights[key + ".weight"]),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        return add(output, const(weights[key + ".bias"].reshape(1, -1)))

    def norm(x, key, eps=None):
        layer = net.add_normalization(
            x,
            const(weights[key + ".weight"].reshape(1, -1)),
            const(weights[key + ".bias"].reshape(1, -1)),
            2,
        )
        layer.epsilon = cfg["layer_norm_eps"] if eps is None else eps
        return layer.get_output(0)

    def concat(tensors, axis):
        layer = net.add_concatenation(tensors)
        layer.axis = axis
        return layer.get_output(0)

    ids = net.add_input("input_ids", trt.int32, (length,))
    valid = net.add_input("attention_mask", trt.int32, (length,))
    pixels = net.add_input("pixel_values", trt.float32, (1, 3, CANVAS, CANVAS))
    image_positions = net.add_input("image_positions", trt.float32, (patches + 1, width))
    image_valid = net.add_input("image_mask", trt.int32, (patches + 1,))
    prefix = "vilt.embeddings"
    text = net.add_gather(
        const(weights[prefix + ".text_embeddings.word_embeddings.weight"]), ids, 0
    ).get_output(0)
    text = add(text, const(weights[prefix + ".text_embeddings.token_type_embeddings.weight"][0:1]))
    text = add(
        text, const(weights[prefix + ".text_embeddings.position_embeddings.weight"][:length])
    )
    text = norm(text, prefix + ".text_embeddings.LayerNorm")
    text = add(text, const(weights[prefix + ".token_type_embeddings.weight"][0:1]))
    projection = prefix + ".patch_embeddings.projection"
    conv = net.add_convolution_nd(
        pixels,
        width,
        (PATCH, PATCH),
        np.ascontiguousarray(weights[projection + ".weight"]),
        np.ascontiguousarray(weights[projection + ".bias"]),
    )
    conv.stride_nd = (PATCH, PATCH)
    layer = net.add_shuffle(conv.get_output(0))
    layer.reshape_dims = (width, patches)
    layer.second_transpose = (1, 0)
    image = concat(
        [const(weights[prefix + ".cls_token"].reshape(1, width)), layer.get_output(0)], 0
    )
    image = add(
        add(image, image_positions), const(weights[prefix + ".token_type_embeddings.weight"][1:2])
    )
    hidden = concat([text, image], 0)
    valid = net.add_cast(concat([valid, image_valid], 0), trt.float32).get_output(0)
    mask = op(const(np.ones(1)), valid, trt.ElementWiseOperation.SUB)
    mask = op(mask, const(np.array([-1e9])), trt.ElementWiseOperation.PROD)
    layer = net.add_shuffle(mask)
    layer.reshape_dims = (1, 1, 1, count)
    mask = layer.get_output(0)

    def split(x):
        layer = net.add_shuffle(x)
        layer.reshape_dims = (1, count, heads, dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    for index in range(cfg["num_hidden_layers"]):
        prefix = f"vilt.encoder.layer.{index}"
        normalized = norm(hidden, prefix + ".layernorm_before")
        q, k, v = [
            split(linear(normalized, prefix + ".attention.attention." + name))
            for name in ("query", "key", "value")
        ]
        scores = net.add_matrix_multiply(
            q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = add(
            op(scores, const(np.array([[[[dim**-0.5]]]])), trt.ElementWiseOperation.PROD), mask
        )
        softmax = net.add_softmax(scores)
        softmax.axes = 8
        context = net.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE
        ).get_output(0)
        layer = net.add_shuffle(context)
        layer.first_transpose = (0, 2, 1, 3)
        layer.reshape_dims = (count, width)
        hidden = add(hidden, linear(layer.get_output(0), prefix + ".attention.output.dense"))
        mlp = linear(norm(hidden, prefix + ".layernorm_after"), prefix + ".intermediate.dense")
        mlp = net.add_activation(mlp, trt.ActivationType.GELU_ERF).get_output(0)
        hidden = add(hidden, linear(mlp, prefix + ".output.dense"))
    hidden = norm(hidden, "vilt.layernorm")
    first = net.add_slice(hidden, (0, 0), (1, width), (1, 1)).get_output(0)
    pooled = net.add_activation(
        linear(first, "vilt.pooler.dense"), trt.ActivationType.TANH
    ).get_output(0)
    answer = norm(linear(pooled, "classifier.0"), "classifier.1", 1e-5)
    answer = net.add_activation(answer, trt.ActivationType.GELU_ERF).get_output(0)
    logits = linear(answer, "classifier.3")
    logits.name = "logits"
    net.mark_output(logits)
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("ViLT VQA engine build failed")
    return bytes(plan)


def build(request, writer):
    if request.task != "images_text_to_text" or request.backend != "trt":
        raise ValueError("ViLT VQA requires images_text_to_text with backend=trt")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.dynamic_kv_cache
        or request.image_height is not None
        or request.image_width is not None
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError(
            "ViLT VQA supports one fp32 image/question with checkpoint resizing"
        )
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    length = request.max_sequence_length or cfg["max_position_embeddings"]
    if not 2 <= length <= cfg["max_position_embeddings"]:
        raise ValueError("ViLT question profile exceeds the checkpoint position table")
    if cfg["patch_size"] != PATCH or cfg.get("max_image_length", -1) != -1:
        raise ValueError("ViLT VQA requires all image patches with patch_size=32")
    import torch

    weights = {
        key: value.float().numpy()
        for key, value in torch.load(
            directory / "pytorch_model.bin", map_location="cpu", weights_only=True
        ).items()
    }
    labels = [cfg["id2label"][str(index)] for index in range(len(cfg["id2label"]))]
    writer.set_header(family="vilt", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(weights, cfg, length, request.verbose))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    writer.add_bytes("tokenizer.json", tokenizer.backend_tokenizer.to_str().encode())
    writer.add_bytes(
        "image_positions.f32",
        weights["vilt.embeddings.position_embeddings"].astype("<f4").tobytes(),
    )
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            hidden_size=cfg["hidden_size"],
            image_size=cfg["image_size"],
            labels=labels,
        ),
    )
