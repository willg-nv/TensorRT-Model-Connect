# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLAP's RoBERTa text encoder, CLS pooler, projection MLP and L2 normalization."""

from pathlib import Path

import numpy as np
import tensorrt as trt

from .builders import Graph


def build(request, writer):
    if request.task != "text_to_embedding" or request.backend != "trt":
        raise ValueError("CLAP requires text_to_embedding with backend=trt")
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
        raise NotImplementedError("CLAP supports one fp32 text embedding")
    from transformers import AutoTokenizer, ClapConfig
    from safetensors import safe_open

    directory = Path(request.model_dir)
    config = ClapConfig.from_pretrained(directory)
    cfg = config.text_config.to_dict()
    length = request.max_sequence_length or 128
    if (
        cfg["hidden_act"] != "gelu"
        or cfg["projection_hidden_act"] != "relu"
        or cfg.get("position_embedding_type", "absolute") != "absolute"
    ):
        raise ValueError("Unsupported CLAP text activation or position embeddings")
    if not 2 <= length <= cfg["max_position_embeddings"] - cfg["pad_token_id"] - 1:
        raise ValueError("CLAP text profile exceeds the learned position table")
    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as handle:
        weights = {
            key: handle.get_tensor(key).float().numpy()
            for key in handle.keys()
            if key.startswith(("text_model.", "text_projection."))
        }
    graph = Graph(weights, request.verbose)
    net = graph.net
    width = cfg["hidden_size"]
    ids = net.add_input("input_ids", trt.int32, (length,))
    valid = net.add_input("attention_mask", trt.int32, (length,))
    positions = net.add_input("position_ids", trt.int32, (length,))
    prefix = "text_model.embeddings"
    hidden = net.add_gather(
        graph.const(weights[prefix + ".word_embeddings.weight"]), ids, 0
    ).get_output(0)
    hidden = graph.add(
        hidden, graph.const(weights[prefix + ".token_type_embeddings.weight"][0].reshape(1, width))
    )
    hidden = graph.add(
        hidden,
        net.add_gather(
            graph.const(weights[prefix + ".position_embeddings.weight"]), positions, 0
        ).get_output(0),
    )
    hidden = graph.norm(hidden, prefix + ".LayerNorm", cfg["layer_norm_eps"])
    mask = graph.reshape(net.add_cast(valid, trt.float32).get_output(0), (1, 1, 1, length))
    mask = net.add_elementwise(
        graph.const(np.ones((1, 1, 1, 1))), mask, trt.ElementWiseOperation.SUB
    ).get_output(0)
    mask = graph.mul(mask, graph.const(np.full((1, 1, 1, 1), -1e9)))
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"text_model.encoder.layer.{index}"
        q, k, v = [
            graph.linear(hidden, prefix + ".attention.self." + key)
            for key in ("query", "key", "value")
        ]
        attended = graph.attention(q, k, v, cfg["num_attention_heads"], mask)
        hidden = graph.norm(
            graph.add(hidden, graph.linear(attended, prefix + ".attention.output.dense")),
            prefix + ".attention.output.LayerNorm",
            cfg["layer_norm_eps"],
        )
        inner = graph.linear(hidden, prefix + ".intermediate.dense")
        inner = net.add_activation(inner, trt.ActivationType.GELU_ERF).get_output(0)
        hidden = graph.norm(
            graph.add(hidden, graph.linear(inner, prefix + ".output.dense")),
            prefix + ".output.LayerNorm",
            cfg["layer_norm_eps"],
        )
    pooled = net.add_slice(hidden, (0, 0), (1, width), (1, 1)).get_output(0)
    pooled = graph.linear(pooled, "text_model.pooler.dense")
    pooled = net.add_activation(pooled, trt.ActivationType.TANH).get_output(0)
    projected = graph.linear(pooled, "text_projection.linear1")
    projected = net.add_activation(projected, trt.ActivationType.RELU).get_output(0)
    projected = graph.linear(projected, "text_projection.linear2")
    squared = graph.mul(projected, projected)
    norm = net.add_reduce(squared, trt.ReduceOperation.SUM, 2, True).get_output(0)
    norm = net.add_unary(norm, trt.UnaryOperation.SQRT).get_output(0)
    norm = net.add_elementwise(
        norm, graph.const(np.full((1, 1), 1e-12)), trt.ElementWiseOperation.MAX
    ).get_output(0)
    embedding = net.add_elementwise(projected, norm, trt.ElementWiseOperation.DIV).get_output(0)
    embedding.name = "embedding"
    net.mark_output(embedding)
    writer.set_header(family="clap", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", graph.finish())
    tokenizer = AutoTokenizer.from_pretrained(directory)
    writer.add_bytes("tokenizer.json", tokenizer.backend_tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            hidden_size=cfg["projection_dim"],
            vocab_size=cfg["vocab_size"],
            pad_token_id=cfg["pad_token_id"],
            embedding_space="",
        ),
    )
