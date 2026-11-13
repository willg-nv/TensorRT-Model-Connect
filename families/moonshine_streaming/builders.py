# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Moonshine frame normalization, local encoder attention and speech decoder."""

import numpy as np
import tensorrt as trt

from . import graph as g


def local_mask(length, left, right):
    distance = np.arange(length)[:, None] - np.arange(length)[None, :]
    allowed = ((distance >= 0) & (distance < left)) | ((distance < 0) & (-distance < right))
    return np.where(allowed, 0, -1e9).astype(np.float32).reshape(1, 1, length, length)


class Graph:
    def __init__(self, weights, verbose):
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.net = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        self.options = self.builder.create_builder_config()
        self.options.builder_optimization_level = 3
        self.options.clear_flag(trt.BuilderFlag.TF32)
        self.weights = weights

    def const(self, value, shape=None):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return self.net.add_constant(value.shape if shape is None else shape, value).get_output(0)

    def reshape(self, x, shape):
        layer = self.net.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def op(self, a, b, kind):
        return self.net.add_elementwise(a, b, kind).get_output(0)

    def add(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.PROD)

    def linear(self, x, name):
        return g.add_linear(
            self.net, x, self.weights[name + ".weight"], self.weights.get(name + ".bias")
        )

    def norm(self, x, name, encoder=False):
        gamma = self.weights[name + (".gamma" if encoder else ".weight")]
        if encoder:
            gamma = gamma + np.float32(1)
        return g.add_layer_norm(self.net, x, gamma, np.zeros_like(gamma), 1e-5)

    def silu(self, x):
        return self.mul(x, self.net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0))

    def mask(self, valid):
        return self.mul(
            self.op(self.const(np.ones((1, 1, 1, 1))), valid, trt.ElementWiseOperation.SUB),
            self.const(np.full((1, 1, 1, 1), -1e9)),
        )

    def heads(self, x, length, heads, dim):
        layer = self.net.add_shuffle(x)
        layer.reshape_dims = (1, length, heads, dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    def attention(self, query, key, value, query_length, heads, dim, mask):
        scores = self.net.add_matrix_multiply(
            query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = self.add(self.mul(scores, self.const(np.full((1, 1, 1, 1), dim**-0.5))), mask)
        softmax = self.net.add_softmax(scores)
        softmax.axes = 8
        context = self.net.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)
        layer = self.net.add_shuffle(context)
        layer.first_transpose = (0, 2, 1, 3)
        layer.reshape_dims = (query_length, heads * dim)
        return layer.get_output(0)

    def finish(self):
        plan = self.builder.build_serialized_network(self.net, self.options)
        if plan is None:
            raise RuntimeError("Moonshine Streaming engine build failed")
        return bytes(plan)


def build_encoder(weights, cfg, samples, verbose=False):
    graph = Graph(weights, verbose)
    net = graph.net
    enc = cfg["encoder_config"]
    width = enc["hidden_size"]
    heads = enc["num_attention_heads"]
    dim = enc["head_dim"]
    frame = round(enc["sample_rate"] * enc["frame_ms"] / 1000)
    frames = samples // frame
    audio = net.add_input("input_values", trt.float32, (samples,))
    valid = net.add_input("frame_mask", trt.float32, (frames,))
    hidden = graph.reshape(audio, (frames, frame))
    hidden = g.add_layer_norm(
        net, hidden, np.ones(frame, np.float32), np.zeros(frame, np.float32), 1e-6
    )
    scale = np.exp(weights["model.encoder.embedder.comp.log_k"])
    hidden = net.add_unary(
        graph.mul(hidden, graph.const(scale, (1, 1))), trt.UnaryOperation.ASINH
    ).get_output(0)
    hidden = graph.silu(graph.linear(hidden, "model.encoder.embedder.linear"))
    hidden = graph.mul(hidden, graph.reshape(valid, (frames, 1)))
    transposed = net.add_shuffle(hidden)
    transposed.first_transpose = (1, 0)
    transposed.reshape_dims = (1, width, 1, frames)
    hidden = transposed.get_output(0)
    valid = graph.reshape(valid, (1, 1, 1, frames))
    for index in (1, 2):
        name = f"model.encoder.embedder.conv{index}"
        weight = weights[name + ".weight"]
        output_width = weight.shape[0]
        conv = net.add_convolution_nd(
            hidden,
            output_width,
            (1, 5),
            np.ascontiguousarray(weight.reshape(output_width, weight.shape[1], 1, 5)),
            np.ascontiguousarray(weights[name + ".bias"]),
        )
        conv.stride_nd = (1, 2)
        conv.pre_padding = (0, 4)
        conv.post_padding = (0, 0)
        pool = net.add_pooling_nd(valid, trt.PoolingType.MAX, (1, 5))
        pool.stride_nd = (1, 2)
        pool.pre_padding = (0, 4)
        pool.post_padding = (0, 0)
        valid = pool.get_output(0)
        hidden = graph.mul(conv.get_output(0), valid)
        if index == 1:
            hidden = graph.silu(hidden)
    encoded_frames = (frames + 3) // 4
    flat = net.add_shuffle(hidden)
    flat.reshape_dims = (width, encoded_frames)
    flat.second_transpose = (1, 0)
    hidden = flat.get_output(0)
    mask = graph.mask(graph.reshape(valid, (1, 1, 1, encoded_frames)))
    for index in range(enc["num_hidden_layers"]):
        prefix = f"model.encoder.layers.{index}"
        normalized = graph.norm(hidden, prefix + ".input_layernorm", encoder=True)
        query, key, value = [
            graph.heads(
                graph.linear(normalized, prefix + ".self_attn." + name), encoded_frames, heads, dim
            )
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        left, right = enc["sliding_windows"][index]
        attended = graph.attention(
            query,
            key,
            value,
            encoded_frames,
            heads,
            dim,
            graph.add(mask, graph.const(local_mask(encoded_frames, left, right))),
        )
        hidden = graph.add(hidden, graph.linear(attended, prefix + ".self_attn.o_proj"))
        inner = graph.linear(
            graph.norm(hidden, prefix + ".post_attention_layernorm", encoder=True),
            prefix + ".mlp.fc1",
        )
        inner = net.add_activation(inner, trt.ActivationType.GELU_ERF).get_output(0)
        hidden = graph.add(hidden, graph.linear(inner, prefix + ".mlp.fc2"))
    hidden = graph.norm(hidden, "model.encoder.final_norm", encoder=True)
    hidden = graph.add(
        hidden, graph.const(weights["model.decoder.pos_emb.weight"][:encoded_frames])
    )
    if "model.decoder.proj.weight" in weights:
        hidden = graph.linear(hidden, "model.decoder.proj")
    hidden.name = "encoder_hidden"
    net.mark_output(hidden)
    valid = graph.reshape(valid, (encoded_frames,))
    valid.name = "encoder_mask"
    net.mark_output(valid)
    return graph.finish()


def build_decoder(weights, cfg, samples, length, verbose=False):
    import torch
    from transformers import MoonshineStreamingConfig
    from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
        MoonshineStreamingRotaryEmbedding,
    )

    graph = Graph(weights, verbose)
    net = graph.net
    heads = cfg["num_attention_heads"]
    dim = cfg["head_dim"]
    width = cfg["hidden_size"]
    frame = round(cfg["encoder_config"]["sample_rate"] * cfg["encoder_config"]["frame_ms"] / 1000)
    source_length = (samples // frame + 3) // 4
    ids = net.add_input("input_ids", trt.int32, (length,))
    valid = net.add_input("attention_mask", trt.int32, (length,))
    memory = net.add_input("encoder_hidden", trt.float32, (source_length, width))
    source_valid = net.add_input("encoder_mask", trt.float32, (source_length,))
    last = net.add_input("last_index", trt.int32, (1,))
    mask = graph.mask(
        graph.reshape(net.add_cast(valid, trt.float32).get_output(0), (1, 1, 1, length))
    )
    mask = graph.add(
        mask, graph.const(np.triu(np.full((1, 1, length, length), -1e9, np.float32), k=1))
    )
    source_mask = graph.mask(graph.reshape(source_valid, (1, 1, 1, source_length)))
    rotary = MoonshineStreamingRotaryEmbedding(MoonshineStreamingConfig(**cfg))
    with torch.no_grad():
        cosine, sine = rotary(torch.empty(1, length, width), torch.arange(length)[None, :])
    rotary_dim = cosine.shape[-1]
    cosine, sine = [
        graph.const(
            np.repeat(x.numpy()[0, :, : rotary_dim // 2], 2, axis=-1).reshape(
                1, 1, length, rotary_dim
            )
        )
        for x in (cosine, sine)
    ]

    def rope(x):
        even = net.add_slice(
            x, (0, 0, 0, 0), (1, heads, length, rotary_dim // 2), (1, 1, 1, 2)
        ).get_output(0)
        odd = net.add_slice(
            x, (0, 0, 0, 1), (1, heads, length, rotary_dim // 2), (1, 1, 1, 2)
        ).get_output(0)
        negative = net.add_unary(odd, trt.UnaryOperation.NEG).get_output(0)
        joined = net.add_concatenation(
            [
                graph.reshape(negative, (1, heads, length, rotary_dim // 2, 1)),
                graph.reshape(even, (1, heads, length, rotary_dim // 2, 1)),
            ]
        )
        joined.axis = 4
        rotated = graph.reshape(joined.get_output(0), (1, heads, length, rotary_dim))
        original = net.add_slice(
            x, (0, 0, 0, 0), (1, heads, length, rotary_dim), (1, 1, 1, 1)
        ).get_output(0)
        rotated = graph.add(graph.mul(original, cosine), graph.mul(rotated, sine))
        if rotary_dim < dim:
            passed = net.add_slice(
                x, (0, 0, 0, rotary_dim), (1, heads, length, dim - rotary_dim), (1, 1, 1, 1)
            ).get_output(0)
            joined = net.add_concatenation([rotated, passed])
            joined.axis = 3
            return joined.get_output(0)
        return rotated

    hidden = net.add_gather(
        graph.const(weights["model.decoder.embed_tokens.weight"]), ids, 0
    ).get_output(0)
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"model.decoder.layers.{index}"
        normalized = graph.norm(hidden, prefix + ".input_layernorm")
        query, key, value = [
            graph.heads(graph.linear(normalized, prefix + ".self_attn." + name), length, heads, dim)
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        attended = graph.attention(rope(query), rope(key), value, length, heads, dim, mask)
        hidden = graph.add(hidden, graph.linear(attended, prefix + ".self_attn.o_proj"))
        normalized = graph.norm(hidden, prefix + ".post_attention_layernorm")
        query = graph.heads(
            graph.linear(normalized, prefix + ".encoder_attn.q_proj"), length, heads, dim
        )
        key = graph.heads(
            graph.linear(memory, prefix + ".encoder_attn.k_proj"), source_length, heads, dim
        )
        value = graph.heads(
            graph.linear(memory, prefix + ".encoder_attn.v_proj"), source_length, heads, dim
        )
        attended = graph.attention(query, key, value, length, heads, dim, source_mask)
        hidden = graph.add(hidden, graph.linear(attended, prefix + ".encoder_attn.o_proj"))
        inner = graph.linear(graph.norm(hidden, prefix + ".final_layernorm"), prefix + ".mlp.fc1")
        middle = cfg["intermediate_size"]
        state = net.add_slice(inner, (0, 0), (length, middle), (1, 1)).get_output(0)
        gate = net.add_slice(inner, (0, middle), (length, middle), (1, 1)).get_output(0)
        hidden = graph.add(
            hidden, graph.linear(graph.mul(state, graph.silu(gate)), prefix + ".mlp.fc2")
        )
    hidden = graph.norm(hidden, "model.decoder.norm")
    selected = net.add_gather(hidden, last, 0).get_output(0)
    logits = graph.linear(selected, "proj_out")
    logits.name = "logits"
    net.mark_output(logits)
    return graph.finish()
