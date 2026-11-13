# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-profile offline transcription with the Moonshine Streaming checkpoint."""

import json
from pathlib import Path

from .builders import build_decoder, build_encoder

MAX_SAMPLES = 384000


def build(request, writer):
    if request.task != "speech_transcription" or request.backend != "trt":
        raise ValueError("Moonshine Streaming requires speech_transcription with backend=trt")
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
        raise NotImplementedError("Moonshine Streaming supports one fp32 mono transcription")
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    encoder = cfg["encoder_config"]
    preprocessing = json.loads((directory / "preprocessor_config.json").read_text())
    if (
        cfg["model_type"] != "moonshine_streaming"
        or cfg["hidden_act"] != "silu"
        or encoder["hidden_act"] != "gelu"
        or cfg["attention_bias"]
        or encoder["attention_bias"]
        or cfg["num_attention_heads"] != cfg["num_key_value_heads"]
        or encoder["num_attention_heads"] != encoder["num_key_value_heads"]
        or encoder["sample_rate"] != 16000
        or preprocessing["do_normalize"]
        or preprocessing["padding_value"] != 0
    ):
        raise ValueError(
            "Unsupported Moonshine Streaming encoder/decoder or waveform preprocessing"
        )
    frame = round(encoder["sample_rate"] * encoder["frame_ms"] / 1000)
    if MAX_SAMPLES % frame or preprocessing["pad_to_multiple_of"] != frame:
        raise ValueError("Moonshine Streaming audio profile must contain complete waveform frames")
    encoded_frames = (MAX_SAMPLES // frame + 3) // 4
    length = request.max_sequence_length or 256
    if (
        not 2 <= length <= cfg["max_position_embeddings"]
        or encoded_frames > encoder["max_position_embeddings"]
    ):
        raise ValueError("Moonshine Streaming profile exceeds learned positions")
    from safetensors import safe_open

    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as handle:
        weights = {key: handle.get_tensor(key).float().numpy() for key in handle.keys()}
    if "proj_out.weight" not in weights and cfg["tie_word_embeddings"]:
        weights["proj_out.weight"] = weights["model.decoder.embed_tokens.weight"]
    writer.set_header(family="moonshine_streaming", task=request.task, backend=request.backend)
    writer.add_bytes("encoder.plan", build_encoder(weights, cfg, MAX_SAMPLES, request.verbose))
    writer.add_bytes(
        "decoder.plan", build_decoder(weights, cfg, MAX_SAMPLES, length, request.verbose)
    )
    writer.add_bytes("tokenizer.json", (directory / "tokenizer.json").read_bytes())
    generation = json.loads((directory / "generation_config.json").read_text())
    writer.add_json(
        "runtime.json",
        dict(
            max_samples=MAX_SAMPLES,
            frame_length=frame,
            sample_rate=16000,
            encoded_frames=encoded_frames,
            source_width=cfg["hidden_size"],
            decoder_length=length,
            vocab_size=cfg["vocab_size"],
            bos_id=generation["decoder_start_token_id"],
            eos_id=generation["eos_token_id"],
            pad_id=generation["pad_token_id"],
            source_language="en",
        ),
    )
