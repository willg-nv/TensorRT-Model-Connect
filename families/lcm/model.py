# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the CLIP encoder, guidance-conditioned LCM UNet, and VAE decoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt
from safetensors import safe_open

from . import config as config_module
from . import text_encoder_builder, unet_builder, vae_builder


def _load_component(model_dir: Path, component: str, filename: str) -> dict:
    path = Path(model_dir) / component / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing {component} weights: {path}")
    weights: dict = {}
    with safe_open(str(path), framework="numpy") as reader:
        for key in reader.keys():
            weights[key] = reader.get_tensor(key)
    return weights


def _compile(populate, *, precision: str, verbose: bool, configure=None) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    builder_config.clear_flag(trt.BuilderFlag.TF32)
    work_np = np.float16 if precision == "fp16" else np.float32
    work_trt = trt.float16 if precision == "fp16" else trt.float32
    populate(network, work_np, work_trt)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("Stable Diffusion TensorRT engine build failed")
    return bytes(plan)


def build_text_encoder_engine(
    weights, cfg, *, precision, verbose=False, penultimate=False, pooled=False
) -> bytes:
    def populate(network, work_np, work_trt):
        ids = network.add_input("input_ids", trt.int32, (1, cfg["max_position_embeddings"]))
        eos = network.add_input("eos_index", trt.int32, (1, 1)) if pooled else None
        out, pool = text_encoder_builder.build_text_encoder(
            network, ids, weights, cfg, work_np, penultimate=penultimate, eos_index=eos
        )
        if out.dtype != trt.float32:
            out = network.add_cast(out, trt.float32).get_output(0)
        out.name = "last_hidden_state"
        network.mark_output(out)
        if pool is not None:
            if pool.dtype != trt.float32:
                pool = network.add_cast(pool, trt.float32).get_output(0)
            pool.name = "pooled_output"
            network.mark_output(pool)

    return _compile(populate, precision=precision, verbose=verbose)


def build_unet_engine(weights, cfg, *, precision, verbose=False, quant=None) -> bytes:
    if quant is not None and precision != "fp16":
        raise NotImplementedError("stable_diffusion FP8 quantization requires precision=fp16")
    return _compile(
        lambda network, work_np, work_trt: unet_builder.build_unet(
            network, weights, cfg, work_np, work_trt, quant=quant
        ),
        precision=precision,
        verbose=verbose,
        configure=None,
    )


def build_vae_engine(weights, cfg, latent_size, *, precision, verbose=False) -> bytes:
    def populate(network, work_np, work_trt):
        latents = network.add_input(
            "latents", trt.float32, (1, cfg["latent_channels"], latent_size, latent_size)
        )
        source = latents
        if source.dtype != work_trt:
            source = network.add_cast(source, work_trt).get_output(0)
        out = vae_builder.build_decoder(network, source, weights, cfg, work_np)
        if out.dtype != trt.float32:
            out = network.add_cast(out, trt.float32).get_output(0)
        out.name = "image"
        network.mark_output(out)

    return _compile(populate, precision=precision, verbose=verbose)


def _tokenizer_bytes(model_dir: Path, component: str = "tokenizer") -> bytes:
    """The runtime reads a fast-tokenizer JSON; SD 1.5 ships the legacy pair.

    Checkpoints that already carry ``tokenizer.json`` are used as-is. The older
    ``vocab.json`` plus ``merges.txt`` layout is converted here rather than in
    the runtime, so the bundle always holds one format.
    """
    directory = Path(model_dir) / component
    path = directory / "tokenizer.json"
    if path.is_file():
        return path.read_bytes()
    if not (directory / "vocab.json").is_file() or not (directory / "merges.txt").is_file():
        raise FileNotFoundError(
            f"{directory} has neither tokenizer.json nor a vocab.json/merges.txt pair"
        )
    from transformers import CLIPTokenizerFast

    return CLIPTokenizerFast.from_pretrained(str(directory)).backend_tokenizer.to_str().encode()


def build(request, writer):
    if request.task != "image_generation" or request.backend != "trt":
        raise ValueError("LCM requires image_generation with backend=trt")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.dynamic_kv_cache
        or request.max_sequence_length is not None
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError("LCM supports one fp32 text-to-image request")
    height = request.image_height or 512
    width = request.image_width or 512
    if height != width or height < 64 or height % 64:
        raise ValueError("LCM builds square images with an edge divisible by 64")
    directory = Path(request.model_dir)
    cfg = config_module.resolve(directory, latent_size=height // 8)
    if cfg["text_encoder_2"] is not None or not cfg["unet"]["time_cond_proj_dim"]:
        raise ValueError("Expected LCM single-CLIP UNet with guidance conditioning")
    from diffusers import LCMScheduler

    scheduler = LCMScheduler.from_pretrained(directory / "scheduler")
    if (
        scheduler.config.prediction_type != "epsilon"
        or scheduler.config.thresholding
        or scheduler.config.clip_sample
    ):
        raise NotImplementedError("LCM requires unclipped epsilon prediction")
    writer.set_header(family="lcm", task=request.task, backend=request.backend)
    writer.add_bytes(
        "text_encoder.plan",
        build_text_encoder_engine(
            _load_component(directory, "text_encoder", "model.safetensors"),
            cfg["text_encoder"],
            precision="fp32",
            verbose=request.verbose,
        ),
    )
    writer.add_bytes(
        "unet.plan",
        build_unet_engine(
            _load_component(directory, "unet", "diffusion_pytorch_model.safetensors"),
            cfg["unet"],
            precision="fp32",
            verbose=request.verbose,
        ),
    )
    writer.add_bytes(
        "vae.plan",
        build_vae_engine(
            _load_component(directory, "vae", "diffusion_pytorch_model.safetensors"),
            cfg["vae"],
            height // 8,
            precision="fp32",
            verbose=request.verbose,
        ),
    )
    writer.add_bytes("tokenizer.json", _tokenizer_bytes(directory))
    writer.add_json(
        "runtime.json",
        dict(
            latent_size=height // 8,
            latent_channels=cfg["vae"]["latent_channels"],
            image_size=height,
            context_length=cfg["text_encoder"]["max_position_embeddings"],
            context_width=cfg["unet"]["cross_attention_dim"],
            scaling_factor=cfg["scaling_factor"],
            guidance_width=cfg["unet"]["time_cond_proj_dim"],
            original_inference_steps=scheduler.config.original_inference_steps,
            timestep_scaling=scheduler.config.timestep_scaling,
            alphas_cumprod=scheduler.alphas_cumprod.tolist(),
        ),
    )
