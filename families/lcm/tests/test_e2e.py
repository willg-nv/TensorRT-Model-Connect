# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned LCM image generation and scheduler parity through native execution."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "lcm"
TASKS = frozenset({"image_generation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            (
                line.strip()
                for line in Path(models_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and (not line.lstrip().startswith("#"))
            )
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    if not model_filters and (not testcase_filters):
        return (sorted(CASES), False)
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or (manifest["name"] in model_filters)
        )
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return (sorted(selected), True)


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(
                    name,
                    marks=pytest.mark.skip(
                        reason="direct E2E requires one of the three explicit E2E selectors"
                    ),
                )
                for name in names
            ]
        metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= 1, f"selected {FAMILY} E2E requires one GPU"
    return (binary, runtime_root)


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            image_height=manifest.get("image_height"),
            image_width=manifest.get("image_width"),
            video_num_frames=manifest.get("video_num_frames"),
            max_batch_size=int(manifest.get("max_batch_size", 1)),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            quantization=manifest.get("quantization"),
            fp32_layers=tuple((int(layer) for layer in manifest.get("fp32_layers", ()))),
        )
    )


def _run_json(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    case: dict,
    command: str,
    *arguments: str,
) -> dict:
    invocation = [
        str(binary),
        command,
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native {command} returned no JSON: {completed.stdout[-1000:]}"
    assert all((payload == payloads[0] for payload in payloads))
    return payloads[0]


def test_scheduler_reference(case_name):
    import numpy as np
    import torch
    from diffusers import LCMScheduler

    _, manifest, _ = CASES[case_name]
    directory = _model_dir(manifest)
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    helper = runtime_root / "test_lcm_scheduler"
    assert helper.is_file(), "build the family scheduler test target"
    for steps in (1, 4, 7, 50):
        scheduler = LCMScheduler.from_pretrained(directory / "scheduler")
        scheduler.set_timesteps(steps)
        source = torch.linspace(-0.5, 0.5, 32)
        sample = source.clone()
        epsilon = [torch.sin(torch.arange(32) * 0.1 + i) * 0.01 for i in range(steps)]
        generator = torch.Generator().manual_seed(123)
        noise_generator = torch.Generator().manual_seed(123)
        noises = [torch.randn(32, generator=noise_generator) for _ in range(steps)]
        expected = []
        for timestep, prediction in zip(scheduler.timesteps, epsilon):
            sample = scheduler.step(prediction, timestep, sample, generator=generator).prev_sample
            expected.append(sample.tolist())
        payload = dict(
            alphas=scheduler.alphas_cumprod.tolist(),
            original_steps=scheduler.config.original_inference_steps,
            scaling=scheduler.config.timestep_scaling,
            steps=steps,
            sample=source.tolist(),
            epsilon=[x.tolist() for x in epsilon],
            noise=[x.tolist() for x in noises],
        )
        completed = subprocess.run(
            [str(helper)], input=json.dumps(payload), capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["timesteps"] == scheduler.timesteps.tolist()
        np.testing.assert_allclose(result["states"], expected, atol=2e-5, rtol=2e-5)


def test_official_checkpoint_e2e(case_name, tmp_path):
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import LatentConsistencyModelPipeline

    _, manifest, case = CASES[case_name]
    directory = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(directory, bundle, manifest)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator().manual_seed(case["seed"])
    side = manifest["image_height"] // 8
    latents = torch.randn((1, 4, side, side), generator=generator)
    initial = tmp_path / "latents.f32"
    initial.write_bytes(latents.numpy().astype("<f4").tobytes())
    pipeline = LatentConsistencyModelPipeline.from_pretrained(
        directory,
        torch_dtype=torch.float32,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to("cuda")
    with torch.no_grad():
        expected = pipeline(
            case["prompt"],
            latents=latents.to("cuda"),
            num_inference_steps=case["num_inference_steps"],
            guidance_scale=case["guidance_scale"],
            height=manifest["image_height"],
            width=manifest["image_width"],
            output_type="np",
        ).images[0]
    del pipeline
    torch.cuda.empty_cache()
    images = []
    for steps in (1, 4, 4):
        output = tmp_path / f"native-{len(images)}.png"
        _run_json(
            binary,
            runtime_root,
            bundle,
            case,
            "generate-image",
            "--prompt",
            case["prompt"],
            "--output",
            str(output),
            "--initial-latents-raw",
            str(initial),
            "--num-steps",
            str(steps),
            "--guidance-scale",
            str(case["guidance_scale"]),
            "--seed",
            str(case["seed"]),
        )
        image = np.asarray(Image.open(output).convert("RGB"), dtype=np.float32) / 255
        assert image.shape == expected.shape == (512, 512, 3)
        assert np.isfinite(image).all() and float(image.std()) > 0.05
        images.append(image)
    mse = float(np.mean((images[0] - expected) ** 2))
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert psnr >= 40, f"LCM one-step image PSNR {psnr}"
    np.testing.assert_array_equal(images[1], images[2])
    assert not np.array_equal(images[0], images[1])
    print(f"LCM one-step image PSNR={psnr:.3f}; four-step native output is reproducible")
