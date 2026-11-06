# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host regression for the real E2E runtime staging boundary."""

import json
from pathlib import Path
import subprocess

import pytest

from .test_e2e import _file_identity, _isolated_runtime, _run_native


def test_isolated_runtime_excludes_siblings_and_dereferences_links(tmp_path):
    source = tmp_path / "build"
    source.mkdir()
    for name in ("core", "runtime", "backend_trt", "model_llama", "model_sibling"):
        (source / f"libtrtmc_{name}.so.1").write_bytes(name.encode())
        (source / f"libtrtmc_{name}.so").symlink_to(f"libtrtmc_{name}.so.1")
    destination = _isolated_runtime(source, tmp_path / "isolated")
    assert {path.name for path in destination.iterdir()} == {
        "libtrtmc_core.so", "libtrtmc_runtime.so",
        "libtrtmc_backend_trt.so", "libtrtmc_model_llama.so",
    }
    for path in destination.iterdir():
        assert not path.is_symlink()
        assert _file_identity(path)["sha256"] == _file_identity(source / path.name)["sha256"]
    # Neither staging nor subsequent writes may modify another family's artifacts.
    (destination / "libtrtmc_model_llama.so").write_bytes(b"replacement")
    assert (source / "libtrtmc_model_llama.so").read_bytes() == b"model_llama"
    assert (source / "libtrtmc_model_sibling.so").read_bytes() == b"model_sibling"


def test_isolated_runtime_requires_complete_selected_closure(tmp_path):
    source = tmp_path / "build"
    source.mkdir()
    (source / "libtrtmc_core.so").write_bytes(b"core")
    with pytest.raises(AssertionError, match="libtrtmc_runtime.so"):
        _isolated_runtime(source, tmp_path / "isolated")


def test_failed_native_process_retains_receipt_and_still_raises(tmp_path):
    # Exercise failure recording with a real failing host command, without a GPU.
    with pytest.raises(subprocess.CalledProcessError) as failure:
        _run_native(Path("/bin/false"), tmp_path, tmp_path / "absent.bundle", "prompt",
                    {"max_new_tokens": 10}, 1, tmp_path)
    receipt = json.loads((tmp_path / "native-execution.json").read_text())
    assert receipt["returncode"] == failure.value.returncode != 0
    assert receipt["argv"] == failure.value.cmd
    assert receipt["stdout"] == failure.value.stdout
    assert receipt["stderr"] == failure.value.stderr
