# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TinyLlama selection and preprocessing contracts, without model execution."""

import pytest

from .test_e2e import _CASES, _native_arguments, _render_prompt, _text_threshold


@pytest.fixture
def tinyllama_cases():
    manifest, chat = _CASES["tinyllama-1.1b"]
    raw_manifest, raw = _CASES["tinyllama-1.1b-raw"]
    assert raw_manifest is manifest
    return manifest, chat, raw


def test_tinyllama_exact_manifest_and_unchanged_chat_oracle(tinyllama_cases):
    manifest, chat, raw = tinyllama_cases
    assert manifest["hf_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert manifest["hf_revision"] == "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    assert manifest["family"] == "llama"
    assert manifest["task"] == "text_generation"
    assert manifest["precision"] == "fp16"
    assert manifest["trust_remote_code"] is False
    assert manifest["max_batch_size"] == manifest["tensor_parallel_size"] == 1
    assert manifest["max_sequence_length"] == 256
    assert "model.safetensors" in manifest["checkpoint_files"]
    assert chat == {
        "name": "tinyllama-1.1b",
        "premerge": True,
        "reference_precision": "fp32",
        "prompt": "What is the capital of France? Answer in one word.",
        "max_new_tokens": 10,
        "use_chat_template": True,
        "enable_thinking": False,
    }
    assert raw == {**chat, "name": "tinyllama-1.1b-raw", "use_chat_template": False,
                   "expected_termination": "eos_only", "expected_continuation_token_ids": [2]}
    assert _text_threshold(chat, {}) == 0.15
    assert _text_threshold(raw, {}) == 0.25


def test_tinyllama_retains_a_nonempty_raw_continuation(tinyllama_cases):
    manifest, _, _ = tinyllama_cases
    continuation_manifest, case = _CASES["tinyllama-1.1b-raw-continuation"]
    assert continuation_manifest is manifest
    assert [c['name'] for c in manifest['testcases']] == [
        'tinyllama-1.1b', 'tinyllama-1.1b-raw', 'tinyllama-1.1b-raw-continuation']
    assert case == {
        'name': 'tinyllama-1.1b-raw-continuation', 'premerge': True,
        'reference_precision': 'fp32', 'prompt': 'The capital of France is',
        'max_new_tokens': 10, 'use_chat_template': False, 'enable_thinking': False,
        'expected_answers': ['Paris'],
    }


@pytest.mark.parametrize("mode", ["chat", "raw"])
def test_tinyllama_reference_and_native_prompt_modes(tinyllama_cases, mode, tmp_path):
    _, chat, raw = tinyllama_cases
    case = chat if mode == "chat" else raw
    calls = []

    class RecordingTokenizer:
        def apply_chat_template(self, messages, **options):
            calls.append((messages, options))
            return "rendered chat"

        def __call__(self, text, **options):
            calls.append((text, options))
            return "tokenized inputs"

    assert _render_prompt(RecordingTokenizer(), case["prompt"], case) == "tokenized inputs"
    if mode == "chat":
        assert calls == [
            ([{"role": "user", "content": case["prompt"]}], {
                "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
            }),
            ("rendered chat", {"return_tensors": "pt", "add_special_tokens": False}),
        ]
    else:
        assert calls == [(case["prompt"], {"return_tensors": "pt"})]
    arguments = _native_arguments(tmp_path / "model.bundle", tmp_path, case["prompt"], case)
    assert arguments[arguments.index("--use-chat-template") + 1] == str(mode == "chat").lower()
    assert arguments[arguments.index("--enable-thinking") + 1] == "false"
