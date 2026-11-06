# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression for capture integrity and full-vocabulary diagnostics."""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from .precision_diagnostics import capture_official_scores, read_trace, score_metrics


@pytest.fixture(scope='module')
def trace_program(tmp_path_factory):
    root = Path(__file__).resolve().parents[3]
    directory = tmp_path_factory.mktemp('llama-logits-trace')
    source = directory / 'trace.cpp'
    source.write_text('''
#include "families/llama/runtime/logits_trace.h"
#include <limits>
int main(int argc, char** argv) {
    const std::vector<int32_t> prefix{1, 7};
    const std::vector<float> scores{0.333333343f, -65504.0f, 1e-30f};
    const auto original = scores;
    trtmc::LlamaLogitsTrace trace(argc > 1 ? argv[1] : nullptr);
    trace.record(prefix, scores);
    trace.record({1, 7, 0}, {1.0f, 2.0f, 3.0f});
    return scores == original ? 0 : 1;
}
''')
    compiler = shutil.which('c++')
    assert compiler, 'native development prerequisites include a C++ compiler'
    binary = directory / 'trace'
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I', str(root), str(source), '-o', str(binary)], check=True,
                   capture_output=True, text=True)
    return binary


def test_native_trace_roundtrips_scores_without_mutating_them(trace_program, tmp_path):
    subprocess.run([str(trace_program)], check=True, cwd=tmp_path)
    assert list(tmp_path.iterdir()) == []
    path = tmp_path / 'trace.jsonl'
    subprocess.run([str(trace_program), str(path)], check=True)
    rows = read_trace(path, [1, 7], [0, 2], 3)
    np.testing.assert_array_equal(np.asarray(rows[0]['logits'], dtype=np.float32),
                                  np.array([0.333333343, -65504.0, 1e-30], dtype=np.float32))
    # Same continuation IDs with wrong preprocessing must never be compared.
    with pytest.raises(AssertionError, match='prefix mismatch'):
        read_trace(path, [7], [0, 2], 3)
    with pytest.raises(AssertionError, match='missing sampler'):
        read_trace(path, [1, 7], [0], 3)
    with pytest.raises(AssertionError, match='incomplete vocabulary'):
        read_trace(path, [1, 7], [0, 2], 2)


def test_trace_rejects_nonfinite_or_unrecorded_scores(tmp_path):
    path = tmp_path / 'trace.jsonl'
    path.write_text(json.dumps({'prefix_token_ids': [1], 'logits': [None, 1.0]}) + '\n')
    with pytest.raises(AssertionError, match='nonfinite'):
        read_trace(path, [1], [1], 2)


def test_full_vocabulary_residual_survives_identical_winning_token():
    result = score_metrics([4, 3, -20], [4, 3, -30])
    assert result['argmax_equal']
    assert result['left_top2']['margin'] == result['right_top2']['margin'] == 1
    assert result['max_abs'] == 10
    assert result['rms'] == pytest.approx(np.sqrt(100 / 3))
    # Reversing score margins exposes a different winner at the same prefix.
    result = score_metrics([3, 4, -20], [4, 3, -20])
    assert not result['argmax_equal']
    assert result['left_top2']['token_ids'] == [1, 0]
    assert result['right_top2']['token_ids'] == [0, 1]


def test_official_cached_capture_matches_independent_full_forced_prefixes():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    with torch.random.fork_rng(devices=[]):
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=32)).eval()
    inputs = {'input_ids': torch.tensor([[1, 7, 9]]),
              'attention_mask': torch.ones((1, 3), dtype=torch.long)}
    with torch.inference_mode():
        winner = model(**inputs).logits[0, -1].argmax().item()
    forced_ids = [(winner + 1) % 32, 4, 2]
    actual = capture_official_scores(model, inputs, forced_ids)
    with torch.inference_mode():
        for i in range(len(forced_ids)):
            prefix = torch.tensor([[1, 7, 9] + forced_ids[:i]])
            expected = model(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                             use_cache=False).logits[0, -1]
            torch.testing.assert_close(torch.from_numpy(actual[i]), expected)
