# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reduced official-operation baselines; these do not qualify TensorRT execution."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.models.llama.configuration_llama import LlamaConfig  # noqa: E402
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm  # noqa: E402


@pytest.fixture(autouse=True)
def preserve_rng_state():
    with torch.random.fork_rng(devices=[]):
        yield


def test_official_projection_separates_storage_from_accumulation():
    # Opposite large products cancel in FP32. Rounding each product to FP16
    # first produces inf - inf; it is not the official linear operation.
    config = LlamaConfig(hidden_size=4, intermediate_size=4, hidden_act="silu",
                         num_attention_heads=1, num_key_value_heads=1)
    operation = LlamaMLP(config).half().eval().gate_proj
    stored = torch.tensor([[256.0, -256.0, 1.0, 1.0]], dtype=torch.float16)
    with torch.no_grad():
        operation.weight.copy_(stored.expand(4, -1))
        x = torch.tensor([[256.0, 256.0, 1.0, 1.0]], dtype=torch.float16)
        actual = operation(x)
    expected = (x.float() @ operation.weight.float().T).half()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual, torch.full_like(actual, 2.0))
    assert not torch.isfinite((x * stored).sum()).item()

    # FP32 accumulation cannot recover information rounded away at storage.
    source = torch.tensor([[1.0004, 1.0004, 1.0004, 1.0004]])
    with torch.no_grad():
        operation.weight.fill_(1.0)
        actual = operation(source.half())
    torch.testing.assert_close(actual, (source.half().float() @ operation.weight.float().T).half(),
                               rtol=0, atol=0)
    assert not torch.equal(actual.float(), source @ operation.weight.float().T)


def test_official_rmsnorm_epsilon_and_rounding_boundaries():
    x = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [1e-7, -1e-7, 2e-7, -2e-7],
        [1e-4, 2e-4, -1e-4, 3e-4],
        [0.3333, -0.8125, 1.2345, -3.4567],
        [65504.0, -65504.0, 32752.0, -32752.0],
    ], dtype=torch.float16)
    operation = LlamaRMSNorm(4, eps=1e-5).half().eval()
    with torch.no_grad():
        operation.weight.copy_(torch.tensor([0.75, 1.125, 0.3333, 1.875]))
        actual = operation(x)
    variance = x.float().square().mean(-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + operation.variance_epsilon)
    expected = operation.weight * normalized.half()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float16 and torch.isfinite(actual).all()
    # Both alternatives are detectably different on these same operands.
    late_round = (operation.weight.float() * normalized).half()
    rounded_eps = torch.tensor(operation.variance_epsilon, dtype=torch.float16).float()
    early_eps_round = operation.weight * (x.float() * torch.rsqrt(variance + rounded_eps)).half()
    assert not torch.equal(actual, late_round)
    assert not torch.equal(actual, early_eps_round)
    assert not torch.isfinite(x[-1].square()).all()


def test_official_silu_is_finite_at_tiny_and_extreme_inputs():
    operation = LlamaMLP(LlamaConfig(hidden_size=4, intermediate_size=4, hidden_act="silu",
                                    num_attention_heads=1, num_key_value_heads=1))
    x = torch.tensor([-65504.0, -100.0, -1e-7, 0.0, 1e-7, 100.0, 65504.0],
                     dtype=torch.float16)
    actual = operation.act_fn(x)
    expected = torch.nn.functional.silu(x.float()).half()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == x.dtype and torch.isfinite(actual).all()
    assert actual[0] == 0 and actual[-1] == x[-1]
