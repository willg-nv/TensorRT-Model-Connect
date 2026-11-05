# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map real sharded BF16 reference parameters without GPU compilation."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from safetensors.torch import save_file  # noqa: E402
from transformers.models.llama.configuration_llama import LlamaConfig  # noqa: E402
from transformers.models.llama.modeling_llama import LlamaForCausalLM  # noqa: E402

from ..checkpoint_mapper import load_standard_weights  # noqa: E402
from ..config import ModelConfig  # noqa: E402


@pytest.mark.parametrize("tied", [False, True])
def test_sharded_official_bf16_weights_map_to_fp16_with_compact_kv(tmp_path, tied):
    config = LlamaConfig(hidden_size=8, intermediate_size=12, vocab_size=16,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=32, tie_word_embeddings=tied)
    # A local test fixture seed does not change the E2E workload or its seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        official = LlamaForCausalLM(config).bfloat16().eval()
    config.save_pretrained(tmp_path)
    tensors = {name: value.detach().clone() for name, value in official.state_dict().items()}
    if tied:
        del tensors["lm_head.weight"]
    shards = [{}, {}]
    weight_map = {}
    for index, (name, tensor) in enumerate(sorted(tensors.items())):
        shard = index % 2
        shards[shard][name] = tensor
        weight_map[name] = f"model-{shard + 1:05d}-of-00002.safetensors"
    # An unindexed duplicate in a later shard must not override the index.
    embedding_name = "model.embed_tokens.weight"
    assigned = 0 if embedding_name in shards[0] else 1
    shards[1 - assigned][embedding_name] = torch.zeros_like(tensors[embedding_name])
    for index, shard in enumerate(shards):
        save_file(shard, tmp_path / f"model-{index + 1:05d}-of-00002.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    weights = load_standard_weights(tmp_path, ModelConfig.from_dir(tmp_path), precision="fp16")
    mapping = {
        "w_q": "self_attn.q_proj.weight", "w_k": "self_attn.k_proj.weight",
        "w_v": "self_attn.v_proj.weight", "w_o": "self_attn.o_proj.weight",
        "w_gate": "mlp.gate_proj.weight", "w_up": "mlp.up_proj.weight",
        "w_down": "mlp.down_proj.weight",
    }
    for target, source in mapping.items():
        expected = tensors[f"model.layers.0.{source}"].float().numpy().T.astype(np.float16)
        actual = weights[f"layer.0.{target}"]
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float16 and actual.flags.c_contiguous
    for target, source in (("input_norm", "input_layernorm"),
                           ("post_attn_norm", "post_attention_layernorm")):
        actual = weights[f"layer.0.{target}"]
        np.testing.assert_array_equal(actual, tensors[f"model.layers.0.{source}.weight"].float().numpy())
        assert actual.dtype == np.float32
    np.testing.assert_array_equal(weights["final_norm"], tensors["model.norm.weight"].float().numpy())
    np.testing.assert_array_equal(weights["embedding"], tensors[embedding_name].float().numpy().astype(np.float16))
    head = embedding_name if tied else "lm_head.weight"
    np.testing.assert_array_equal(weights["w_out"], tensors[head].float().numpy().T.astype(np.float16))
    assert weights["_attention_size"] == 8
    assert weights["_kv_attention_size"] == 4
    assert weights["layer.0.w_k"].shape == weights["layer.0.w_v"].shape == (8, 4)
    assert not np.shares_memory(weights["embedding"], weights["w_out"])
