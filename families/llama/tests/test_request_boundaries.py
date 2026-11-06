# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invalid public build requests fail before weights or GPU allocation."""

import json

import pytest

from tensorrt_model_connect import BuildRequest, build


@pytest.mark.parametrize('changes,message', [
    ({'task': 'embedding'}, 'task'),
    ({'precision': 'int8'}, 'precision'),
    ({'backend': 'unknown'}, 'backend'),
    ({'tensor_parallel_size': 2}, 'tensor.parallel'),
    ({'context_parallel_size': 2}, 'context_parallel_size'),
    ({'max_batch_size': 2}, 'max_batch_size'),
    ({'max_sequence_length': 0}, 'max_sequence_length'),
    ({'max_sequence_length': 2049}, 'context capacity'),
    ({'quantization': 'int4'}, 'quantization'),
    ({'image_height': 32}, 'image_height'),
])
def test_invalid_llama_request_never_opens_weights_or_publishes(tmp_path, changes, message):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    # TinyLlama's real geometry and context; deliberately no weights are supplied.
    (checkpoint / 'config.json').write_text(json.dumps({
        'model_type': 'llama', 'architectures': ['LlamaForCausalLM'],
        'vocab_size': 32000, 'hidden_size': 2048, 'intermediate_size': 5632,
        'num_hidden_layers': 22, 'num_attention_heads': 32, 'num_key_value_heads': 4,
        'hidden_act': 'silu', 'max_position_embeddings': 2048,
        'rms_norm_eps': 1e-5, 'rope_theta': 10000.0,
        'bos_token_id': 1, 'eos_token_id': 2,
    }))
    output = tmp_path / 'candidate.bundle'
    options = dict(model_dir=checkpoint, output_path=output, family='llama',
                   task='text_generation', precision='fp16', max_sequence_length=256,
                   max_batch_size=1, tensor_parallel_size=1)
    options.update(changes)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        build(BuildRequest(**options))
    assert not output.exists()
    assert {path.name for path in tmp_path.iterdir()} == {'checkpoint'}
