# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import pytest
from families.dinov2_with_registers import config, position


def test_position_interpolation_matches_reference_and_keeps_class_row():
    import torch
    from transformers import Dinov2WithRegistersConfig
    from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import Dinov2WithRegistersEmbeddings

    cfg = Dinov2WithRegistersConfig(image_size=56, patch_size=14, hidden_size=8,
                                    num_attention_heads=2, num_register_tokens=4)
    embeddings = Dinov2WithRegistersEmbeddings(cfg)
    torch.manual_seed(17)
    with torch.no_grad():
        embeddings.position_embeddings.copy_(torch.randn_like(embeddings.position_embeddings))
    table = embeddings.position_embeddings.detach().numpy()
    actual = position.interpolate_position_table(table, 4, 2)
    expected = embeddings.interpolate_pos_encoding(torch.empty(1,5,8),28,28).detach().numpy()[0]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[0], table[0,0])
    np.testing.assert_array_equal(position.interpolate_position_table(table,4,4),table[0])


def _fixture(tmp_path, count=4, shape=(1,4,8)):
    (tmp_path/'config.json').write_text(json.dumps({'architectures':['Dinov2WithRegistersModel'],
        'hidden_size':8,'num_attention_heads':2,'num_register_tokens':count,'image_size':56}))
    return {'embeddings.register_tokens':np.zeros(shape,np.float32),
            'embeddings.position_embeddings':np.zeros((1,17,8),np.float32),
            'embeddings.patch_embeddings.projection.weight':np.zeros((8,3,14,14),np.float32),
            'encoder.layer.0.mlp.fc1.weight':np.zeros((32,8),np.float32)}


def test_register_geometry_is_checked_against_checkpoint(tmp_path):
    weights = _fixture(tmp_path)
    resolved = config.resolve(tmp_path, weights)
    assert resolved['num_register_tokens'] == 4
    assert resolved['hidden_size'] == 8
    weights = _fixture(tmp_path,count=3)
    with pytest.raises(ValueError, match='num_register_tokens'):
        config.resolve(tmp_path,weights)
    weights = _fixture(tmp_path,shape=(1,4,16))
    with pytest.raises(ValueError, match='register width'):
        config.resolve(tmp_path,weights)


def test_zero_registers_are_rejected(tmp_path):
    weights = _fixture(tmp_path,count=0,shape=(1,0,8))
    with pytest.raises(ValueError, match='positive register'):
        config.resolve(tmp_path,weights)
