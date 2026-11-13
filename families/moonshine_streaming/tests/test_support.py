# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.moonshine_streaming.support import describe


def test_moonshine_streaming_has_one_speech_owner():
    family, support = resolve_family(ModelMetadata({"model_type": "moonshine_streaming"}, {}))
    assert family == "moonshine_streaming"
    assert support.tasks == ("speech_transcription",)


def test_encoder_only_is_not_a_transcription_model():
    assert describe(ModelMetadata({"model_type": "moonshine_streaming_encoder"}, {})) is None


def test_local_attention_excludes_both_window_endpoints():
    from families.moonshine_streaming.builders import local_mask

    mask = local_mask(40, 16, 4)[0, 0]
    assert mask[20, 5] == 0 and mask[20, 4] < -1e8
    assert mask[20, 23] == 0 and mask[20, 24] < -1e8
    causal = local_mask(40, 16, 0)[0, 0]
    assert causal[20, 20] == 0 and causal[20, 21] < -1e8
