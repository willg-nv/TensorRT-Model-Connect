# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.vilt.support import describe


def test_vqa_checkpoint_has_one_owner():
    family, support = resolve_family(
        ModelMetadata({"model_type": "vilt", "architectures": ["ViltForQuestionAnswering"]}, {})
    )
    assert family == "vilt"
    assert support.tasks == ("images_text_to_text",)


def test_other_vilt_heads_are_not_claimed():
    assert (
        describe(
            ModelMetadata(
                {"model_type": "vilt", "architectures": ["ViltForTokenClassification"]}, {}
            )
        )
        is None
    )
