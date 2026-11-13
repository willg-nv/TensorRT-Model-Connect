# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_owlvit_and_owlv2_have_distinct_owners():
    for model_type,architecture in [('owlvit','OwlViTForObjectDetection'),('owlv2','Owlv2ForObjectDetection')]:
        family,support=resolve_family(ModelMetadata({'model_type':model_type,'architectures':[architecture]},{}))
        assert family == model_type
        assert support.tasks == ('image_text_to_boxes',)
