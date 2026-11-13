# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_pipeline_has_one_owner():
    family, support = resolve_family(
        ModelMetadata({}, {"_class_name": "LatentConsistencyModelPipeline"})
    )
    assert family == "lcm"
    assert support.tasks == ("image_generation",)
