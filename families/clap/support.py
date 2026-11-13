# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The checkpoint's normalized text tower; audio Tasks are not advertised."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("clap",),
    tasks=("text_to_embedding",),
    default_task="text_to_embedding",
    default_precision="fp32",
)
