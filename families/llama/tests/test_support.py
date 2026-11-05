# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free discovery checks; runnable with python -S -m unittest."""

import unittest

from tensorrt_model_connect.model_support import (
    AmbiguousFamilyError,
    FamilyResolutionError,
    ModelMetadata,
    resolve_family,
)

from families.llama.support import describe


class LlamaSupportTest(unittest.TestCase):
    def test_exact_normalized_identity_and_defaults(self):
        for identity in ("llama", "LLaMA", " LLAMA "):
            with self.subTest(identity=identity):
                metadata = ModelMetadata(
                    {"model_type": identity, "architectures": ["LlamaForCausalLM"]}, {}
                )
                support = describe(metadata)
                self.assertIsNotNone(support)
                self.assertEqual(support.tasks, ("text_generation",))
                self.assertEqual(support.default_task, "text_generation")
                self.assertEqual(support.default_precision, "fp32")
                self.assertEqual(resolve_family(metadata), ("llama", support))

    def test_names_prefixes_and_architecture_alone_do_not_claim_ownership(self):
        configs = (
            {},
            {"model_type": None},
            {"model_type": ["llama"]},
            {"model_type": "tinyllama"},
            {"model_type": "llama2"},
            {"model_type": "llama_moe"},
            {"model_type": "mllama"},
            {"model_type": "qwen2", "architectures": ["LlamaForCausalLM"]},
            {"architectures": ["LlamaForCausalLM"]},
            {"_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
        )
        for config in configs:
            with self.subTest(config=config):
                self.assertIsNone(describe(ModelMetadata(config, {}, ("config.json",))))

    def test_unknown_identity_has_no_fallback_owner(self):
        with self.assertRaises(FamilyResolutionError):
            resolve_family(ModelMetadata({"model_type": "llama_unknown"}, {}))

    def test_conflicting_exact_identities_are_ambiguous(self):
        # Contradictory root metadata must never select an owner by priority.
        metadata = ModelMetadata(
            {"model_type": "llama"}, {"_class_name": "StableDiffusionPipeline"}
        )
        with self.assertRaises(AmbiguousFamilyError) as raised:
            resolve_family(metadata)
        self.assertEqual(
            {family for family, _ in raised.exception.matches}, {"llama", "stable_diffusion"}
        )


if __name__ == "__main__":
    unittest.main()
