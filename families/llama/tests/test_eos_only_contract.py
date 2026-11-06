# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict nonempty-output contract: EOS agreement cannot bypass the family oracle."""
import pytest
from .test_e2e import _assert_correctness


def check(case, ids=None, reference=None, text="", decoded="", reference_text=""):
    _assert_correctness({"token_ids": [2] if ids is None else ids, "text": text},
                        {"max_new_tokens": 10, **case}, {"contract_ned_threshold": 0.25},
                        [2] if reference is None else reference, reference_text, None, decoded)


def test_eos_only_agreement_still_fails_original_nonempty_oracle():
    with pytest.raises(AssertionError):
        check({"expected_eos_only_token_ids": [2]})


@pytest.mark.parametrize("change", [
    {"ids": [3]}, {"reference": [3]}, {"ids": [2, 3]},
    {"decoded": "unreported text"}, {"reference_text": "expected answer"},
    {"text": "different output"},
])
def test_reviewed_exception_rejects_mismatch_or_nonempty_reference(change):
    with pytest.raises(AssertionError):
        check({"expected_eos_only_token_ids": [2]}, **change)


def test_unreviewed_case_still_requires_nonempty_text():
    with pytest.raises(AssertionError):
        check({})


def test_nonempty_reference_comparison_keeps_original_threshold():
    check({}, ids=[7], reference=[7], text="Paris", decoded="Paris", reference_text="Paris")
    with pytest.raises(AssertionError):
        check({}, ids=[7], reference=[7], text="London", decoded="London", reference_text="Paris")
