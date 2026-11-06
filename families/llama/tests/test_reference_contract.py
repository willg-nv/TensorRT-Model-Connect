# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-contract tests need no model download or GPU."""
import pytest

from families.llama.tests.reference_contract import self_check
from families.llama.tests.test_e2e import _assert_correctness


def test_existing_nonempty_oracle_still_rejects_eos_only_reference():
    case = {'max_new_tokens': 10, 'use_chat_template': False}
    result = self_check(case, {'normalized_edit_distance': 0.25}, [2], '')
    assert result == {'status': 'failed', 'reason': 'reference_rejected_by_existing_oracle'}


def test_nonempty_reference_must_also_meet_its_semantic_assertions():
    case = {'max_new_tokens': 10, 'use_chat_template': True, 'expected_answers': ['Paris']}
    assert self_check(case, {}, [10], 'Capital: Paris')['status'] == 'passed'
    assert self_check(case, {}, [10], 'Capital: London')['status'] == 'failed'


@pytest.mark.parametrize('ids,text', [([], 'Paris'), ([1, 2, 3], 'Paris')])
def test_reference_retains_token_count_contract(ids, text):
    assert self_check({'max_new_tokens': 2}, {}, ids, text)['status'] == 'failed'


def test_explicit_eos_reference_contract_accepts_only_declared_termination():
    case = {'max_new_tokens': 10, 'expected_termination': 'eos_only',
            'expected_continuation_token_ids': [2]}
    assert self_check(case, {}, [2], '')['status'] == 'passed'
    assert self_check(case, {}, [3], '')['status'] == 'failed'


@pytest.mark.parametrize('native_ids,native_text,reference_ids,reference_text,decoded', [
    ([3], '', [2], '', ''),
    ([2], '', [3], '', ''),
    ([2, 2], '', [2], '', ''),
    ([2], 'unexpected', [2], '', ''),
    ([2], '', [2], 'unexpected', ''),
    ([2], '', [2], '', 'unexpected'),
])
def test_eos_native_contract_rejects_inexact_agreement(
        native_ids, native_text, reference_ids, reference_text, decoded):
    case = {'max_new_tokens': 10, 'expected_termination': 'eos_only',
            'expected_continuation_token_ids': [2]}
    with pytest.raises(AssertionError):
        _assert_correctness({'token_ids': native_ids, 'text': native_text}, case, {},
                            reference_ids, reference_text, None, decoded)


def test_unknown_termination_contract_is_rejected():
    assert self_check({'max_new_tokens': 10, 'expected_termination': 'unknown'},
                      {}, [10], 'Paris')['status'] == 'failed'
