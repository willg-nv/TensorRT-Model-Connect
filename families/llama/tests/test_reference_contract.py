# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-contract tests need no model download or GPU."""
import pytest

from families.llama.tests.reference_contract import self_check


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
