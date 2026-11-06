# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute a frozen family reference matrix before spending a native build.

This self-check applies the existing family oracle to its own reference output.
It never grants native qualification or changes an accepted testcase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


REFERENCE_CONTRACT_SCHEMA = 1

def self_check(case, thresholds, token_ids, text):
    from families.llama.tests.test_e2e import _assert_correctness

    try:
        _assert_correctness({'token_ids': token_ids, 'text': text}, case, thresholds,
                            token_ids, text, None, text)
    except AssertionError:
        return {'status': 'failed', 'reason': 'reference_rejected_by_existing_oracle'}
    return {'status': 'passed'}


def run(manifest_path, output, device='cpu'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from families.llama.tests.test_e2e import _checkpoint, _hf_reference, _thresholds

    if device == 'cpu':
        torch.set_num_threads(4)
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw)
    if manifest['family'] != 'llama' or manifest['task'] != 'text_generation' or manifest.get('trust_remote_code'):
        raise ValueError('Expected a local llama text-generation manifest without remote code')
    report = {'schema': 1, 'kind': 'reference_contract', 'diagnostic_only': True,
              'checkpoint': manifest['hf_id'], 'revision': manifest['hf_revision'],
              'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'device': device,
              'cpu_threads': torch.get_num_threads(),
              'cases': [], 'status': 'running', 'qualification': 'not_established'}
    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        checkpoint = _checkpoint(manifest)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        models = {}
        dtypes = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
        for index, case in enumerate(manifest['testcases']):
            if case.get('reference') == 'none' or case.get('contract_only'):
                raise ValueError('This reference self-check requires a declared executable reference')
            precision = case.get('reference_precision', manifest.get('reference_precision', manifest['precision']))
            if precision not in models:
                models[precision] = AutoModelForCausalLM.from_pretrained(
                    checkpoint, local_files_only=True, trust_remote_code=False,
                    dtype=dtypes[precision]).eval().to(device)
            frame = output / str(index)
            frame.mkdir()
            result = _hf_reference(checkpoint, manifest, case, case['prompt'], [], torch,
                                   reference=(tokenizer, models[precision]), diagnostic_dir=frame)
            outcome = self_check(case, _thresholds(case['name']), result[0], result[1])
            report['cases'].append({'name': case['name'], 'reference_precision': precision,
                                    'token_ids': result[0], 'text': result[1], **outcome})
            save()
        report['status'] = 'passed' if report['cases'] and all(c['status'] == 'passed' for c in report['cases']) else 'failed'
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as error:
        report.update(status='execution_error', error_type=type(error).__name__)
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Family manifest basename, without .json')
    parser.add_argument('--output', required=True, type=Path, help='New private evidence directory')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.manifest):
        parser.error('Use a manifest basename')
    result = run(Path(__file__).parent / 'manifests' / (args.manifest + '.json'), args.output, args.device)
    print(json.dumps({'kind': 'reference_contract', 'status': result['status'], 'qualification': 'not_established'}))
    return 0 if result['status'] == 'passed' else 2 if result['status'] == 'execution_error' else 1


if __name__ == '__main__':
    raise SystemExit(main())
