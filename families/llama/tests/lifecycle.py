# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the requested checkpoint through one loaded public C++ Task."""

import json
import math
from pathlib import Path
import shutil
import subprocess

from .precision_diagnostics import read_trace


def compile_consumer(runtime_root, destination):
    root = Path(__file__).resolve().parents[3]
    compiler = shutil.which('c++')
    assert compiler, 'native development prerequisites include a C++ compiler'
    command = [compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
               '-I', str(root / 'core/runtime/include'),
               str(Path(__file__).with_name('cpp') / 'text_generation_lifecycle.cpp'),
               '-L', str(runtime_root), '-ltrtmc_runtime', '-ltrtmc_core',
               f'-Wl,-rpath,{runtime_root}', '-o', str(destination)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    destination.with_suffix('.compile.json').write_text(json.dumps({
        'argv': command, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr}, indent=2) + '\n')
    result.check_returncode()
    return destination


def requests(tokenizer, manifest):
    """Use the bundle limit, rather than the larger checkpoint context limit."""
    capacity = manifest['max_sequence_length']
    maximum = ' hello' * (capacity - 2)
    assert len(tokenizer.encode(maximum)) + 1 == capacity
    assert tokenizer.encode('') == [tokenizer.bos_token_id]
    raw = next(case for case in manifest['testcases']
               if case['name'] == 'tinyllama-1.1b-raw-continuation')
    chat = next(case for case in manifest['testcases'] if case['name'] == manifest['name'])

    def request(name, prompt, budget, chat=False, mode='auto', error=''):
        return dict(name=name, prompt=prompt, max_new_tokens=budget,
                    use_chat_template=chat, enable_thinking=False, mode=mode, error=error)

    return [
        request('raw-before', raw['prompt'], raw['max_new_tokens']),
        request('chat-before', chat['prompt'], chat['max_new_tokens'], True),
        request('minimum-before', '', 1),
        request('maximum', maximum, 1),
        request('overflow', maximum, 2, error='exceed the model\'s fixed KV cache capacity'),
        request('invalid-mode', raw['prompt'], 1, mode='unsupported',
                error='unsupported generation mode'),
        request('unicode', 'Café\n你好, 世界!', 1),
        request('chat-after', chat['prompt'], chat['max_new_tokens'], True),
        request('raw-after', raw['prompt'], raw['max_new_tokens']),
        request('minimum-after', '', 1),
    ]


def exercise_lifecycle(bundle, runtime_root, model_dir, manifest, baseline, directory):
    from transformers import AutoTokenizer
    from .test_e2e import _file_identity, _render_prompt

    directory.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True,
                                              trust_remote_code=False)
    cases = requests(tokenizer, manifest)
    (directory / 'inputs.json').write_text(json.dumps({
        'manifest': manifest, 'bundle': _file_identity(bundle),
        'requests': cases, 'scope': 'one loaded batch-1 TP-1 Task; independent requests'},
        indent=2, ensure_ascii=False) + '\n')
    consumer = compile_consumer(runtime_root, directory / 'consumer')
    command = [str(consumer), str(bundle), str(runtime_root), str(directory)]
    for case in cases:
        command.extend([case['name'], case['prompt'], str(case['max_new_tokens']),
                        str(case['use_chat_template']).lower(), case['mode'], case['error']])
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    (directory / 'execution.json').write_text(json.dumps({
        'argv': command, 'returncode': completed.returncode,
        'stdout': completed.stdout, 'stderr': completed.stderr}, indent=2) + '\n')
    completed.check_returncode()
    outcomes = {}
    for case in cases:
        base = directory / case['name']
        error = base.with_suffix('.error')
        if case['error']:
            assert error.is_file(), f"{case['name']} unexpectedly accepted"
            assert case['error'] in error.read_text()
            assert not base.with_suffix('.ids').exists()
            assert not base.with_suffix('.logits.jsonl').exists(), 'rejected request executed'
            outcomes[case['name']] = {'error': error.read_text()}
            continue
        assert not error.exists(), error.read_text() if error.exists() else ''
        ids = [int(value) for value in base.with_suffix('.ids').read_text().split()]
        assert 1 <= len(ids) <= case['max_new_tokens']
        assert all(0 <= token < 32000 for token in ids)
        prompt_ids = _render_prompt(tokenizer, case['prompt'], case)['input_ids'][0].tolist()
        rows = read_trace(base.with_suffix('.logits.jsonl'), prompt_ids, ids, 32000)
        timing = [float(value) for value in base.with_suffix('.timing').read_text().split()]
        assert len(timing) == 3 and all(math.isfinite(value) and value >= 0 for value in timing)
        text = base.with_suffix('.text').read_text()
        assert text == tokenizer.decode(ids, skip_special_tokens=True)
        outcomes[case['name']] = dict(token_ids=ids, text=text, prompt_tokens=len(prompt_ids),
                                     score_rows=len(rows), timing_ms=timing)
    for prefix in ('raw', 'chat', 'minimum'):
        before, after = outcomes[prefix + '-before'], outcomes[prefix + '-after']
        assert before['token_ids'] == after['token_ids'], f'{prefix}: stale request state'
        assert before['text'] == after['text']
    assert outcomes['raw-before']['token_ids'] == baseline['token_ids']
    assert outcomes['raw-before']['text'] == baseline['text']
    assert outcomes['raw-before']['text'], 'nonempty raw continuation remains required'
    assert outcomes['maximum']['prompt_tokens'] + 1 == manifest['max_sequence_length']
    report = {'requests': outcomes, 'consumer': _file_identity(consumer),
              'unadvertised': ['batch>1', 'external attention masks/padding',
                               'cancel/finish sessions', 'thinking mode']}
    (directory / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return report
