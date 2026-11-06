# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured sampler scores and typed graphs; never replace the E2E oracle."""

from contextlib import contextmanager
import gc
import json
from pathlib import Path
from unittest.mock import patch


def requested_profile(manifest):
    return (manifest['hf_id'] == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
            and manifest['hf_revision'] == 'fe8a4ea1ffedaf415f4da2f062534de366a451e6'
            and manifest['precision'] == 'fp16')


@contextmanager
def capture_typed_graphs(destination):
    """Observe the actual build networks without changing any layer or option."""
    import tensorrt as trt

    original = trt.Builder
    graphs = []

    def tensor(t):
        return None if t is None else {'name': t.name, 'shape': list(t.shape), 'dtype': str(t.dtype)}

    class ObservedBuilder:
        def __init__(self, *args, **kwargs):
            self.builder = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.builder, name)

        def build_serialized_network(self, network, config):
            graph = {
                'strongly_typed': network.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED),
                'inputs': [tensor(network.get_input(i)) for i in range(network.num_inputs)],
                'outputs': [tensor(network.get_output(i)) for i in range(network.num_outputs)],
                'layers': [],
            }
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                graph['layers'].append({
                    'index': i, 'name': layer.name, 'type': str(layer.type),
                    'inputs': [tensor(layer.get_input(j)) for j in range(layer.num_inputs)],
                    'outputs': [tensor(layer.get_output(j)) for j in range(layer.num_outputs)],
                })
            graphs.append(graph)
            Path(destination).write_text(json.dumps({'graphs': graphs}, indent=2) + '\n')
            assert graph['strongly_typed']
            assert any(t['dtype'] == str(trt.float16) and t['name'].startswith('cache_k_')
                       for t in graph['inputs'])
            assert any(t['name'] == 'logits' and t['dtype'] == str(trt.float32)
                       for t in graph['outputs'])
            return self.builder.build_serialized_network(network, config)

    with patch.object(trt, 'Builder', ObservedBuilder):
        yield


def read_trace(path, prompt_ids, generated_ids, vocab_size):
    import numpy as np

    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert len(rows) == len(generated_ids), 'missing sampler score rows'
    for i, row in enumerate(rows):
        assert row['prefix_token_ids'] == prompt_ids + generated_ids[:i], 'prefix mismatch'
        scores = np.asarray(row['logits'], dtype=np.float32)
        assert scores.shape == (vocab_size,), 'incomplete vocabulary'
        assert np.isfinite(scores).all(), 'nonfinite sampler scores'
    return rows


def score_metrics(left, right):
    import numpy as np

    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    assert left.shape == right.shape and left.ndim == 1 and left.size > 1
    assert np.isfinite(left).all() and np.isfinite(right).all()
    delta = left - right

    def top(scores):
        indices = np.argsort(-scores, kind='stable')[:2]
        return {'token_ids': indices.tolist(), 'scores': scores[indices].tolist(),
                'margin': float(scores[indices[0]] - scores[indices[1]])}

    return {'max_abs': float(np.abs(delta).max()), 'rms': float(np.sqrt(np.mean(delta ** 2))),
            'left_top2': top(left), 'right_top2': top(right),
            'argmax_equal': bool(left.argmax() == right.argmax())}


def capture_official_scores(model, inputs, generated_ids):
    """Prefill once, then teacher-force recorded tokens with the official cache."""
    import numpy as np
    import torch

    prompt_length = inputs['input_ids'].shape[1]
    next_inputs = {key: value.to(model.device) for key, value in inputs.items()}
    captured = []
    with torch.inference_mode():
        for i, token_id in enumerate(generated_ids):
            output = model(**next_inputs, use_cache=True)
            captured.append(output.logits[0, -1].float().cpu().numpy())
            # Feed the recorded native token, never the reference argmax.
            next_inputs = {
                'input_ids': torch.tensor([[token_id]], device=model.device),
                'attention_mask': torch.ones((1, prompt_length + i + 1),
                                             dtype=torch.long, device=model.device),
                'past_key_values': output.past_key_values,
            }
    return np.stack(captured)


def compare_captured_scores(model_dir, manifest, case, generated_ids, destination, *, device='cuda'):
    """Force the official models through every prefix actually consumed natively.

    Report all-vocabulary residuals separately for native/FP16, FP16/FP32,
    and native/FP32. No numeric pass threshold is invented here. The unchanged
    full continuation oracle in test_e2e remains authoritative.
    """
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama import modeling_llama
    from .test_e2e import _file_identity, _render_prompt, _text_threshold, _thresholds

    destination = Path(destination)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    inputs = _render_prompt(tokenizer, case['prompt'], case)
    prompt_ids = inputs['input_ids'][0].tolist()
    config = json.loads((Path(model_dir) / 'config.json').read_text())
    rows = read_trace(destination / 'native-logits.jsonl', prompt_ids, generated_ids, config['vocab_size'])
    precisions = [('fp16', torch.float16), ('fp32', torch.float32)]
    assert manifest['precision'] == 'fp16' and case['reference_precision'] == 'fp32'
    report = {
        'schema': 1, 'diagnostic_only': True, 'qualification': 'not_established',
        'model': manifest['hf_id'], 'revision': manifest['hf_revision'], 'case': case,
        'batch_size': 1, 'tensor_parallel_size': 1, 'device': device,
        'seed': case.get('seed', -1), 'sampling': False,
        'transformers': transformers.__version__, 'torch': torch.__version__,
        'reference_source': _file_identity(Path(modeling_llama.__file__)),
        'family_sources': [_file_identity(Path(__file__).parents[1] / name) for name in [
            'model.py', 'checkpoint_mapper.py', 'standard_decoder_builder.py',
            'dual_profile_decoder_builder.py', 'graph_ops.py', 'graph_blocks.py',
            'runtime/pipeline.cpp', 'runtime/logits_trace.h',
            'tests/test_e2e.py', 'tests/precision_diagnostics.py']],
        'native_trace': _file_identity(destination / 'native-logits.jsonl'),
        'case_inputs': _file_identity(destination / 'case-inputs.json'),
        'typed_graphs': _file_identity(destination / 'typed-graphs.json'),
        'contract_ned_threshold': _text_threshold(case, _thresholds(case['name'])),
        'initial_state': 'empty KV cache; positions start at zero; all-one attention mask',
        'attention_mask': inputs['attention_mask'].tolist(),
        'position_ids': list(range(len(prompt_ids))),
        'prompt_token_ids': prompt_ids, 'generated_token_ids': generated_ids,
        'references': {}, 'prefixes': [],
    }
    scores = {'native': np.asarray([row['logits'] for row in rows], dtype=np.float32)}
    np.save(destination / 'native-logits.npy', scores['native'], allow_pickle=False)
    for name, dtype in precisions:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False, dtype=dtype).eval().to(device)
        report['references'][name] = {'dtype': str(model.dtype),
                                    'attention_implementation': model.config._attn_implementation}
        scores[name] = capture_official_scores(model, inputs, generated_ids)
        np.save(destination / f'official-{name}-logits.npy', scores[name], allow_pickle=False)
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    for i, row in enumerate(rows):
        report['prefixes'].append({
            'prefix_token_ids': row['prefix_token_ids'],
            **{f'{left}_vs_{right}': score_metrics(scores[left][i], scores[right][i])
               for left, right in [('native', 'fp16'), ('fp16', 'fp32'), ('native', 'fp32')]},
        })
    report['score_files'] = [_file_identity(destination / f'{name}-logits.npy')
                             for name in ['native', 'official-fp16', 'official-fp32']]
    (destination / 'precision-diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
