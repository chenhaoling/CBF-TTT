"""Publish compact numeric evidence without raw text, tokens, or gradient tensors."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def summarize(paths):
    rows = [json.loads(Path(p).read_text()) for p in paths]
    if len(rows) != 2 or {r['length'] for r in rows} != {6144, 12288}:
        raise ValueError('expected both lengths exactly once')
    if len({r['model'] for r in rows}) != 1:
        raise ValueError('mixed checkpoints')
    output = {'protocol': 'optimizer_audit_v1', 'model': rows[0]['model'], 'lengths': {}}
    for path, row in zip(paths, rows):
        ms = row['measurements']
        if [m['step'] for m in ms] != [1, 2, 3] or not row['frozen_backbone_unchanged']:
            raise ValueError('incomplete or mutating run')
        if row['checkpoint_saved'] or row['reference_max_loss_error'] > 1e-5:
            raise ValueError('trajectory audit failed')
        if set(row['interventions']) != {'full', 'without_last_conv', 'only_last_conv', 'half_last_conv', 'before_second_update'}:
            raise ValueError('incomplete interventions')
        if abs(row['interventions']['full']-ms[2]['fused_loss']) > 1e-5:
            raise ValueError('intervention restore changed trajectory')
        params = ms[0]['displacements'].keys()
        if len(params) != 14 or any(m['displacements'].keys() != params for m in ms):
            raise ValueError('writer tensor mismatch')
        for m in ms:
            if not all(math.isfinite(v) for d in m['displacements'].values() for v in d.values()):
                raise ValueError('nonfinite displacement')
        output['lengths'][str(row['length'])] = {
            'raw_result_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'input_ids_sha256': row['input_ids_sha256'],
            'reference_max_loss_error': row['reference_max_loss_error'],
            'frozen_backbone_unchanged': row['frozen_backbone_unchanged'],
            'gradient_tripwire': row['gradient_tripwire'],
            'fused_losses': [m['fused_loss'] for m in ms],
            'max_loss_abs_difference': max(m['loss_abs_difference'] for m in ms),
            'head_gradient_relative_l2_by_step': [m['head_gradient']['relative_l2'] for m in ms],
            'writer_gradient_global_relative_l2_by_step': [m['writer_gradients']['global_relative_l2'] for m in ms],
            'writer_gradient_global_cosine_by_step': [m['writer_gradients']['global_cosine'] for m in ms],
            'max_layer_gradient_relative_l2': max(v['relative_l2'] for m in ms for v in m['writer_gradients']['layers'].values()),
            'relative_update_by_tensor_and_step': {n: [m['displacements'][n]['relative_update'] for m in ms] for n in params},
            'initial_norm_by_tensor': {n: ms[0]['displacements'][n]['before_norm'] for n in params},
            'absolute_update_by_tensor_and_step': {n: [m['displacements'][n]['update_norm'] for m in ms] for n in params},
            'interventions': row['interventions'],
            'seconds_by_step': [m['seconds'] for m in ms],
            'max_peak_allocated_gib': max(m['peak_allocated_gib'] for m in ms),
            'max_peak_reserved_gib': max(m['peak_reserved_gib'] for m in ms)}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = summarize(args.inputs)
    Path(args.output).write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
