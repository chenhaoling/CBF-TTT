"""Recompute a completed readability summary and audit its source/row provenance."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import datetime
import io
import json
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace

from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_readability import summarize, validate


def audit(root):
    root = Path(root)
    design = json.loads((root / 'data/design.json').read_text())
    source = Path(design['source_root'])
    data = root / 'data/scenes.jsonl'
    scenes = [json.loads(x) for x in data.read_text().splitlines()]
    validate(scenes)
    assert file_digest(data) == design['data_sha256']
    assert file_digest(source / 'data/design.json') == design['source_design_sha256']
    assert file_digest(source / 'data/scenes.jsonl') == design['source_data_sha256']
    original = {r['id']: r for r in map(json.loads, (source / 'data/scenes.jsonl').read_text().splitlines())}
    for r in scenes:
        old = original[r['id']]
        assert old['split'] == 'dev' and old['regime'] == 'stable'
        assert r['sources'] == old['sources'] and r['group'] == old['group']
        if r['cell'] == 'bridge':
            assert r['chunks'] == old['chunks']
            assert r['queries'] == {k: {'qa': v} for k, v in old['queries'].items()}
    assert design['context_hashes'] == [
        {k: r[k] for k in ('id', 'cell', 'context_sha256', 'sources')} for r in scenes]
    inputs = [root / f'rows_{i}.jsonl' for i in (0, 1)]
    summary = json.loads((root / 'summary.json').read_text())
    # Recompute from raw outputs rather than trusting archived summary fields.
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        output = Path(tmp) / 'summary.json'
        summarize(SimpleNamespace(data=str(data), inputs=[str(p) for p in inputs],
            reference=[str(source / f'q_{i}.jsonl') for i in (0, 1)], output=str(output)))
        assert json.loads(output.read_text()) == summary
    profiles = {}
    for shard, path in enumerate(inputs):
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        a = json.loads(Path(str(path) + '.audit.json').read_text())
        assert a['weights_sha256'] == summary['weights_sha256']
        assert len(rows) == 64
        assert all((r['group'] // 2 + r['group'] % 2) % 2 == shard for r in rows)
        for r in rows:
            cell = profiles.setdefault(r['cell'], {'rollout_seconds': [], 'query_seconds': []})
            cell['rollout_seconds'].append(r['rollout']['seconds'])
            cell['query_seconds'].extend(q['seconds'] for fs in r['queries'].values() for q in fs.values())
    profiles = {cell: {kind: {'count': len(v), 'mean': statistics.fmean(v), 'max': max(v)}
        for kind, v in values.items()} for cell, values in profiles.items()}
    start = (root / 'started_at.txt').read_text().strip()
    end = (root / 'completed_at.txt').read_text().strip()
    assert not (root / 'failed_exit_code.txt').exists()
    assert not list(root.glob('confirm_*.jsonl'))
    assert not list(source.glob('confirm_*.jsonl'))
    assert summary['contexts'] == 128 and summary['queries'] == 384
    return {'audit_passed': True, 'summary_recomputed': True,
        'started_at': start, 'completed_at': end,
        'wall_seconds': (datetime.datetime.fromisoformat(end) - datetime.datetime.fromisoformat(start)).total_seconds(),
        'code_commit': (root / 'code_commit.txt').read_text().strip(),
        'audit_script_sha256': file_digest(Path(__file__)),
        'design_sha256': file_digest(root / 'data/design.json'),
        'data_sha256': design['data_sha256'], 'summary_sha256': file_digest(root / 'summary.json'),
        'rows_sha256': {p.name: file_digest(p) for p in inputs},
        'weights_sha256': summary['weights_sha256'],
        'groups': sorted({r['group'] for r in scenes}),
        'cells': dict(Counter(r['cell'] for r in scenes)),
        'contexts': 128, 'queries': 384, 'confirm_scored': False,
        'bridge_checks': summary['bridge_checks'], 'bridge_max_error': summary['bridge_max_error'],
        'profile': summary['profile'], 'timing_by_cell': profiles}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
