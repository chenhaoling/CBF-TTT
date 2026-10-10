"""Post-hoc description of saved generations; no new scoring or gate changes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import data, STEPS


def summarize(root):
    root = Path(root)
    rows, manifest = data(root)
    answers = {r['id']: r['answer'] for r in rows}
    checkpoints = {}
    hashes = {}
    for step in STEPS:
        path = root / f'evaluation_{step}.jsonl'
        hashes[path.name] = sha(path)
        counts = {}
        for row in map(json.loads, path.read_text().splitlines()):
            for policy, value in row['policies'].items():
                c = counts.setdefault(policy, Counter())
                text = value['generated']
                answer = answers[row['id']]
                code = re.search(r'\bcode_[0-9]+\b', text)
                c['queries'] += 1
                c['strict_exact'] += int(text == answer)
                c['first_line_exact'] += int(text.split('\n')[0].strip() == answer)
                c['first_code_correct'] += int(code is not None and code.group() == answer)
                c['has_any_numeric_code'] += int(code is not None)
                c['bare_code_prefix'] += int(text == 'code_')
                c['token_budget_reached'] += int(len(value['generated_ids']) == 16)
        checkpoints[str(step)] = {k: dict(v) for k, v in counts.items()}
    return {'analysis': 'post-hoc, descriptive only; does not alter preregistered strict EM or memory gate',
            'new_model_calls': 0, 'test_scored': False, 'checkpoints': checkpoints,
            'data_sha256': manifest['data_sha256'], 'evaluation_sha256': hashes,
            'script_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
