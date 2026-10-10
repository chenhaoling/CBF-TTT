"""Read saved resource measurements only; no model calls or changed study metrics."""
import argparse
import json
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.cbf_fact_interference import ARMS,EXPOSURES
from tasks.train_cbf_event_writer import write_json


def profile(values):
    return {'count':len(values),'total_seconds':sum(v['seconds'] for v in values),
            'mean_seconds':statistics.fmean(v['seconds'] for v in values),
            'peak_allocated_gib':max(v['peak_allocated_gib'] for v in values),
            'peak_reserved_gib':max(v['peak_reserved_gib'] for v in values)}


def summarize(root):
    root=Path(root);arms={};hashes={}
    def read(path,lines=False):
        hashes[str(path.relative_to(root))]=sha(path)
        return list(map(json.loads,path.read_text().splitlines())) if lines else json.loads(path.read_text())
    for arm in ARMS:
        p=root/arm;queries=[];writes=[];contexts=[];trials=[]
        for exposure in EXPOSURES:
            for r in read(p/f'evaluation_{exposure}.jsonl',True):queries.extend(r['policies'].values())
            writes.extend(read(p/f'evaluation_{exposure}.summary.json')['write_profiles'].values())
            if exposure:
                for r in read(p/f'probes_{exposure}.jsonl',True):
                    contexts.append(r['profile']);trials.extend(r['single_trials']+[r['joint_trial']])
        arms[arm]={'training':profile(read(p/'steps.jsonl',True)),'evaluation_queries':profile(queries),
                   'evaluation_writes':profile(writes),'probe_contexts':profile(contexts),'probe_trials':profile(trials)}
    return {'new_model_calls':0,'arms':arms,'source_sha256':hashes,'script_sha256':sha(Path(__file__)),
            'note':'probe_contexts includes probe_trials: do not add both time totals; timings exclude model loading/hash work'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    a=p.parse_args();value=summarize(a.root);write_json(Path(a.root)/'resources.json',value);print(json.dumps(value,indent=2))
