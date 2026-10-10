"""Post-hoc description of already-scored first-token choices; no model forwards."""
import argparse
from collections import Counter
import json
from pathlib import Path

from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_checkpoint_native import STEPS


def summarize(root, tokenizer):
    from transformers import AutoTokenizer
    root=Path(root); summary=json.loads((root/'summary.json').read_text())
    audit=json.loads((root/'execution_audit.json').read_text())
    if not audit['audit_passed'] or audit['summary_sha256']!=file_digest(root/'summary.json'):
        raise ValueError('completion audit mismatch')
    tok=AutoTokenizer.from_pretrained(tokenizer); result={}
    for step in STEPS:
        for mode in ('plain','native'):
            arm=f'step{step}_{mode}'; rows=[]
            for shard in (0,1):
                p=root/f'{arm}_{shard}.jsonl'
                if file_digest(p)!=summary['rows_sha256'][p.name]:raise ValueError('rows changed')
                rows.extend(map(json.loads,p.read_text().splitlines()))
            counts=Counter(r['queries'][q]['binding']['greedy_id'] for r in rows for q in ('target','anchor'))
            if sum(counts.values())!=128:raise ValueError('incomplete query count')
            result[arm]={'queries':128,'top_tokens':[{'id':k,'count':v,'decoded':tok.decode([k])} for k,v in counts.most_common(10)]}
    result['_provenance']={'post_hoc_descriptive_only':True,'new_forwards':0,'tokenizer':tokenizer,
        'summary_sha256':file_digest(root/'summary.json'),'rows_sha256':summary['rows_sha256'],
        'script_sha256':file_digest(Path(__file__)),
        'tokenizer_files_sha256':{p.name:file_digest(p) for p in sorted(Path(tokenizer).glob('*'))
                                if p.name in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')}}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','tokenizer','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();Path(a.output).write_text(json.dumps(summarize(a.root,a.tokenizer),indent=2)+'\n')
