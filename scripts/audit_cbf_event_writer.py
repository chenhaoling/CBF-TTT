"""Verify the completed 100-step writer run without additional model scoring."""
import argparse
import datetime
import json
import math
from pathlib import Path
import random

from tasks.pack_cbf_event_warmup import sha,encode_answer
from tasks.train_cbf_event_writer import data,summarize,STEPS


def audit(root,model_path):
    import torch
    from transformers import AutoTokenizer
    root=Path(root);rows,m=data(root);stored=json.loads((root/'summary.json').read_text())
    if summarize(root,save=False)!=stored:raise ValueError('summary recomputation changed')
    tok=AutoTokenizer.from_pretrained(model_path)
    source=Path(m['source']);source_manifest=json.loads((source/'manifest.json').read_text())
    if sha(source/'manifest.json')!=m['source_manifest_sha256']:raise ValueError('source changed')
    source_rows={}
    for split in ('train','dev'):
        p=source/f'{split}.warmup.jsonl'
        if sha(p)!=source_manifest['files_sha256'][p.name]:raise ValueError('warmup source changed')
        source_rows.update((r['id'],r) for r in map(json.loads,p.read_text().splitlines()))
    for r in rows:
        old=source_rows[r['id']]
        suffix=tok.encode('\n\nActive session records:\n'+old['write_blocks'][0]+'\n',add_special_tokens=False)
        q,a=encode_answer(tok,old['read_query'],old['answer'])
        if r['context_ids'][-len(suffix):]!=suffix or r['query_ids']!=q or r['answer_ids']!=a or r['answer']!=old['answer']:
            raise ValueError('facts/query/answer packing mismatch')
    records=list(map(json.loads,(root/'steps.jsonl').read_text().splitlines()))
    training=[r for r in rows if r['split']=='train'];byid={r['id']:r for r in rows}
    rng=random.Random(301);order=[]
    if len(records)!=100:raise ValueError('training step count')
    for step,rec in enumerate(records,1):
        if not order:order=list(range(len(training)));rng.shuffle(order)
        r=training[order.pop()]
        if rec['step']!=step or rec['id']!=r['id']:raise ValueError('training order/leakage')
        for key in ('loss','answer_nll','eos_nll','seconds','peak_allocated_gib','peak_reserved_gib'):
            if not math.isfinite(rec[key]) or rec[key]<=0:raise ValueError('invalid training profile')
        n=len(r['answer_ids'])
        if abs(rec['loss']*(n+1)-rec['answer_nll']*n-rec['eos_nll'])>1e-4:raise ValueError('loss mask inconsistency')
    if sha(root/'steps.jsonl')!=stored['train']['steps_sha256']:raise ValueError('steps audit changed')
    initial=torch.load(root/'writer_0.pt',map_location='cpu',weights_only=True)
    final=torch.load(root/'writer_100.pt',map_location='cpu',weights_only=True)
    for n,p in final['writer'].items():
        if p.dtype!=torch.float32 or not torch.isfinite(p).all():raise ValueError('writer dtype/numerics')
        if '.ttt_conv.' in n and torch.count_nonzero(initial['writer'][n])!=0:raise ValueError('conv initialization')
        change=float((p-initial['writer'][n]).norm())
        if change<=0 or abs(change-stored['train']['writer_update_norms'][n])>1e-7:raise ValueError('saved writer update mismatch')
    optimizer=torch.load(root/'optimizer_100.pt',map_location='cpu',weights_only=True)
    if any(v.dtype!=torch.float32 for st in optimizer['state'].values() for v in st.values() if isinstance(v,torch.Tensor)):
        raise ValueError('saved optimizer dtype')
    if optimizer['param_groups'][0]['lr']!=1e-7 or optimizer['param_groups'][0]['weight_decay']!=0:raise ValueError('optimizer config')
    query_count=0
    for step in STEPS:
        evaluations=list(map(json.loads,(root/f'evaluation_{step}.jsonl').read_text().splitlines()))
        for r in evaluations:
            source_row=byid[r['id']]
            expected={'correct','empty','wrong','full_kv'}|({'twin'} if source_row['anchor_query'] else set())
            if set(r['policies'])!=expected or source_row['split']!='dev':raise ValueError('evaluation policy/split')
            for p,v in r['policies'].items():
                query_count+=1
                if tok.decode(v['generated_ids'],skip_special_tokens=False).strip()!=v['generated']:raise ValueError('generation text mismatch')
                if v['exact_match']!=int(v['generated']==source_row['answer']):raise ValueError('EM mismatch')
                if len(v['generated_ids'])>16:raise ValueError('generation budget exceeded')
                for key in ('answer_nll','terminal_nll','seconds','peak_allocated_gib','peak_reserved_gib'):
                    if not math.isfinite(v[key]) or v[key]<=0:raise ValueError('invalid evaluation score/profile')
    if query_count!=272:raise ValueError('evaluation count')
    for name,h in final['manifest']['source_files_sha256'].items():
        if sha(Path(model_path)/name)!=h:raise ValueError('original model source changed')
    if (root/'failed_exit_code.txt').exists() or (root/'train_failed.txt').exists():raise ValueError('failed run')
    start=(root/'started_at.txt').read_text().strip();end=(root/'completed_at.txt').read_text().strip()
    return {'audit_passed':True,'summary_recomputed':True,'steps':100,'dev_policy_queries':272,
            'train_probe_queries':16,'test_scored':False,'packing_answers_verified':80,
            'optimizer_and_writer_fp32_verified':True,'original_source_files_rehashed':True,
            'started_at':start,'completed_at':end,
            'wall_seconds':(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds(),
            'code_commit':(root/'code_commit.txt').read_text().strip(),
            'summary_sha256':sha(root/'summary.json'),'data_sha256':m['data_sha256'],
            'packed_manifest_sha256':sha(root/'data/manifest.json'),'audit_script_sha256':sha(Path(__file__)),
            'checkpoint_sha256':{str(s):sha(root/f'writer_{s}.pt') for s in STEPS},
            'passed_memory_gate':stored['passed_memory_gate']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--model',default='/home/ctj/models/Qwen3-4B');p.add_argument('--output',required=True)
    a=p.parse_args();v=audit(a.root,a.model);Path(a.output).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
