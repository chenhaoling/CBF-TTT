"""Independent reconstruction and completion audit for fastweight_content_v1."""
import argparse
import json
from pathlib import Path

from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_fastweight_content import (Q_GROUPS,V_GROUPS,SOURCE_DATA_SHA,SOURCE_DESIGN_SHA,
    load_data,source_backgrounds,summarize_q,summarize)
from tasks.cbf_readability import rebuild


def reconstruct(root,tokenizer_path):
    from transformers import AutoTokenizer
    root=Path(root);rows,design=load_data(root);source=Path(design['source_root'])
    source_design=json.loads((source/'data/design.json').read_text());original=list(map(json.loads,(source/'data/scenes.jsonl').read_text().splitlines()))
    selected=[r for r in original if r['group'] in Q_GROUPS+V_GROUPS and r['regime']=='stable']
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path);backgrounds=source_backgrounds(source_design,tokenizer,Q_GROUPS+V_GROUPS);expected=[]
    for old in selected:
        row=rebuild(old,backgrounds[old['group']],tokenizer,'short','single');row['queries']={'target':{'qa':row['queries']['target']['qa']}}
        row.update(twin_id=old['twin_id'],split='q' if old['group'] in Q_GROUPS else 'v');expected.append(row)
    if rows!=expected:raise ValueError('scene reconstruction differs')
    return rows,design


def audit(root,tokenizer):
    root=Path(root);rows,design=reconstruct(root,tokenizer);q_stored=json.loads((root/'q_summary.json').read_text());q_new=summarize_q(root)
    if q_stored!=q_new:raise ValueError('Q summary recomputation')
    source=Path(design['source_root'])
    if file_digest(source/'data/design.json')!=SOURCE_DESIGN_SHA or file_digest(source/'data/scenes.jsonl')!=SOURCE_DATA_SHA:raise ValueError('source hashes')
    files={};
    for stage in ('q',):
        for shard in (0,1):
            path=root/f'{stage}_{shard}.jsonl';files[path.name]=file_digest(path)
    if not q_stored['passed']:
        if any((root/f'v_{s}.jsonl').exists() for s in (0,1)) or (root/'summary.json').exists():raise ValueError('V scored after failed Q')
        status='stopped_by_content_Q';summary_sha=None;decision=None
    else:
        stored=json.loads((root/'summary.json').read_text());new=summarize(root)
        if stored!=new:raise ValueError('V summary recomputation')
        for shard in (0,1):
            path=root/f'v_{shard}.jsonl';files[path.name]=file_digest(path)
        status=stored['terminal_status'];summary_sha=file_digest(root/'summary.json');decision=stored['decision']
    return {'protocol':'fastweight_content_v1','audit_passed':True,'data_sha256':design['data_sha256'],
        'design_sha256':file_digest(root/'data/design.json'),'q_summary_sha256':file_digest(root/'q_summary.json'),
        'summary_sha256':summary_sha,'source_data_sha256':SOURCE_DATA_SHA,'source_design_sha256':SOURCE_DESIGN_SHA,
        'rows_sha256':files,'audit_script_sha256':file_digest(Path(__file__)),'terminal_status':status,'decision':decision,
        'reconstructed_contexts':len(rows),'q_v_source_disjoint':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.root,a.tokenizer);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

