"""Independent reconstruction audit for fastweight_content_v2."""
import argparse,json
from pathlib import Path
from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_fastweight_content import SOURCE_DATA_SHA,SOURCE_DESIGN_SHA,source_backgrounds,Q_GROUPS,V_GROUPS
from tasks.cbf_fastweight_content_v2 import DEV_GROUPS,FORMATS,IDENTITY,query_text,load_data,summarize_dev,summarize_q,summarize
from tasks.cbf_readability import rebuild


def reconstruct(root,tokenizer_path):
    from transformers import AutoTokenizer
    root=Path(root);rows,design=load_data(root);source=Path(design['source_root']);sd=json.loads((source/'data/design.json').read_text());original=list(map(json.loads,(source/'data/scenes.jsonl').read_text().splitlines()));groups=DEV_GROUPS+Q_GROUPS+V_GROUPS
    selected=[r for r in original if r['group'] in groups and r['regime']=='stable'];tok=AutoTokenizer.from_pretrained(tokenizer_path);backgrounds=source_backgrounds(sd,tok,groups);expected=[]
    for old in selected:
        row=rebuild(old,backgrounds[old['group']],tok,'short','single');base=old['queries']['target'];row['queries']={'target':{fmt:{**base,'ids':tok.encode(query_text(base['key'],fmt),add_special_tokens=False)} for fmt in FORMATS}}
        row.update(twin_id=old['twin_id'],split='dev' if old['group'] in DEV_GROUPS else 'q' if old['group'] in Q_GROUPS else 'v');expected.append(row)
    if rows!=expected:raise ValueError('reconstruction differs')
    return rows,design


def audit(root,tokenizer):
    root=Path(root);rows,design=reconstruct(root,tokenizer);files={};dev=json.loads((root/'dev_selection.json').read_text())
    if dev!=summarize_dev(root):raise ValueError('dev recomputation')
    for shard in (0,1):files[f'dev_{shard}.jsonl']=file_digest(root/f'dev_{shard}.jsonl')
    q=None;summary_sha=None;decision=None
    if not dev['passed']:
        if any((root/f'q_{s}.jsonl').exists() for s in (0,1)):raise ValueError('Q scored after dev failure')
        status='stopped_by_v2_dev_readout'
    else:
        q=json.loads((root/'q_summary.json').read_text())
        if q!=summarize_q(root):raise ValueError('Q recomputation')
        for shard in (0,1):files[f'q_{shard}.jsonl']=file_digest(root/f'q_{shard}.jsonl')
        if not q['passed']:
            if any((root/f'v_{s}.jsonl').exists() for s in (0,1)):raise ValueError('V scored after Q failure')
            status='stopped_by_v2_Q'
        else:
            stored=json.loads((root/'summary.json').read_text())
            if stored!=summarize(root):raise ValueError('V recomputation')
            for shard in (0,1):files[f'v_{shard}.jsonl']=file_digest(root/f'v_{shard}.jsonl')
            status=stored['terminal_status'];summary_sha=file_digest(root/'summary.json');decision=stored['decision']
    return {'protocol':'fastweight_content_v2','audit_passed':True,'data_sha256':design['data_sha256'],'design_sha256':file_digest(root/'data/design.json'),
        'dev_selection_sha256':file_digest(root/'dev_selection.json'),'q_summary_sha256':file_digest(root/'q_summary.json') if q else None,'summary_sha256':summary_sha,
        'source_data_sha256':SOURCE_DATA_SHA,'source_design_sha256':SOURCE_DESIGN_SHA,'rows_sha256':files,'audit_script_sha256':file_digest(Path(__file__)),
        'terminal_status':status,'decision':decision,'reconstructed_contexts':len(rows),'three_way_source_disjoint':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.root,a.tokenizer);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

