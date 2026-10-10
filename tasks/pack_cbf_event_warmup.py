"""Pack causal single-block event warmup; train/dev only, no held-out test reads."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_answer(tokenizer, query, answer):
    prompt=query+'\nAnswer:'
    q=tokenizer.encode(prompt,add_special_tokens=False)
    joint=tokenizer.encode(prompt+' '+answer,add_special_tokens=False)
    if joint[:len(q)]!=q or len(joint)==len(q):
        raise ValueError('answer crosses tokenizer boundary')
    a=joint[len(q):]
    if tokenizer.decode(a).strip()!=answer:raise ValueError('answer token roundtrip')
    return q,a


def pack(source, output, tokenizer, tokenizer_path, chunk=4096):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError(output)
    if chunk<64:raise ValueError('chunk too short')
    manifest=json.loads((source/'manifest.json').read_text())
    packed=[];backgrounds={};contexts={}
    for split in ('train','dev'):
        path=source/f'{split}.warmup.jsonl'
        if sha(path)!=manifest['files_sha256'][path.name]:raise ValueError('source hash mismatch')
        rows=list(map(json.loads,path.read_text().splitlines()))
        if len(rows)!=(64 if split=='train' else 16):raise ValueError('frozen pilot counts changed')
        for r in rows:
            if r['split']!=split or len(r['write_blocks'])!=1 or r['decision_boundaries']:
                raise ValueError('not single-block warmup')
            gid=r['group_id'];twin=r['id'].split('.')[1];context_id=gid+'.'+twin
            if gid not in backgrounds:
                # Unique procedural background per world, shared by its twins. No natural corpus claims.
                text='\n'.join(f'Unrelated maintenance entry {gid}-{i}: sensor inspection completed at station {i+7000}; duration {i%37+1} minutes.' for i in range(512))
                backgrounds[gid]=tokenizer.encode(text,add_special_tokens=False)
            suffix=tokenizer.encode('\n\nActive session records:\n'+r['write_blocks'][0]+'\n',add_special_tokens=False)
            need=chunk-len(suffix)
            if need<0 or len(backgrounds[gid])<need:raise ValueError('facts do not fit')
            ids=backgrounds[gid][:need]+suffix
            q,a=encode_answer(tokenizer,r['read_query'],r['answer'])
            if context_id in contexts and contexts[context_id]!=ids:raise ValueError('context identity collision')
            contexts[context_id]=ids
            packed.append({'id':r['id'],'context_id':context_id,'group_id':gid,'split':split,
                           'context_ids':ids,'query_ids':q,'answer_ids':a,'answer':r['answer'],
                           'fact_suffix_tokens':len(suffix),'answer_start':len(q)-1})
    if {r['group_id'] for r in packed if r['split']=='train'} & {r['group_id'] for r in packed if r['split']=='dev'}:
        raise ValueError('group overlap')
    for r in packed:
        groups=sorted({x['group_id'] for x in packed if x['split']==r['split']})
        other=groups[(groups.index(r['group_id'])+1)%len(groups)]
        r['wrong_context_id']=other+'.'+r['context_id'].split('.')[1]
        r['twin_context_id']=r['group_id']+'.'+str(1-int(r['context_id'].split('.')[1]))
        r['anchor_query']='.asset_' in r['id']
    output.mkdir(parents=True);path=output/'rows.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in packed))
    result={'protocol':'event_writer_warmup_v1','source':str(source),'source_manifest_sha256':sha(source/'manifest.json'),
            'data_sha256':sha(path),'chunk':chunk,'rows':{'train':64,'dev':16},'contexts':{'train':16,'dev':4},
            'groups':{'train':8,'dev':2},'test_tokenized':False,'background':'unique procedural maintenance text; no natural sources',
            'tokenizer':tokenizer_path,'tokenizer_files_sha256':{p.name:sha(p) for p in Path(tokenizer_path).glob('*') if p.name in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')},
            'answer_tokens_range':[min(len(r['answer_ids']) for r in packed),max(len(r['answer_ids']) for r in packed)],
            'fact_suffix_tokens_range':[min(r['fact_suffix_tokens'] for r in packed),max(r['fact_suffix_tokens'] for r in packed)],
            'eos_token_id':tokenizer.eos_token_id}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','tokenizer'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    from transformers import AutoTokenizer
    print(json.dumps(pack(a.source,a.output,AutoTokenizer.from_pretrained(a.tokenizer),a.tokenizer),indent=2))
