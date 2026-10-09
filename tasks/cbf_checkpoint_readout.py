"""Frozen M=0 checkpoint comparison, with independent native/repository path checks."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_cbf_memory_value import file_digest, REGIMES
from tasks.cbf_disjoint_readout import validate, aggregate, choose_format, IDENTITY, FORMATS

DATA_SHA='6740c33875d20cb42f53865daedff1dee71dda53dbcda590e13e6fda447cafe4'
ARMS=('original_repo','final_repo','original_native')
METRICS=('correct','nll','greedy_correct')


def is_adapter(name):
    parts=name.split('.')
    return (len(parts)==6 and parts[:2]==['model','layers'] and parts[2].isdigit()
            and parts[3]=='mlp' and parts[4] in ('ttt_conv','ttt_proj') and parts[5]=='weight')


def check_loading(info,expected):
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('backbone loading incomplete')
    if set(info.get('unexpected_keys',[]))!=set(expected) or any(not is_adapter(k) for k in expected):
        raise ValueError('unexpected checkpoint parameters')


def parameter_digest(model):
    import torch
    h=hashlib.sha256()
    for name,p in sorted(model.named_parameters()):
        h.update(name.encode());h.update(str((tuple(p.shape),str(p.dtype))).encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_plain(path,native=False):
    import torch
    from safetensors import safe_open
    if native:
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    else:
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
    root=Path(path);config=Qwen3Config.from_pretrained(path);config.ttt_mode=False
    files=sorted(root.glob('*.safetensors'));keys=set()
    if not files:raise ValueError('missing safetensors weights')
    for p in files:
        with safe_open(str(p),framework='pt',device='cpu') as f:keys.update(f.keys())
    adapters=sorted(k for k in keys if is_adapter(k))
    model,info=Qwen3ForCausalLM.from_pretrained(path,config=config,dtype=torch.bfloat16,
        attn_implementation='sdpa',output_loading_info=True)
    check_loading(info,adapters)
    if any('ttt_' in n for n,_ in model.named_parameters()):raise ValueError('active TTT parameters')
    model.to('cuda').eval().requires_grad_(False)
    metadata={'path':path,'class':type(model).__module__+'.'+type(model).__name__,
        'loading_info':info,'excluded_adapters':adapters,'config_sha256':file_digest(root/'config.json'),
        'config':config.to_dict(),'attention':model.config._attn_implementation,
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'source_files_sha256':{p.name:file_digest(p) for p in files}}
    return model,metadata


class PlainSession:
    """Read-only full KV; every query owns its copy, no fast-weight state exists."""
    def __init__(self,model,cache=None):
        from transformers.cache_utils import DynamicCache
        self.model=model;self.cache=cache if cache is not None else DynamicCache(config=model.config)
    @property
    def device(self):return next(self.model.parameters()).device
    def clone(self):return PlainSession(self.model,copy.deepcopy(self.cache))
    def _forward(self,ids,collect=False):
        import torch
        if collect:raise ValueError('M=0 comparison cannot collect updates')
        with torch.inference_mode():
            out=self.model.model(input_ids=torch.tensor([ids],device=self.device),past_key_values=self.cache,use_cache=True)
        self.cache=out.past_key_values
        return out.last_hidden_state


def cache_digest(cache):
    import torch
    h=hashlib.sha256(str(cache.get_seq_length()).encode())
    for layer in cache.layers:
        for name in ('keys','values'):
            t=getattr(layer,name,None)
            if t is not None:h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def collect(args, data_path=None, validator=validate, identity_fields=IDENTITY, expected_data_sha=DATA_SHA):
    import torch
    import transformers
    from tasks.cbf_selective import measure
    from tasks.cbf_memory_value import score_query
    data=Path(data_path) if data_path is not None else Path(args.source)/'data/scenes.jsonl'
    if file_digest(data)!=expected_data_sha:raise ValueError('frozen data changed')
    scenes=list(map(json.loads,data.read_text().splitlines()));validator(scenes)
    selected=[r for r in scenes if (r['group']//2+r['group']%2)%2==args.shard
              and (args.arm!='original_native' or r['is_bridge'])]
    selected.sort(key=lambda r:(not r['is_bridge'],r['group'],r['id'],r['cell']))
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    torch.manual_seed(211);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter();model,meta=load_plain(args.model,args.arm=='original_native')
    if bool(meta['excluded_adapters'])!=(args.arm=='final_repo'):raise ValueError('wrong checkpoint arm')
    before=parameter_digest(model);versions=[p._version for p in model.parameters()]
    with out.open('x') as sink:
        for i,r in enumerate(selected):
            torch.cuda.empty_cache()
            def prefix():
                session=PlainSession(model)
                for chunk in r['chunks']:session._forward(chunk)
                return session
            session,profile=measure(prefix);parent=cache_digest(session.cache);queries={}
            for name,fs in r['queries'].items():
                queries[name]={}
                for fmt,q in fs.items():
                    # score_query injects an empty cbf_memory attribute only into the
                    # cloned standard cache; plain decoder never reads or writes it.
                    v,p=measure(lambda:score_query(session,q,r['choice_ids'],'full_kv',{}))
                    queries[name][fmt]={**v,**p}
            if parent!=cache_digest(session.cache):raise RuntimeError('query changed prefix')
            row={k:r[k] for k in identity_fields}
            row.update(arm=args.arm,model=args.model,weights_sha256=before,data_sha256=expected_data_sha,
                       parent_unchanged=True,queries=queries,rollout=profile)
            sink.write(json.dumps(row)+'\n');sink.flush()
            print(json.dumps({'arm':args.arm,'completed':i+1,'total':len(selected)}),flush=True)
            del session
    if before!=parameter_digest(model) or versions!=[p._version for p in model.parameters()]:raise RuntimeError('weights changed')
    meta.update(arm=args.arm,shard=args.shard,rows=len(selected),rows_sha256=file_digest(out),
        weights_sha256=before,backbone_unchanged=True,seconds=time.perf_counter()-start,
        torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name())
    Path(str(out)+'.audit.json').write_text(json.dumps(meta,indent=2)+'\n')


def compare_scores(rows,refs):
    index={(r['id'],r['cell']):r for r in refs};errors=[]
    for r in rows:
        old=index[(r['id'],r['cell'])]
        if r['context_sha256']!=old['context_sha256'] or r['data_sha256']!=old['data_sha256']:
            raise ValueError('path check inputs differ')
        for name,fs in r['queries'].items():
            for fmt,v in fs.items():
                b=old['queries'][name][fmt]
                if any(v[k]!=b[k] for k in ('prediction','greedy_id')):raise ValueError('path predictions differ')
                errors.append(max(abs(x-y) for x,y in zip(v['choice_nll'],b['choice_nll'])))
    if not errors or max(errors)>1e-5:raise ValueError('path NLL differs')
    return {'queries':len(errors),'max_choice_nll_error':max(errors),'predictions_identical':True}


def validate_scores(rows,scenes,data_sha):
    index={(r['id'],r['cell']):r for r in scenes}
    if len(rows)!=len(index) or {(r['id'],r['cell']) for r in rows}!=set(index):raise ValueError('missing/duplicate contexts')
    for r in rows:
        s=index[r['id'],r['cell']]
        if r['data_sha256']!=data_sha or not r['parent_unchanged'] or any(r[k]!=s[k] for k in IDENTITY):raise ValueError('row provenance')
        if set(r['queries'])!=set(s['queries']):raise ValueError('query matrix')
        for name,fs in r['queries'].items():
            if set(fs)!=set(s['queries'][name]):raise ValueError('format matrix')
            for fmt,v in fs.items():
                q=s['queries'][name][fmt];scores=v['choice_nll']
                if len(scores)!=8 or not all(math.isfinite(x) for x in scores+[v['nll']]):raise ValueError('invalid scores')
                pred=min(range(8),key=lambda i:scores[i])
                if abs(v['nll']-scores[q['label']])>1e-6 or v['prediction']!=pred or v['correct']!=int(pred==q['label']) or v['greedy_correct']!=int(v['greedy_id']==q['answer_id']):raise ValueError('score mismatch')
        profiles=[r['rollout']]+[v for fs in r['queries'].values() for v in fs.values()]
        if any(not math.isfinite(p[k]) or p[k]<=0 for p in profiles for k in ('seconds','peak_allocated_gib','peak_reserved_gib')):raise ValueError('invalid profile')


def summarize(root,source):
    root=Path(root);source=Path(source);data=source/'data/scenes.jsonl'
    if file_digest(data)!=DATA_SHA:raise ValueError('frozen data changed')
    scenes=list(map(json.loads,data.read_text().splitlines()));validate(scenes)
    arms={};audits={}
    for arm in ARMS:
        rows=[];audits[arm]=[]
        for shard in (0,1):
            p=root/f'{arm}_{shard}.jsonl';part=list(map(json.loads,p.read_text().splitlines()))
            a=json.loads(Path(str(p)+'.audit.json').read_text())
            if a['rows_sha256']!=file_digest(p) or a['rows']!=len(part) or not a['backbone_unchanged']:raise ValueError('shard audit')
            if a['arm']!=arm or a['shard']!=shard:raise ValueError('arm audit')
            if any((r['group']//2+r['group']%2)%2!=shard or r['arm']!=arm or r['weights_sha256']!=a['weights_sha256'] or r['model']!=a['path'] for r in part):raise ValueError('arm provenance')
            check_loading(a['loading_info'],a['excluded_adapters'])
            if bool(a['excluded_adapters'])!=(arm=='final_repo'):raise ValueError('arm checkpoint')
            rows+=part;audits[arm].append(a)
        if any(len({json.dumps(a[k],sort_keys=True) for a in audits[arm]})!=1 for k in ('weights_sha256','config','source_files_sha256','class')):raise ValueError('mixed model')
        validate_scores(rows,[s for s in scenes if arm!='original_native' or s['is_bridge']],DATA_SHA);arms[arm]=rows
    if audits['original_repo'][0]['weights_sha256']!=audits['original_native'][0]['weights_sha256']:raise ValueError('native weights differ')
    config_keys=('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','vocab_size','rope_theta','rope_scaling','rms_norm_eps','tie_word_embeddings','max_position_embeddings','sliding_window','attention_bias','hidden_act')
    if any(audits['original_repo'][0]['config'][k]!=audits['final_repo'][0]['config'][k] for k in config_keys):raise ValueError('architecture differs')
    refs=[]
    for shard in (0,1):
        p=source/f'rows_{shard}.jsonl';a=json.loads(Path(str(p)+'.audit.json').read_text())
        if a['rows_sha256']!=file_digest(p) or not a['backbone_unchanged']:raise ValueError('prior reference audit')
        refs+=list(map(json.loads,p.read_text().splitlines()))
    validate_scores(refs,scenes,DATA_SHA)
    if {r['model'] for r in refs}!={r['model'] for r in arms['final_repo']}:raise ValueError('reference checkpoint path')
    bridges={'final_vs_previous_cbf':compare_scores(arms['final_repo'],refs),
             'original_native_vs_repo':compare_scores(arms['original_native'],arms['original_repo'])}
    models={}
    for arm in ('original_repo','final_repo'):
        rows=[r for r in arms[arm] if not r['is_bridge']]
        domains={d:aggregate([r for r in rows if r['domain']==d]) for d in ('fineweb','longcrawl')}
        profiles=[p for r in arms[arm] for p in [r['rollout']]+[v for fs in r['queries'].values() for v in fs.values()]]
        models[arm]={'means':aggregate(rows),'domain_means':domains,
            'group_means':{str(g):aggregate([r for r in rows if r['group']==g]) for g in range(8)},
            'selection':choose_format(domains),'profile':{k:max(p[k] for p in profiles) for k in ('peak_allocated_gib','peak_reserved_gib')},
            'weights_sha256':audits[arm][0]['weights_sha256']}
    differences={g:{r:{q:{f:{k:models['final_repo']['group_means'][g][r][q][f][k]-models['original_repo']['group_means'][g][r][q][f][k]
        for k in METRICS} for f in FORMATS} for q in ('target','anchor')} for r in REGIMES} for g in map(str,range(8))}
    return {'protocol':'checkpoint_readout_v1','data_sha256':DATA_SHA,'contexts':176,'queries':608,
        'confirm_scored':False,'M':0,'bridges':bridges,'models':models,'final_minus_original_group_differences':differences,
        'loading_audits':audits,'prior_summary_sha256':file_digest(source/'summary.json'),
        'terminal_status':'completed_checkpoint_diagnostic','scope':'Observed dev, no training or V/F; classify readability only.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect');c.add_argument('--arm',choices=ARMS,required=True);c.add_argument('--shard',type=int,choices=(0,1),required=True)
    c.add_argument('--model',required=True);c.add_argument('--output',required=True);c.add_argument('--source',required=True)
    s=sub.add_parser('summarize');s.add_argument('--root',required=True);s.add_argument('--source',required=True);s.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='collect':collect(args)
    else:
        result=summarize(args.root,args.source);Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'status':result['terminal_status'],'bridges':result['bridges'],'gates':{k:v['selection'] for k,v in result['models'].items()}}))


if __name__=='__main__':main()
