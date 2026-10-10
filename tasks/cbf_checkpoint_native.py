"""Fixed 12K development diagnostic: checkpoint trajectory and actual native TTT."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_cbf_memory_value import file_digest, REGIMES
from tasks.cbf_checkpoint_readout import (PlainSession, cache_digest, load_plain, parameter_digest,
                                         check_loading, validate_scores)
from tasks.cbf_disjoint_readout import IDENTITY, aggregate, choose_format
from tasks.cbf_length_readout import load_data as load_length_data
from tasks.cbf_record_interference import row_hash

SOURCE_SHA = 'dfaab6b69f23468d99881fa2a620f3ae4a79de60146caf4521edc6ac1e04aa22'
SUMMARY_SHA = '0bc121d5d92e1314b8c364849b88e18a302f1f19eaa01e7a725666c8e2e21b60'
STEPS = (10000, 40000, 81381)
ARMS = ('original_plain',) + tuple(f'step{s}_{m}' for s in STEPS for m in ('plain', 'native'))
ALL_ARMS = ARMS + ('final_zero_lr',)
LAYERS = (0, 6, 12, 18, 24, 30, 35)
FIELDS = IDENTITY + ('source_context_sha256',)


def reblock(rows):
    result = []
    for old in rows:
        if old['segment_tokens'] != 2048:
            continue
        r = copy.deepcopy(old)
        if len(r['chunks']) != 6 or any(len(c) != 2048 for c in r['chunks']):
            raise ValueError('expected six 2048 segments')
        r['chunks'] = [r['chunks'][i]+r['chunks'][i+1] for i in (0, 2, 4)]
        for i, rec in enumerate(r['records']):
            if rec['chunk'] != i+1:
                raise ValueError('expected one-based record chunk index')
            rec['offset'] += (i % 2)*2048
            rec['chunk'] = i//2+1
        r.update(source_context_sha256=old['context_sha256'], context_sha256=row_hash(r['chunks']), chunk_size=4096)
        # cell/id remain the source identifiers; segmentation is explicit in the manifest.
        if [x for c in r['chunks'] for x in c] != [x for c in old['chunks'] for x in c]:
            raise ValueError('token stream changed')
        if any(len(q['ids']) >= 4096 for fs in r['queries'].values() for q in fs.values()):
            raise ValueError('query crosses a native update boundary')
        result.append(r)
    if len(result) != 64 or len({(r['id'],r['cell']) for r in result}) != 64:
        raise ValueError('incomplete source matrix')
    return result


def build(source, root):
    source, root = Path(source), Path(root)
    rows, d = load_length_data(source)
    if d['data_sha256'] != SOURCE_SHA or file_digest(source/'summary.json') != SUMMARY_SHA:
        raise ValueError('wrong frozen length study')
    audit = json.loads((source/'execution_audit.json').read_text())
    if not audit['audit_passed'] or audit['summary_sha256'] != SUMMARY_SHA:
        raise ValueError('unaudited source')
    scenes = reblock(rows)
    data = root/'data'; data.mkdir()
    (data/'scenes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scenes))
    manifest = {'protocol': 'checkpoint_native_v1', 'source_root': str(source),
                'source_data_sha256': SOURCE_SHA, 'source_summary_sha256': SUMMARY_SHA,
                'data_sha256': file_digest(data/'scenes.jsonl'), 'arms': list(ALL_ARMS),
                'steps': list(STEPS), 'primary_format': 'binding', 'auxiliary_format': 'qa',
                'context_tokens': 12288, 'forward_chunk': 4096, 'source_segment_tokens': 2048,
                'contexts': 464, 'queries': 1856, 'zero_lr_groups': [0,2], 'confirm_scored': False,
                'context_hashes': [{k:r[k] for k in FIELDS} for r in scenes]}
    (data/'design.json').write_text(json.dumps(manifest, indent=2)+'\n')


def load_data(root):
    root = Path(root); d = json.loads((root/'data/design.json').read_text())
    source = Path(d['source_root']); old, old_d = load_length_data(source)
    if old_d['data_sha256'] != SOURCE_SHA or file_digest(source/'summary.json') != SUMMARY_SHA:
        raise ValueError('source changed')
    rows = list(map(json.loads, (root/'data/scenes.jsonl').read_text().splitlines()))
    if rows != reblock(old) or file_digest(root/'data/scenes.jsonl') != d['data_sha256']:
        raise ValueError('derived data changed')
    if d['arms'] != list(ALL_ARMS) or d['primary_format'] != 'binding' or d['contexts'] != 464:
        raise ValueError('matrix changed')
    if d['context_hashes'] != [{k:r[k] for k in FIELDS} for r in rows]:
        raise ValueError('context manifest changed')
    return rows, d


class NativeSession(PlainSession):
    """Repository native path, with partial-chunk buffers and full adapted weights."""
    def __init__(self, model, cache=None):
        from inference_model.hf_qwen3.modeling_qwen3 import TTTDynamicCache
        self.model = model
        self.cache = cache if cache is not None else TTTDynamicCache(config=model.config)
        if getattr(self.cache, 'cbf_enabled', False):
            raise ValueError('CBF path is not native TTT')

    def clone(self):
        return NativeSession(self.model, copy.deepcopy(self.cache))

    def _forward(self, ids, collect=False):
        before = self.cache.get_seq_length(); chunk = self.model.config.ttt_chunk
        weights = {i:self.cache.ttt_states[i][2] for i in self.model.config.ttt_layers}
        hidden = super()._forward(ids, collect)
        after = self.cache.get_seq_length()
        if after != before+len(ids):
            raise RuntimeError('cache length mismatch')
        for i, w in weights.items():
            h, t, new_w = self.cache.ttt_states[i]
            remainder = after % chunk
            if new_w is None or (h is None) != (remainder == 0) or (t is None) != (remainder == 0):
                raise RuntimeError('native state missing')
            if remainder and (h.shape[1] != remainder or t.shape[1] != remainder):
                raise RuntimeError('native tail length')
            if before//chunk == after//chunk and w is not None and new_w is not w:
                raise RuntimeError('fast weight changed without complete chunk')
        return hidden


def native_digest(cache):
    import torch
    h = hashlib.sha256(cache_digest(cache).encode())
    for i, state in enumerate(getattr(cache, 'ttt_states', [])):
        h.update(str(i).encode())
        for value in state:
            h.update(b'none' if value is None else value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def native_stats(session):
    import torch
    values = {}
    for i in session.model.config.ttt_layers:
        w = session.cache.ttt_states[i][2]
        base = session.model.model.layers[i].mlp.down_proj.weight
        if w is None or not torch.isfinite(w).all():
            raise RuntimeError('nonfinite native fast weight')
        relative = float((w.float()-base.float()).norm()/base.float().norm().clamp_min(1e-12))
        values[str(i)] = relative
    return values


def load_model(path, arm):
    if arm.endswith('_plain'):
        model, meta = load_plain(path)
        if bool(meta['excluded_adapters']) != (arm != 'original_plain'):
            raise ValueError('wrong plain checkpoint')
        return model, meta
    import torch
    from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
    from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
    p = Path(path); config = Qwen3Config.from_pretrained(path)
    if (not config.ttt_mode or config.ttt_chunk != 4096 or tuple(config.ttt_layers) != LAYERS
            or config.ttt_lr != .3 or config.ttt_target != 'hidden_states'):
        raise ValueError('unexpected trained native configuration')
    model, info = Qwen3ForCausalLM.from_pretrained(path, config=config, dtype=torch.bfloat16,
        attn_implementation='sdpa', output_loading_info=True)
    check_loading(info, [])
    model.to('cuda').eval().requires_grad_(False)
    if arm == 'final_zero_lr':
        for layer in model.model.layers:
            if hasattr(layer.mlp, 'ttt_lr'):
                layer.mlp.ttt_lr = 0.
    meta = {'path': path, 'class': type(model).__module__+'.'+type(model).__name__,
            'loading_info': info, 'excluded_adapters': [], 'config': config.to_dict(),
            'config_sha256': file_digest(p/'config.json'), 'attention': model.config._attn_implementation,
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'source_files_sha256': {x.name:file_digest(x) for x in sorted(p.glob('*.safetensors'))},
            'effective_ttt_lr': 0. if arm == 'final_zero_lr' else .3}
    return model, meta


def collect(args):
    import torch
    from tasks.cbf_selective import measure
    from tasks.cbf_memory_value import score_query
    scenes, d = load_data(args.root)
    selected = [r for r in scenes if (r['group']//2+r['group']%2)%2 == args.shard
                and (args.arm != 'final_zero_lr' or r['group'] in (0,2))]
    selected.sort(key=lambda r:(r['group'],r['id'],r['cell']))
    out = Path(args.output)
    if out.exists(): raise FileExistsError(out)
    torch.manual_seed(211); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    start = time.perf_counter(); model, meta = load_model(args.model, args.arm)
    if args.arm.startswith(('step10000_', 'step40000_')):
        audit_path = Path(args.model)/'export_audit.json'
        audit = json.loads(audit_path.read_text())
        if not audit['source_inventory_unchanged'] or audit['source_files_sha256'] != meta['source_files_sha256'] or audit['config_sha256'] != meta['config_sha256']:
            raise ValueError('export provenance mismatch')
        meta['export_audit_sha256'] = file_digest(audit_path)
    before = parameter_digest(model); versions = [p._version for p in model.parameters()]
    is_native = not args.arm.endswith('_plain')
    cls = NativeSession if is_native else PlainSession
    with out.open('x') as sink:
        for j, r in enumerate(selected):
            torch.cuda.empty_cache(); trace = []
            def prefix():
                session = cls(model)
                for chunk in r['chunks']:
                    session._forward(chunk)
                    if is_native: trace.append(native_stats(session))
                return session
            session, profile = measure(prefix); parent = native_digest(session.cache); queries = {}
            for name, fs in r['queries'].items():
                queries[name] = {}
                for fmt, q in fs.items():
                    v, p = measure(lambda:score_query(session, q, r['choice_ids'], 'full_kv', {}))
                    queries[name][fmt] = {**v, **p}
            if parent != native_digest(session.cache): raise RuntimeError('query modified parent state')
            row = {k:r[k] for k in FIELDS}
            row.update(arm=args.arm, model=args.model, data_sha256=d['data_sha256'], weights_sha256=before,
                       parent_unchanged=True, queries=queries, rollout=profile, native_update_trace=trace)
            sink.write(json.dumps(row)+'\n'); sink.flush()
            print(json.dumps({'arm':args.arm, 'completed':j+1, 'total':len(selected)}), flush=True)
            del session
    if before != parameter_digest(model) or versions != [p._version for p in model.parameters()]:
        raise RuntimeError('model weights modified')
    meta.update(arm=args.arm, shard=args.shard, rows=len(selected), rows_sha256=file_digest(out),
                weights_sha256=before, backbone_unchanged=True, seconds=time.perf_counter()-start,
                torch=torch.__version__, transformers=__import__('transformers').__version__, gpu=torch.cuda.get_device_name())
    Path(str(out)+'.audit.json').write_text(json.dumps(meta, indent=2)+'\n')


def score_difference(rows, refs, reblocked=False):
    index = {(r['id'],r['cell']):r for r in refs}; errors=[]; same=[]; greedy=[]
    for r in rows:
        old = index[r['id'],r['cell']]
        if r['source_context_sha256' if reblocked else 'context_sha256'] != old['context_sha256']:
            raise ValueError('comparison contexts differ')
        for name, fs in r['queries'].items():
            for fmt, v in fs.items():
                b = old['queries'][name][fmt]
                errors.append(max(abs(x-y) for x,y in zip(v['choice_nll'],b['choice_nll'])))
                same.append(v['prediction'] == b['prediction']); greedy.append(v['greedy_id'] == b['greedy_id'])
    return {'queries':len(errors), 'max_choice_nll_error':max(errors),
            'mean_max_choice_nll_error':statistics.fmean(errors),
            'prediction_agreement':statistics.fmean(same), 'greedy_agreement':statistics.fmean(greedy)}


def summarize(root):
    root = Path(root); scenes, d = load_data(root); arms={}; audits={}; tables={}; hashes={}
    for arm in ALL_ARMS:
        rows=[]; audits[arm]=[]
        expected = [r for r in scenes if arm != 'final_zero_lr' or r['group'] in (0,2)]
        for shard in (0,1):
            p=root/f'{arm}_{shard}.jsonl'; part=list(map(json.loads,p.read_text().splitlines()))
            a=json.loads(Path(str(p)+'.audit.json').read_text())
            if a['rows'] != len(part) or a['rows_sha256'] != file_digest(p) or not a['backbone_unchanged'] or a['arm'] != arm or a['shard'] != shard:
                raise ValueError('shard audit mismatch')
            check_loading(a['loading_info'], a['excluded_adapters'])
            if a['attention'] != 'sdpa' or bool(a['config']['ttt_mode']) == arm.endswith('_plain'):
                raise ValueError('wrong inference path')
            for r in part:
                if r['arm'] != arm or r['weights_sha256'] != a['weights_sha256'] or r['model'] != a['path'] or (r['group']//2+r['group']%2)%2 != shard:
                    raise ValueError('row provenance mismatch')
                trace=r['native_update_trace']
                if len(trace) != (0 if arm.endswith('_plain') else 3): raise ValueError('update count')
                if any(set(t) != set(map(str,LAYERS)) or any(not math.isfinite(v) or v < 0 for v in t.values()) for t in trace):
                    raise ValueError('invalid native update')
                if arm == 'final_zero_lr' and any(v != 0 for t in trace for v in t.values()): raise ValueError('zero lr updated weights')
            rows+=part; audits[arm].append(a); hashes[p.name]=file_digest(p)
        if any(len({json.dumps(a[k],sort_keys=True) for a in audits[arm]}) != 1 for k in ('weights_sha256','source_files_sha256','config','path')):
            raise ValueError('mixed model shards')
        validate_scores(rows, expected, d['data_sha256'])
        exp={(r['id'],r['cell']):r for r in expected}
        if any(any(r[k] != exp[r['id'],r['cell']][k] for k in FIELDS) for r in rows): raise ValueError('input identity mismatch')
        arms[arm]=rows
        if arm not in ARMS: continue
        domains={domain:aggregate([r for r in rows if r['domain']==domain]) for domain in ('fineweb','longcrawl')}
        profiles=[p for r in rows for p in [r['rollout']]+[v for fs in r['queries'].values() for v in fs.values()]]
        tables[arm]={'means':aggregate(rows), 'domain_means':domains,
                     'group_means':{str(g):aggregate([r for r in rows if r['group']==g]) for g in range(8)},
                     'binding_gate':choose_format(domains)['formats']['binding'],
                     'peak':{k:max(p[k] for p in profiles) for k in ('peak_allocated_gib','peak_reserved_gib')},
                     'rollout_seconds_mean':statistics.fmean(r['rollout']['seconds'] for r in rows),
                     'native_relative_displacement_mean':[{str(i):statistics.fmean(r['native_update_trace'][t][str(i)] for r in rows) for i in LAYERS} for t in range(3)] if arm.endswith('_native') else []}
    bridges={}
    source=Path(d['source_root'])
    for arm, old_arm in (('original_plain','original_repo'),('step81381_plain','final_repo')):
        refs=[]
        for shard in (0,1):
            p=source/f'{old_arm}_{shard}.jsonl'; a=json.loads(Path(str(p)+'.audit.json').read_text())
            if a['rows_sha256']!=file_digest(p):raise ValueError('reference rows changed')
            for k in ('weights_sha256','source_files_sha256','config_sha256'):
                if a[k]!=audits[arm][shard][k]:raise ValueError('bridge model changed')
            refs += [r for r in map(json.loads,p.read_text().splitlines()) if r['segment_tokens']==2048]
        bridges[arm+'_4096_vs_2048']=score_difference(arms[arm],refs,True)
    for step in STEPS:
        a,b=(audits[f'step{step}_{m}'][0] for m in ('plain','native'))
        if a['source_files_sha256'] != b['source_files_sha256'] or a['config_sha256'] != b['config_sha256']:
            raise ValueError('paired checkpoint differs')
    if audits['final_zero_lr'][0]['weights_sha256'] != audits['step81381_native'][0]['weights_sha256']:
        raise ValueError('zero lr checkpoint differs')
    zero=score_difference(arms['final_zero_lr'],arms['step81381_plain'])
    zero['diagnostic_passed']=zero['max_choice_nll_error']<=.05 and zero['prediction_agreement']>=.95
    bridges['final_zero_lr_vs_plain']=zero
    differences={}
    for step in STEPS:
        differences[str(step)]={str(g):{q:statistics.fmean(tables[f'step{step}_native']['group_means'][str(g)][reg][q]['binding']['correct']-tables[f'step{step}_plain']['group_means'][str(g)][reg][q]['binding']['correct'] for reg in REGIMES) for q in ('target','anchor')} for g in range(8)}
    return {'protocol':'checkpoint_native_v1','contexts':sum(len(x) for x in arms.values()), 'queries':sum(len(fs) for rs in arms.values() for r in rs for fs in r['queries'].values()),
            'primary_format':'binding','confirm_scored':False,'data_sha256':d['data_sha256'],
            'source_summary_sha256':SUMMARY_SHA,'models':tables,'bridges':bridges,
            'native_minus_plain_group_accuracy':differences,'loading_audits':audits,'rows_sha256':hashes,
            'original_binding_gate_passed':tables['original_plain']['binding_gate']['passed'],
            'terminal_status':'completed_checkpoint_native_diagnostic' if zero['diagnostic_passed'] else 'completed_with_native_path_tripwire',
            'scope':'Observed dev only; no training, controller, labels, or held-out evaluation.'}


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build'); b.add_argument('--source',required=True); b.add_argument('--root',required=True)
    c=sub.add_parser('collect')
    for name in ('root','model','output'): c.add_argument('--'+name,required=True)
    c.add_argument('--arm',choices=ALL_ARMS,required=True);c.add_argument('--shard',type=int,choices=(0,1),required=True)
    s=sub.add_parser('summarize');s.add_argument('--root',required=True);s.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='build':build(a.source,a.root)
    elif a.command=='collect':collect(a)
    else:Path(a.output).write_text(json.dumps(summarize(a.root),indent=2)+'\n')


if __name__=='__main__':main()
