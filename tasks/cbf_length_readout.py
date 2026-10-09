"""Frozen nested-background length diagnostic using unchanged disjoint records."""
import argparse
import copy
import json
from pathlib import Path
import statistics

from tasks.build_cbf_memory_value import file_digest, REGIMES
from tasks.cbf_record_interference import row_hash
from tasks.cbf_disjoint_readout import validate as validate_source, aggregate, choose_format, IDENTITY, FORMATS
from tasks.cbf_checkpoint_readout import (
    DATA_SHA, METRICS, collect as collect_plain, check_loading, compare_scores, validate_scores,
    summarize as checkpoint_summary,
)

SHORT_LENGTHS=(1024,2048)
BRIDGE_GROUPS=(0,2)
ARMS=('original_repo','final_repo')
FIELDS=IDENTITY+('segment_tokens','source_cell','source_context_sha256')
REFERENCE_SHA='407d5b1cfca505f07e3bf30630417c06c4d270ef6ea166be9ba0bc5ef5857763'


def make_rows(sources,lengths=SHORT_LENGTHS):
    """Retain each segment's exact suffix; records and query tokens are never rewritten."""
    validate_source(sources);rows=[]
    for old in sources:
        if old['is_bridge']:continue
        for length in (*lengths,4096):
            if length==4096 and old['group'] not in BRIDGE_GROUPS:continue
            if old['chunk_size']!=4096:raise ValueError('source segment size')
            row=copy.deepcopy(old);drop=4096-length
            if length<=0 or drop<0:raise ValueError('invalid segment length')
            for record in row['records']:
                if record['offset']<drop:raise ValueError('window cuts session records/header')
                record['offset']-=drop
            row['chunks']=[c[drop:] for c in old['chunks']]
            row.pop('changed_positions',None)
            row.update(segment_tokens=length,chunk_size=length,source_cell=old['cell'],
                source_context_sha256=old['context_sha256'],cell=f'length{length}/{old["regime"]}',
                is_bridge=length==4096,context_sha256=row_hash(row['chunks']))
            rows.append(row)
    return rows


def validate(rows,sources,lengths=SHORT_LENGTHS):
    expected=make_rows(sources,lengths)
    index={(r['id'],r['cell']):r for r in rows}
    if len(index)!=len(expected) or len(rows)!=len(expected):raise ValueError('incomplete/duplicate matrix')
    for r in expected:
        if index.get((r['id'],r['cell']))!=r:raise ValueError('source/window/record/query mismatch')
    # A target flip remains exactly one token at each shorter length.
    for r in rows:
        twin=index[r['twin_id'],r['cell']]
        if sum(a!=b for c,d in zip(r['chunks'],twin['chunks']) for a,b in zip(c,d))!=1:
            raise ValueError('twin changed outside target token')


def load_sources(source):
    source=Path(source);data=source/'data/scenes.jsonl'
    if file_digest(data)!=DATA_SHA:raise ValueError('original data changed')
    rows=list(map(json.loads,data.read_text().splitlines()));validate_source(rows)
    return rows


def reference_summary(reference,source):
    reference=Path(reference);audit=json.loads((reference/'execution_audit.json').read_text())
    if not audit['audit_passed'] or audit['summary_sha256']!=REFERENCE_SHA or file_digest(reference/'summary.json')!=REFERENCE_SHA:
        raise ValueError('reference audit/hash')
    old=json.loads((reference/'summary.json').read_text())
    if checkpoint_summary(reference,source)!=old:raise ValueError('reference recomputation')
    return old


def build(args):
    source=Path(args.source);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    rows=make_rows(load_sources(source));validate(rows,load_sources(source))
    reference_summary(args.reference,source)
    out.mkdir(parents=True);data=out/'scenes.jsonl'
    data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    d={'protocol':'length_readout_v1','source_root':str(source),'reference_root':args.reference,
        'source_data_sha256':DATA_SHA,'reference_summary_sha256':REFERENCE_SHA,'data_sha256':file_digest(data),
        'source_design_sha256':file_digest(source/'data/design.json'),
        'segment_lengths':[*SHORT_LENGTHS,4096],'context_lengths':[6*x for x in (*SHORT_LENGTHS,4096)],
        'bridge_groups':list(BRIDGE_GROUPS),'contexts_per_model':144,'queries_per_model':576,
        'new_contexts_both_models':288,'new_queries_both_models':1152,
        'analysis_conditions':384,'analysis_queries':1536,'confirm_scored':False,
        'construction':'Exact last L tokens per original segment; no retokenization or record deletion.',
        'context_hashes':[{k:r[k] for k in (*FIELDS,'sources')} for r in rows]}
    (out/'design.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({k:v for k,v in d.items() if k!='context_hashes'}))


def load_data(root):
    root=Path(root);d=json.loads((root/'data/design.json').read_text());data=root/'data/scenes.jsonl'
    if file_digest(data)!=d['data_sha256'] or d['source_data_sha256']!=DATA_SHA or d['reference_summary_sha256']!=REFERENCE_SHA:
        raise ValueError('data manifest changed')
    if d['segment_lengths']!=[*SHORT_LENGTHS,4096] or d['bridge_groups']!=list(BRIDGE_GROUPS):raise ValueError('protocol changed')
    rows=list(map(json.loads,data.read_text().splitlines()));sources=load_sources(d['source_root'])
    if file_digest(Path(d['source_root'])/'data/design.json')!=d['source_design_sha256']:raise ValueError('source design changed')
    validate(rows,sources)
    if d['context_hashes']!=[{k:r[k] for k in (*FIELDS,'sources')} for r in rows]:raise ValueError('manifest contexts')
    if len(rows)!=144 or sum(len(fs) for r in rows for fs in r['queries'].values())!=576:raise ValueError('matrix size')
    return rows,d


def collect(args):
    rows,d=load_data(args.root)
    sources=load_sources(d['source_root'])
    collect_plain(args,data_path=Path(args.root)/'data/scenes.jsonl',
        validator=lambda rs:validate(rs,sources),identity_fields=FIELDS,expected_data_sha=d['data_sha256'])


def paired(a,b):
    return {g:{r:{q:{f:{k:a[g][r][q][f][k]-b[g][r][q][f][k] for k in METRICS}
        for f in FORMATS} for q in ('target','anchor')} for r in REGIMES} for g in map(str,range(8))}


def select_length(tables):
    eligible=[length for length in SHORT_LENGTHS if tables[str(length)]['selection']['passed']]
    length=max(eligible) if eligible else None
    return {'passed':length is not None,'segment_tokens':length,'context_tokens':6*length if length else None,
        'format':tables[str(length)]['selection']['selected_format'] if length else None,
        'rule':'Longest passing prespecified short length; every domain/regime/query >=75%; max worst format, QA tie.'}


def summarize(root):
    root=Path(root);scenes,d=load_data(root);old=reference_summary(d['reference_root'],d['source_root'])
    models={};audits={};bridges={};rows_hashes={}
    for arm in ARMS:
        rows=[];audits[arm]=[]
        for shard in (0,1):
            p=root/f'{arm}_{shard}.jsonl';part=list(map(json.loads,p.read_text().splitlines()))
            a=json.loads(Path(str(p)+'.audit.json').read_text())
            if a['rows_sha256']!=file_digest(p) or a['rows']!=72 or len(part)!=72 or not a['backbone_unchanged']:raise ValueError('shard audit')
            if a['arm']!=arm or a['shard']!=shard:raise ValueError('arm audit')
            if any((r['group']//2+r['group']%2)%2!=shard or r['arm']!=arm or r['model']!=a['path'] or r['weights_sha256']!=a['weights_sha256'] for r in part):raise ValueError('row model/shard')
            check_loading(a['loading_info'],a['excluded_adapters'])
            previous=old['loading_audits'][arm][shard]
            for k in ('weights_sha256','source_files_sha256','config_sha256','config','class','path','attention','excluded_adapters'):
                if a[k]!=previous[k]:raise ValueError('checkpoint/config differs from reference')
            rows+=part;audits[arm].append(a);rows_hashes[p.name]=file_digest(p)
        validate_scores(rows,scenes,d['data_sha256'])
        index={(r['id'],r['cell']):r for r in scenes}
        if any(any(r[k]!=index[r['id'],r['cell']][k] for k in FIELDS) for r in rows):raise ValueError('length identity')
        ref=[]
        for shard in (0,1):ref+=list(map(json.loads,(Path(d['reference_root'])/f'{arm}_{shard}.jsonl').read_text().splitlines()))
        checks=[]
        for r in rows:
            if r['segment_tokens']!=4096:continue
            item=copy.deepcopy(r);item['cell']=r['source_cell'];item['data_sha256']=DATA_SHA;checks.append(item)
        bridges[arm]=compare_scores(checks,ref)
        if bridges[arm]['queries']!=64:raise ValueError('bridge incomplete')
        tables={}
        for length in SHORT_LENGTHS:
            part=[r for r in rows if r['segment_tokens']==length]
            if len(part)!=64:raise ValueError('short matrix')
            domains={dom:aggregate([r for r in part if r['domain']==dom]) for dom in ('fineweb','longcrawl')}
            tables[str(length)]={'means':aggregate(part),'domain_means':domains,
                'group_means':{str(g):aggregate([r for r in part if r['group']==g]) for g in range(8)},
                'selection':choose_format(domains),'source':'new_collection'}
        tables['4096']={k:old['models'][arm][k] for k in ('means','domain_means','group_means','selection')}
        tables['4096']['source']='audited_previous_full_64_clean_contexts'
        profiles={}
        for length in (*SHORT_LENGTHS,4096):
            part=[r for r in rows if r['segment_tokens']==length]
            ps=[p for r in part for p in [r['rollout']]+[v for fs in r['queries'].values() for v in fs.values()]]
            profiles[str(length)]={'new_contexts':len(part),
                **{k:max(p[k] for p in ps) for k in ('peak_allocated_gib','peak_reserved_gib')},
                'rollout_mean_seconds':statistics.fmean(r['rollout']['seconds'] for r in part),
                'query_mean_seconds':statistics.fmean(v['seconds'] for r in part for fs in r['queries'].values() for v in fs.values())}
        models[arm]={'lengths':tables,'candidate':select_length(tables),'profile':profiles,
            'short_minus_24k_group_differences':{str(n):paired(tables[str(n)]['group_means'],tables['4096']['group_means']) for n in SHORT_LENGTHS}}
    return {'protocol':'length_readout_v1','data_sha256':d['data_sha256'],'design_sha256':file_digest(root/'data/design.json'),
        'source_data_sha256':DATA_SHA,'reference_summary_sha256':REFERENCE_SHA,
        'new_contexts':288,'new_queries':1152,'analysis_conditions':384,'analysis_queries':1536,
        'models':models,'final_minus_original_group_differences':{str(n):paired(models['final_repo']['lengths'][str(n)]['group_means'],
            models['original_repo']['lengths'][str(n)]['group_means']) for n in (*SHORT_LENGTHS,4096)},
        'bridges':bridges,'loading_audits':audits,'rows_sha256':rows_hashes,'M':0,'confirm_scored':False,
        'terminal_status':'dev_length_candidate' if models['final_repo']['candidate']['passed'] else 'stopped_by_length_Q',
        'scope':'Observed dev nested tail windows; no training/new-source confirmation/M/V/F.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--source',required=True);b.add_argument('--reference',required=True);b.add_argument('--output',required=True)
    c=sub.add_parser('collect');c.add_argument('--root',required=True);c.add_argument('--arm',choices=ARMS,required=True)
    c.add_argument('--model',required=True);c.add_argument('--shard',type=int,choices=(0,1),required=True);c.add_argument('--output',required=True)
    s=sub.add_parser('summarize');s.add_argument('--root',required=True);s.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='build':build(args)
    elif args.command=='collect':collect(args)
    else:
        result=summarize(args.root);Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'status':result['terminal_status'],'bridges':result['bridges'],'candidates':{k:v['candidate'] for k,v in result['models'].items()}}))


if __name__=='__main__':main()
