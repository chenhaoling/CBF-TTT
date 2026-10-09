"""Exact nested windows, full matrix preservation, selection and score provenance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_record_interference import row_hash
from tasks.cbf_length_readout import make_rows,validate,select_length,summarize,FIELDS,ARMS
from tasks.cbf_disjoint_readout import aggregate,choose_format


class LengthReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_cbf_disjoint_readout import DisjointReadoutTests
        DisjointReadoutTests.setUpClass();cls.sources=copy.deepcopy(DisjointReadoutTests.rows)
        for r in cls.sources:
            padding=4096-r['chunk_size'];r['chunk_size']=4096
            r['chunks']=[[987654]*padding+c for c in r['chunks']]
            for rec in r['records']:rec['offset']+=padding
            if 'changed_positions' in r:r['changed_positions']=[[i+padding for i in p] for p in r['changed_positions']]
            r['context_sha256']=row_hash(r['chunks'])
        cls.rows=make_rows(cls.sources)

    def test_nested_windows_preserve_all_records_and_queries(self):
        validate(self.rows,self.sources);self.assertEqual(len(self.rows),144)
        old={r['id']:r for r in self.sources if not r['is_bridge']}
        index={(r['id'],r['segment_tokens']):r for r in self.rows}
        for r in self.rows:
            src=old[r['id']];length=r['segment_tokens']
            self.assertEqual(r['queries'],src['queries']);self.assertEqual(r['sources'],src['sources'])
            for i,(chunk,rec) in enumerate(zip(r['chunks'],r['records'])):
                self.assertEqual(chunk,src['chunks'][i][-length:])
                self.assertEqual(rec['entries'],src['records'][i]['entries'])
                self.assertEqual(chunk[rec['offset']:],src['chunks'][i][src['records'][i]['offset']:])
            if length==1024:
                self.assertEqual(r['chunks'],[c[-1024:] for c in index[r['id'],2048]['chunks']])

    def test_tampering_cut_records_and_missing_context_rejected(self):
        with self.assertRaisesRegex(ValueError,'cuts'):make_rows(self.sources,(4,))
        with self.assertRaisesRegex(ValueError,'matrix'):validate(self.rows[:-1],self.sources)
        bad=copy.deepcopy(self.rows);bad[0]['chunks'][0][0]+=1;bad[0]['context_sha256']=row_hash(bad[0]['chunks'])
        with self.assertRaisesRegex(ValueError,'mismatch'):validate(bad,self.sources)
        bad=copy.deepcopy(self.rows);bad[0]['queries']['target']['qa']['ids'].append(1)
        with self.assertRaisesRegex(ValueError,'mismatch'):validate(bad,self.sources)

    def test_choose_longest_only_if_gate_passed(self):
        tables={str(n):{'selection':{'passed':False,'selected_format':None}} for n in (1024,2048,4096)}
        self.assertFalse(select_length(tables)['passed'])
        tables['1024']['selection'].update(passed=True,selected_format='binding')
        self.assertEqual(select_length(tables)['context_tokens'],6144)
        tables['2048']['selection'].update(passed=True,selected_format='qa')
        self.assertEqual(select_length(tables)['context_tokens'],12288)
        self.assertEqual(select_length(tables)['format'],'qa')

    def test_complete_summary_rejects_changed_length_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir();(root/'data/design.json').write_text('{}')
            reference=root/'reference';reference.mkdir();old={'models':{},'loading_audits':{}}
            parts={};profile=dict(seconds=1.,peak_allocated_gib=1.,peak_reserved_gib=2.)
            for arm in ARMS:
                allrows=[];old['loading_audits'][arm]=[]
                for shard in (0,1):
                    part=[]
                    for r in self.rows:
                        if (r['group']//2+r['group']%2)%2!=shard:continue
                        queries={}
                        for name,fs in r['queries'].items():
                            queries[name]={}
                            for fmt,q in fs.items():
                                scores=[2.]*8;scores[q['label']]=1.
                                queries[name][fmt]=dict(choice_nll=scores,nll=1.,prediction=q['label'],correct=1,
                                    greedy_id=q['answer_id'],greedy_correct=1,**profile)
                        item={k:r[k] for k in FIELDS};item.update(queries=queries,rollout=profile,arm=arm,
                            model=arm,weights_sha256=arm,parent_unchanged=True,data_sha256='new');part.append(item)
                    allrows+=part;parts[arm,shard]=part
                    p=root/f'{arm}_{shard}.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in part))
                    adapters=['model.layers.0.mlp.ttt_proj.weight'] if arm=='final_repo' else []
                    a=dict(rows=72,rows_sha256=file_digest(p),arm=arm,shard=shard,backbone_unchanged=True,
                        path=arm,weights_sha256=arm,source_files_sha256={},config_sha256='config',config={},
                        attention='sdpa',excluded_adapters=adapters,loading_info={'unexpected_keys':adapters})
                    a['class']='test';Path(str(p)+'.audit.json').write_text(json.dumps(a));old['loading_audits'][arm].append(a)
                    refs=[]
                    for r in part:
                        if r['segment_tokens']!=4096:continue
                        rr=copy.deepcopy(r);rr.update(cell=r['source_cell'],data_sha256='source');refs.append(rr)
                    (reference/f'{arm}_{shard}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in refs))
                # Synthetic previous full clean matrix has identical scores, all passing.
                rows=[r for r in allrows if r['segment_tokens']==1024]
                domains={d:aggregate([r for r in rows if r['domain']==d]) for d in ('fineweb','longcrawl')}
                old['models'][arm]={'means':aggregate(rows),'domain_means':domains,'selection':choose_format(domains),
                    'group_means':{str(g):aggregate([r for r in rows if r['group']==g]) for g in range(8)}}
            d=dict(data_sha256='new',source_root='source',reference_root=str(reference))
            with patch('tasks.cbf_length_readout.load_data',return_value=(self.rows,d)),patch('tasks.cbf_length_readout.reference_summary',return_value=old),patch('tasks.cbf_length_readout.DATA_SHA','source'):
                s=summarize(root);self.assertEqual(s['new_queries'],1152)
                self.assertEqual(s['bridges']['original_repo']['queries'],64)
                self.assertEqual(s['models']['final_repo']['candidate']['context_tokens'],12288)
                part=parts['final_repo',0];part[0]['segment_tokens']=2048
                p=root/'final_repo_0.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in part))
                ap=Path(str(p)+'.audit.json');a=json.loads(ap.read_text());a['rows_sha256']=file_digest(p);ap.write_text(json.dumps(a))
                with self.assertRaisesRegex(ValueError,'length identity'):summarize(root)


if __name__=='__main__':unittest.main()
