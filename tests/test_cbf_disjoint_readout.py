"""Disjoint distractors, paired truth, strict format gate and scored-output validation."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from tasks.build_cbf_memory_value import make_scenes,domain,REGIMES,file_digest
from tasks.cbf_disjoint_readout import make_rows,validate,choose_format,summarize,FORMATS,IDENTITY
from tests.test_cbf_memory_value import Tokenizer


class DisjointReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok=Tokenizer()
        groups=[{'domain':domain(g),'sources':[f'source-{g}'],
            'chunks':[[1000+g*10000+c*256+i for i in range(256)] for c in range(6)]} for g in range(24)]
        cls.originals=[r for r in make_scenes(groups,tok,256) if r['split']=='dev']
        cls.rows=make_rows(cls.originals,tok)

    def test_only_color_tokens_change_and_critical_facts_survive(self):
        validate(self.rows);old={r['id']:r for r in self.originals}
        for r in self.rows:
            origin=old[r['id']]
            if r['is_bridge']:
                self.assertEqual(r['chunks'],origin['chunks']);continue
            keys={fs['qa']['key'] for fs in r['queries'].values()}
            for i,(before,after) in enumerate(zip(origin['records'],r['records'])):
                self.assertEqual(before['offset'],after['offset'])
                self.assertEqual([x[0] for x in before['entries']],[x[0] for x in after['entries']])
                changes=sum(a[1]!=b[1] for a,b in zip(before['entries'],after['entries']))
                self.assertEqual(len(r['changed_positions'][i]),changes)
                self.assertEqual(r['chunks'][i][:before['offset']],origin['chunks'][i][:before['offset']])
                for a,b in zip(before['entries'],after['entries']):
                    if a[0] in keys:self.assertEqual(a[1],b[1])
                    else:self.assertNotIn(b[1],r['protected_labels'])
            for fs in r['queries'].values():
                for q in fs.values():self.assertNotIn(q['answer_id'],q['ids'])

    def test_overlap_missing_and_heldout_rejected(self):
        with self.assertRaises(ValueError):validate(self.rows[:-1])
        bad=copy.deepcopy(self.rows);bad[0]['split']='confirm'
        with self.assertRaisesRegex(ValueError,'identity'):validate(bad)
        bad=copy.deepcopy(self.rows)
        r=next(r for r in bad if not r['is_bridge'])
        r['records'][0]['entries'][0][1]=r['queries']['target']['qa']['label']
        with self.assertRaisesRegex(ValueError,'overlap'):validate(bad)

    def test_gate_requires_every_domain_regime_and_query(self):
        tables={d:{r:{q:{f:{'correct':1.} for f in FORMATS} for q in ('target','anchor')} for r in REGIMES}
            for d in ('fineweb','longcrawl')}
        self.assertEqual(choose_format(tables)['selected_format'],'qa')
        tables['fineweb']['stable']['anchor']['qa']['correct']=.5
        self.assertEqual(choose_format(tables)['selected_format'],'binding')
        tables['longcrawl']['recent2_distractor']['target']['binding']['correct']=.625
        gate=choose_format(tables);self.assertFalse(gate['passed']);self.assertIsNone(gate['selected_format'])
        self.assertEqual(len(gate['formats']['qa']['failed_cells']),1)

    def test_summary_rejects_wrong_scores_with_valid_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'scenes.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
            parts=[[],[]];refs=[[],[]]
            for r in self.rows:
                shard=(r['group']//2+r['group']%2)%2
                profile=dict(seconds=1.,peak_allocated_gib=1.,peak_reserved_gib=2.)
                queries={}
                for name,fs in r['queries'].items():
                    queries[name]={}
                    for fmt,q in fs.items():
                        scores=[2.]*8;scores[q['label']]=1.
                        queries[name][fmt]=dict(nll=1.,choice_nll=scores,prediction=q['label'],correct=1,
                            greedy_id=q['answer_id'],greedy_correct=1,**profile)
                item={k:r[k] for k in IDENTITY};item.update(queries=queries,rollout=profile,parent_unchanged=True,
                    model='test',weights_sha256='test',data_sha256=file_digest(data));parts[shard].append(item)
                if r['is_bridge']:refs[shard].append(dict(id=r['id'],model='test',weights_sha256='test',
                    queries={'full_kv':{'self':{n:fs['qa'] for n,fs in queries.items()}}}))
            def save():
                for shard in (0,1):
                    p=root/f'rows_{shard}.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in parts[shard]))
                    Path(str(p)+'.audit.json').write_text(json.dumps(dict(rows=40,rows_sha256=file_digest(p),
                        backbone_unchanged=True,weights_sha256='test',seconds=1.)))
                    (root/f'ref_{shard}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in refs[shard]))
            save()
            args=SimpleNamespace(data=str(data),inputs=[str(root/f'rows_{i}.jsonl') for i in (0,1)],
                reference=[str(root/f'ref_{i}.jsonl') for i in (0,1)],output=str(root/'summary.json'))
            with redirect_stdout(io.StringIO()):summarize(args)
            s=json.loads((root/'summary.json').read_text());self.assertEqual(s['bridge_checks'],32)
            self.assertEqual(s['selection']['selected_format'],'qa')
            parts[0][0]['queries']['target']['qa']['nll']+=1;save()
            with self.assertRaisesRegex(ValueError,'score mismatch'):summarize(args)


if __name__=='__main__':unittest.main()
