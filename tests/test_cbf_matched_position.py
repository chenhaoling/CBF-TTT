"""Matched identity interventions, exact bridges and fail-closed score aggregation."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from tasks.build_cbf_memory_value import make_scenes,domain,file_digest
from tasks.cbf_record_interference import make_rows as previous_rows,row_hash
from tasks.cbf_matched_position import make_rows,validate,chunk_slots,summarize,BRIDGES
from tests.test_cbf_record_interference import TestTokenizer


class MatchedPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok=TestTokenizer()
        groups=[{'domain':domain(g),'sources':[f'source-{g}'],
            'chunks':[[1000+g*10000+c*256+i for i in range(256)] for c in range(6)]} for g in range(24)]
        cls.originals=[r for r in make_scenes(groups,tok,256) if r['split']=='dev' and r['regime']=='stable']
        cls.previous=previous_rows(cls.originals,{g:r['chunks'] for g,r in enumerate(groups)},
            {r['id']:r for r in cls.originals},tok)
        cls.rows=make_rows(cls.previous,cls.originals)

    def test_same_tokens_at_all_positions_and_exact_bridges(self):
        validate(self.rows)
        originals={r['id']:r for r in self.originals}
        old={(r['id'],r['cell']):r for r in self.previous}
        for r in self.rows:
            self.assertEqual(r['chunks'][1],originals[r['id']]['chunks'][1])
            if r['cell'] in BRIDGES:self.assertEqual(r['chunks'],old[r['id'],BRIDGES[r['cell']]]['chunks'])
            if not r['donor_chunk']:continue
            for src,dst in zip(chunk_slots(r,r['donor_chunk']),chunk_slots(r,r['position'])):
                self.assertEqual(r['chunks'][r['position']-1][dst['start']:dst['end']],
                    originals[r['id']]['chunks'][r['donor_chunk']-1][src['start']:src['end']])

    def test_background_and_donor_tampering_rejected(self):
        for inside,error in ((False,'outside destination'),(True,'donor tokens')):
            bad=copy.deepcopy(self.rows)
            for r in bad:
                if r['group']==0 and r['cell']=='d3_p1':
                    offset=chunk_slots(r,1)[0]['start'] if inside else 0
                    r['chunks'][0][offset]+=1;r['context_sha256']=row_hash(r['chunks'])
            with self.assertRaisesRegex(ValueError,error):validate(bad)

    def test_incomplete_heldout_and_token_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):validate(self.rows[:-1])
        bad=copy.deepcopy(self.rows);bad[0]['group']=8
        with self.assertRaisesRegex(ValueError,'dev-only'):validate(bad)
        bad=copy.deepcopy(self.previous)
        base=next(r for r in bad if r['cell']=='natural0')
        chunk_slots(base,1)[0]['end']+=1
        with self.assertRaisesRegex(ValueError,'boundaries differ'):make_rows(bad,self.originals)

    def test_valid_summary_and_tampered_score_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'scenes.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
            parts=[[],[]];refs=[[],[]]
            for r in self.rows:
                shard=(r['group']//2+r['group']%2)%2
                profile=dict(seconds=1.,peak_allocated_gib=1.,peak_reserved_gib=2.)
                queries={}
                for name,q in r['queries'].items():
                    scores=[2.]*8;scores[q['label']]=1.
                    queries[name]=dict(nll=1.,choice_nll=scores,prediction=q['label'],correct=1,
                        greedy_id=q['answer_id'],greedy_correct=1,**profile)
                item={k:r[k] for k in ('id','cell','group','domain','variant','active_colors','context_sha256')}
                item.update(queries=queries,rollout=profile,parent_unchanged=True,model='test',
                    weights_sha256='test',data_sha256=file_digest(data));parts[shard].append(item)
                if r['cell'] in BRIDGES:refs[shard].append(dict(item,cell=BRIDGES[r['cell']]))
            def save():
                for shard in (0,1):
                    p=root/f'rows_{shard}.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in parts[shard]))
                    Path(str(p)+'.audit.json').write_text(json.dumps(dict(rows=56,rows_sha256=file_digest(p),
                        backbone_unchanged=True,weights_sha256='test',seconds=1.)))
                    (root/f'ref_{shard}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in refs[shard]))
            save()
            args=SimpleNamespace(data=str(data),inputs=[str(root/f'rows_{i}.jsonl') for i in (0,1)],
                reference=[str(root/f'ref_{i}.jsonl') for i in (0,1)],output=str(root/'summary.json'))
            with redirect_stdout(io.StringIO()):summarize(args)
            s=json.loads((root/'summary.json').read_text());self.assertEqual(s['bridge_checks'],96)
            self.assertFalse(s['consistent_late_target_interference'])
            parts[0][-1]['queries']['target']['correct']=0;save()
            with self.assertRaisesRegex(ValueError,'score mismatch'):summarize(args)


if __name__=='__main__':unittest.main()
