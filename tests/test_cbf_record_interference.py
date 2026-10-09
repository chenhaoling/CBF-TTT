"""Token-level intervention isolation and fail-closed record-load construction."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
import unittest
from tasks.build_cbf_memory_value import make_scenes,domain
from tasks.cbf_readability import rebuild
from tasks.cbf_record_interference import make_rows,validate,slots,row_hash,summarize
from tasks.build_cbf_memory_value import file_digest
from tests.test_cbf_memory_value import Tokenizer


class TestTokenizer(Tokenizer):
    def encode(self,text,add_special_tokens=False):
        if text=='\n':return [999]
        return super().encode(text,add_special_tokens)


class InterferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = TestTokenizer()
        groups = [{'domain':domain(g),'sources':[f'source-{g}'],
            'chunks':[[1000+g*100000+c*4096+i for i in range(4096)] for c in range(6)]} for g in range(24)]
        cls.bg = {g:r['chunks'] for g,r in enumerate(groups)}
        cls.originals = [r for r in make_scenes(groups,cls.tok) if r['split']=='dev' and r['regime']=='stable']
        rebuilt = {r['id']:rebuild(r,cls.bg[r['group']],cls.tok,'long_far','dual') for r in cls.originals}
        cls.rows = make_rows(cls.originals,cls.bg,rebuilt,cls.tok)

    def test_exact_original_and_target_with_local_interventions(self):
        validate(self.rows)
        original = {r['id']:r for r in self.originals}
        for r in self.rows:
            if r['cell']=='original10':self.assertEqual(r['chunks'],original[r['id']]['chunks'])
            if r['cell']=='rebuilt_bridge':continue
            self.assertEqual(r['chunks'][1],original[r['id']]['chunks'][1])
            for s in r['slots']:
                c,a,b = s['chunk'],s['start'],s['end']
                expected = (original[r['id']]['chunks'][c-1][a:b] if c in r['active_chunks'] else
                    self.bg[r['group']][c-1][a:b] if r['filler']=='natural' else [999]*(b-a))
                self.assertEqual(r['chunks'][c-1][a:b],expected)
                self.assertEqual(r['chunks'][c-1][b],original[r['id']]['chunks'][c-1][b])

    def test_background_mutation_and_incomplete_matrix_rejected(self):
        with self.assertRaises(ValueError):validate(self.rows[:-1])
        bad = copy.deepcopy(self.rows)
        r = next(r for r in bad if r['cell']=='natural0')
        # Mutate both twins to evade the one-token twin check; outside-slot audit must catch it.
        for x in bad:
            if x['cell']=='natural0' and x['group']==r['group']:
                x['chunks'][0][0]+=1;x['context_sha256']=row_hash(x['chunks'])
        with self.assertRaisesRegex(ValueError,'outside intervention'):validate(bad)

    def test_heldout_and_suffix_mismatch_rejected(self):
        bad = copy.deepcopy(self.rows);bad[0]['group']=8
        with self.assertRaisesRegex(ValueError,'dev-only'):validate(bad)
        bad = copy.deepcopy(self.originals[0]);bad['chunks'][0][-1]+=1
        with self.assertRaisesRegex(ValueError,'suffix mismatch'):slots(bad,self.tok)


    def test_summary_rejects_score_tampering_even_with_valid_shard_hash(self):
        # Tiny valid contexts exercise coverage, bridges and logit-derived metrics without a GPU.
        tok = TestTokenizer()
        groups = [{'domain':domain(g),'sources':[f'source-{g}'],
            'chunks':[[1000+g*10000+c*256+i for i in range(256)] for c in range(6)]} for g in range(24)]
        originals = [r for r in make_scenes(groups,tok,256) if r['split']=='dev' and r['regime']=='stable']
        rows = make_rows(originals,{g:r['chunks'] for g,r in enumerate(groups)},
                         {r['id']:r for r in originals},tok)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);data = root/'data.jsonl'
            data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            parts = [[],[]];refs = [[],[]];cal = [[],[]]
            for r in rows:
                shard = (r['group']//2+r['group']%2)%2
                profile = dict(seconds=1.,peak_allocated_gib=1.,peak_reserved_gib=2.)
                queries = {}
                for name,q in r['queries'].items():
                    scores = [2.]*8;scores[q['label']]=1.
                    queries[name] = dict(nll=1.,choice_nll=scores,prediction=q['label'],correct=1,
                        greedy_id=q['answer_id'],greedy_correct=1,**profile)
                output = {k:r[k] for k in ('id','cell','group','domain','variant','context_sha256','active_colors')}
                output.update(queries=queries,rollout=profile,parent_unchanged=True,model='test',
                    weights_sha256='test',data_sha256=file_digest(data))
                parts[shard].append(output)
                if r['cell']=='original10':
                    refs[shard].append(dict(id=r['id'],regime='stable',model='test',weights_sha256='test',
                        queries={'full_kv':{'self':queries}}))
                if r['cell']=='rebuilt_bridge':
                    cal[shard].append(dict(id=r['id'],cell='long_far/dual/target_last',model='test',weights_sha256='test',
                        queries={k:{'qa':v} for k,v in queries.items()}))
            def save():
                for shard in (0,1):
                    path = root/f'rows_{shard}.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in parts[shard]))
                    Path(str(path)+'.audit.json').write_text(json.dumps(dict(rows=80,rows_sha256=file_digest(path),
                        weights_sha256='test',backbone_unchanged=True,seconds=1.)))
                    (root/f'ref_{shard}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in refs[shard]))
                    (root/f'cal_{shard}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cal[shard]))
            save()
            args = SimpleNamespace(data=str(data),inputs=[str(root/f'rows_{i}.jsonl') for i in (0,1)],
                reference=[str(root/f'ref_{i}.jsonl') for i in (0,1)],
                readability_reference=[str(root/f'cal_{i}.jsonl') for i in (0,1)],output=str(root/'summary.json'))
            with redirect_stdout(io.StringIO()):summarize(args)
            self.assertEqual(json.loads((root/'summary.json').read_text())['bridge_checks'],64)
            parts[0][0]['queries']['target']['nll']+=1
            save()
            with self.assertRaisesRegex(ValueError,'score mismatch'):summarize(args)


if __name__=='__main__':unittest.main()
