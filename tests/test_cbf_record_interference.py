"""Token-level intervention isolation and fail-closed record-load construction."""
import copy
import unittest
from tasks.build_cbf_memory_value import make_scenes,domain
from tasks.cbf_readability import rebuild
from tasks.cbf_record_interference import make_rows,validate,slots,row_hash
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


if __name__=='__main__':unittest.main()
