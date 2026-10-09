"""Factorial calibration construction and held-out exclusion checks."""
import copy
import unittest
from tasks.cbf_readability import rebuild,validate,aggregate,LAYOUTS,LOADS
from tasks.build_cbf_memory_value import make_scenes,domain,digest
from tests.test_cbf_memory_value import Tokenizer


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=Tokenizer()
        groups=[{'domain':domain(g),'sources':[f's{g}'],
                 'chunks':[[1000+g*100000+c*4096+i for i in range(4096)] for c in range(6)]} for g in range(24)]
        originals=make_scenes(groups,cls.tokenizer)
        cls.rows=[];cls.originals=[r for r in originals if r['split']=='dev' and r['regime']=='stable']
        for scene in cls.originals:
            bg=groups[scene['group']]['chunks']
            for layout in LAYOUTS:
                for load in LOADS:cls.rows.append(rebuild(scene,bg,cls.tokenizer,layout,load))
            cls.rows.append(rebuild(scene,bg,cls.tokenizer,'long_far','dual','target_first'))
            row={k:scene[k] for k in ('id','group','domain','variant','sources','choice_ids','chunks')}
            row.update(cell='bridge',layout='bridge',load='dual',order='original',
                context_sha256=digest(__import__('json').dumps(scene['chunks']).encode()),
                queries={name:{'qa':q} for name,q in scene['queries'].items()})
            cls.rows.append(row)

    def test_complete_matrix_is_balanced_dev_only(self):
        validate(self.rows)
        self.assertEqual(sum(len(fs) for r in self.rows for fs in r['queries'].values()),384)
        bad=list(self.rows[:-1])
        with self.assertRaises(ValueError):validate(bad)
        bad=list(self.rows);bad[0]=dict(bad[0],group=8)
        with self.assertRaisesRegex(ValueError,'non-dev'):validate(bad)

    def test_slot_swap_and_short_long_prefix(self):
        index={r['cell']:r for r in self.rows if r['id']==self.originals[0]['id']}
        a=index['long_far/dual/target_last'];b=index['long_far/dual/target_first']
        self.assertEqual(a['suffix_start'],b['suffix_start'])
        self.assertEqual(a['chunks'][1][:a['suffix_start']],b['chunks'][1][:b['suffix_start']])
        self.assertEqual(sorted(a['chunks'][1]),sorted(b['chunks'][1]))
        self.assertNotEqual(a['chunks'][1],b['chunks'][1])
        self.assertEqual(index['short/dual/target_last']['chunks'],a['chunks'][:2])
        for c in (0,2,3,4,5):self.assertEqual(a['chunks'][c],b['chunks'][c])

    def test_labels_are_not_query_inputs(self):
        for row in self.rows:
            for name,formats in row['queries'].items():
                for fmt,q in formats.items():
                    self.assertNotIn(q['answer_id'],q['ids'])
                    self.assertEqual(q['answer_id'],row['choice_ids'][q['label']])
            if row['load']=='single':self.assertEqual(set(row['queries']),{'target'})


if __name__=='__main__':unittest.main()
