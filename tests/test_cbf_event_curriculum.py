"""Semantic rollback correctness, disjoint splits and causal training windows."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tasks.build_cbf_event_curriculum import (generate_group, materialize, replay, validate,
                                             supervision_records, warmup_records, SPLITS)


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs=[p for split in SPLITS for p in generate_group(73,split,0)]

    def test_rollback_restores_history_without_erasing_unrelated_fact(self):
        events=[{'op':'assign','batch':'a','facts':{'x':'old','anchor':'safe'}},
                {'op':'assign','batch':'b','facts':{'x':'new'}},
                {'op':'assign','batch':'c','facts':{'y':'keep'}},
                {'op':'revoke','batch':'b'}]
        self.assertEqual(replay(events),{'x':'old','anchor':'safe','y':'keep'})
        self.assertEqual(replay(events+[{'op':'reset'},{'op':'assign','batch':'d','facts':{'z':'fresh'}}]),{'z':'fresh'})

    def test_complete_disjoint_reproducible_matrix(self):
        self.assertEqual(self.pairs,[p for s in SPLITS for p in generate_group(73,s,0)])
        a=validate(self.pairs);self.assertEqual(a['episodes'],72);self.assertEqual(a['queries'],432)
        self.assertFalse(a['oracle_actions_generated'])
        self.assertEqual(len(list(warmup_records(self.pairs))),24)
        # Independent truth reconstruction from surviving batches, for each causal prefix.
        for inp,target in self.pairs:
            truth={a['id']:a for a in target['answers']}
            for q in inp['queries']:
                ev=[x for b in target['semantic_events'][:q['after_block']] for x in b]
                start=max([i+1 for i,e in enumerate(ev) if e['op']=='reset'] or [0]);ev=ev[start:]
                canceled={e['batch'] for e in ev if e['op']=='revoke'};value='unknown'
                for e in reversed(ev):
                    if e['op']=='assign' and e['batch'] not in canceled and truth[q['id']]['key'] in e['facts']:
                        value=e['facts'][truth[q['id']]['key']];break
                self.assertEqual(value,truth[q['id']]['answer'])

    def test_future_and_semantic_oracle_excluded_from_training_inputs(self):
        for inp,target in self.pairs:
            rows=list(supervision_records(inp,target))
            for row,q in zip(rows,inp['queries']):
                self.assertEqual(len(row['write_blocks']),q['after_block'])
                self.assertEqual(row['decision_boundaries'],[b for b in inp['decision_boundaries'] if b<=q['after_block']])
                self.assertNotIn('semantic_events',row);self.assertNotIn('family',row)
                self.assertNotIn('retention',row);self.assertNotIn('oracle_action',row)
                self.assertEqual(row['required_primary_readout'],'fresh_kv_preserve_fast_memory')

    def test_corrupt_truth_and_twin_rejected(self):
        bad=copy.deepcopy(self.pairs);bad[0][1]['answers'][0]['answer']='forged'
        with self.assertRaisesRegex(ValueError,'causal answer'):validate(bad)
        with self.assertRaisesRegex(ValueError,'missing twin'):validate(self.pairs[1:])

    def test_output_split_hashes_and_no_overwrite(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'data';m=materialize(73,{s:1 for s in SPLITS},p)
            self.assertEqual(len(m['files_sha256']),12)
            for name,sha in m['files_sha256'].items():self.assertEqual(hashlib.sha256((p/name).read_bytes()).hexdigest(),sha)
            with self.assertRaises(FileExistsError):materialize(73,{s:1 for s in SPLITS},p)
            for split in SPLITS:
                warm=[json.loads(s) for s in (p/f'{split}.warmup.jsonl').read_text().splitlines()]
                self.assertEqual(len(warm),8)


if __name__=='__main__':unittest.main()
