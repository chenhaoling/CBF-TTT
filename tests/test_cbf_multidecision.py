"""Validate nested timing, fixed search budgets, causal rollouts, and completeness."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tasks.cbf_multidecision import GROUPS, KEEP, decode_plan, group_design, plan_id, rollout, summarize


class MultiDecisionTests(unittest.TestCase):
    def test_design_is_deterministic_nested_and_budget_matched(self):
        for group in GROUPS:
            d=group_design(group)
            self.assertEqual(d,group_design(group))
            for k in (1,2,3):
                points=d['schedules'][str(k)]
                f=d['families'][str(k)]
                self.assertEqual(len(points),k)
                self.assertIn(4,points)
                self.assertEqual(len(set(f['global_exact'])),3**k)
                self.assertEqual(len(set(f['local_sample'])),27)
                if k>1:
                    self.assertTrue(set(d['schedules'][str(k-1)]).issubset(points))
                    self.assertTrue(set(d['families'][str(k-1)]['global_exact']).issubset(f['global_exact']))
                    self.assertEqual(len(f['local_matched']),3**k)
                    self.assertTrue(any(len({decode_plan(p)[i] for i in points})>1 for p in f['local_matched']))
                for name in f['union']:
                    self.assertEqual(plan_id(decode_plan(name)),name)
                    self.assertTrue(all(a==KEEP for point,a in decode_plan(name).items() if point not in points))

    def test_pending_prefix_rollout_matches_full_causal_execution(self):
        import torch
        from cbf_ttt.selective import CohortSession
        from tests import test_cbf_memory_stability
        model=test_cbf_memory_stability.StabilityTests().model()
        scene={'prefix':[[1,2,3,4]]*4,'future':[{'ids':[5,6,7,8],'query_ids':[9],'answer_ids':[10,11]} for _ in range(3)]}
        root=CohortSession(model)
        for chunk in scene['prefix'][:-1]: root.advance(chunk)
        root._forward(scene['prefix'][-1],collect=True)
        memory={i:x.clone() for i,x in root.cache.cbf_memory.items()}
        for actions in ({4:(.5,)*3},{4:(0.,)*3,5:(1.,1.,0.),7:(0.,1.,1.)}):
            name=plan_id(actions)
            obtained=rollout(root.clone(),scene,name)
            direct=CohortSession(model); expected=[]
            for point,chunk in enumerate(scene['prefix']+[f['ids'] for f in scene['future']],1):
                direct.advance(chunk,actions.get(point,KEEP))
                if point>=5:
                    future=scene['future'][point-5]
                    expected.append(direct.score_answer(future['query_ids'],future['answer_ids']))
            self.assertEqual(obtained,expected)
            self.assertTrue(all(torch.equal(x,root.cache.cbf_memory[i]) for i,x in memory.items()))
            self.assertEqual(root.cache.get_seq_length(),16)
            self.assertEqual(set(root.cache.cbf_candidates),set(root.layers))

    def test_summary_rejects_missing_plans(self):
        from tasks.build_cbf_selective import REGIMES
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            design={'groups':{str(g):group_design(g) for g in GROUPS},'data_sha256':'test',
                    'scene_ids':[f'g{g}_{r}' for g in GROUPS for r in REGIMES]}
            rows=[]
            for g in GROUPS:
                for regime in REGIMES:
                    value={'losses':[2.,2.,2.],'nll':2.,'seconds':1.,'peak_allocated_gib':1.,'peak_reserved_gib':1.,'reused':False}
                    rows.append({'id':f'g{g}_{regime}','group':g,'regime':regime,'split':'pilot' if g<4 else 'confirm',
                                 'model':'tiny','data_sha256':'test','results':{n:value for n in design['groups'][str(g)]['all_plans']},
                                 'controls':{n:value for n in ('global_half','window2','window3','clear','none')},
                                 'reference_checks':[{'max_error':0.}]*3,'clear_reference_max_error':0.,'smoke':False,'scene_seconds':1.})
            (p/'design.json').write_text(json.dumps(design))
            f=p/'rows.jsonl'; f.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            Path(str(f)+'.audit.json').write_text(json.dumps({'backbone_unchanged':True}))
            args=SimpleNamespace(design=str(p/'design.json'),inputs=[str(f)],output=str(p/'summary.json'))
            summarize(args)
            report=json.loads((p/'summary.json').read_text())
            self.assertFalse(report['temporal_signal'])
            self.assertFalse(report['selective_signal_beyond_controls'])
            rows[0]['results'].pop(next(iter(rows[0]['results'])))
            f.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError): summarize(args)


if __name__=='__main__': unittest.main()
