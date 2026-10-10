"""Matched example budgets, shared-memory gradients and exact intervention rollback."""
from collections import Counter
import contextlib
import unittest

from tasks.cbf_fact_interference import schedules,describe_probes
try:
    import torch
except ImportError:
    torch=None


class ScheduleTests(unittest.TestCase):
    def test_equal_exposures_and_grouping(self):
        rows=[{'split':split,'context_id':split+str(c),'id':split+str(c)+'.'+str(q)}
              for split in ('train','dev') for c in range(4) for q in range(4)]
        result=schedules(rows);lookup={r['id']:r for r in rows}
        self.assertEqual({k:len(v) for k,v in result.items()},
                         {'sequential':800,'joint_context':200,'mixed_context':200})
        for arm,batches in result.items():
            ids=[rid for b in batches for rid in b['ids']]
            self.assertEqual(Counter(ids),Counter({r['id']:50 for r in rows if r['split']=='train'}))
            for b in batches:
                contexts={lookup[rid]['context_id'] for rid in b['ids']}
                self.assertEqual(len(contexts),4 if arm=='mixed_context' else 1)
        self.assertEqual([rid for b in result['sequential'] for rid in b['ids']],
                         [rid for b in result['joint_context'] for rid in b['ids']])

    def test_interference_requires_own_improvement(self):
        before=[{'digits':1.} for _ in range(4)]
        matrix=[[-.1,.2,0,0],[.2,.1,0,0],[0,0,-.2,0],[0,0,0,-.1]]
        r={'context_id':'c','ids':['a','b','c','d'],'before':before,
           'single_trials':[{'after':[{'digits':1+x} for x in row]} for row in matrix],
           'joint_trial':{'after':[{'digits':.9} for _ in range(4)]},
           'gradient_cosines':[[1. if i==j else -.2 for j in range(4)] for i in range(4)]}
        s=describe_probes([r]);self.assertEqual(s['interference_events'],1)
        self.assertEqual(s['harmed_other_facts'],2);self.assertEqual(s['negative_gradient_pairs'],6)
        self.assertEqual(s['joint_harmed_facts'],0)


@unittest.skipIf(torch is None,'torch tests run on experiment server')
class InterferenceTests(unittest.TestCase):
    def fixture(self):
        from tests.test_cbf_event_writer import WriterTests
        model=WriterTests().model()
        with torch.no_grad():
            for layer in model.model.layers:layer.mlp.ttt_conv.weight.normal_(0,.001)
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        rows=[{'id':str(i),'context_id':'c','context_ids':[1,2,3,4],
               'query_ids':[6,7+i],'answer_ids':[12,13,14+i],
               'digit_positions':[1,2],'prefix_positions':[0]} for i in range(4)]
        return model,params,rows

    def test_shared_write_matches_independent_mean_gradient(self):
        from tasks.cbf_fact_interference import batch_loss
        model,params,rows=self.fixture()
        batch_loss(model,rows,1)[0].backward();expected={n:p.grad.clone() for n,p in params.items()}
        model.zero_grad(set_to_none=True)
        for r in rows:(batch_loss(model,[r],1)[0]/4).backward()
        for n,p in params.items():torch.testing.assert_close(p.grad,expected[n],rtol=1e-5,atol=1e-7)

    def test_adam_interventions_restore_weights_moments_and_readout(self):
        from tasks.cbf_fact_interference import batch_loss,probe_context,cpu_copy,equal_state
        from cbf_ttt.event_writer import backbone_digest
        model,params,rows=self.fixture()
        # Tiny random MLPs otherwise hide a genuine parameter change below FP32 loss resolution.
        # Amplify only this CPU fixture; the real experiment keeps its frozen base and lr1e-7.
        with torch.no_grad():
            model.lm_head.weight.mul_(100.)
            for layer in model.model.layers:
                layer.mlp.ttt_conv.weight.mul_(100.)
                layer.mlp.ttt_proj.weight.mul_(20.)
        opt=torch.optim.AdamW(params.values(),lr=1e-3,weight_decay=0.)
        batch_loss(model,rows,1)[0].backward();opt.step();opt.zero_grad(set_to_none=True)
        weights=cpu_copy(params);state=cpu_copy(opt.state_dict());digest=backbone_digest(model)
        rec=probe_context(model,params,opt,rows,1,contextlib.nullcontext,
                          measure_fn=lambda f:(f(),{'seconds':0.,'peak_allocated_gib':0.,'peak_reserved_gib':0.}))
        self.assertTrue(equal_state(weights,cpu_copy(params)));self.assertTrue(equal_state(state,opt.state_dict()))
        self.assertEqual(digest,backbone_digest(model));self.assertEqual(len(rec['single_trials']),4)
        self.assertLessEqual(rec['restored_baseline_max_abs'],1e-6)
        self.assertTrue(any(a['ce']!=b['ce'] for t in rec['single_trials'] for a,b in zip(t['after'],rec['before'])))
        for i in range(4):self.assertAlmostEqual(rec['gradient_cosines'][i][i],1.,places=5)


if __name__=='__main__':unittest.main()
