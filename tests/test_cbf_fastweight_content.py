"""Construction, causal schedule and statistics tests for fastweight content."""
import copy
import json
import unittest

from tasks.build_cbf_memory_value import digest
from tasks.cbf_fastweight_content import (Q_GROUPS,V_GROUPS,aggregate_v,intervention_weights,
    paired,rollout_one_write,validate_scenes)


def scenes():
    rows=[]
    for group in Q_GROUPS+V_GROUPS:
        domain='fineweb' if group%8 in (0,1,6,7) else 'longcrawl'
        for variant in (0,1):
            label=(group%8+4*variant)%8;chunks=[[0]*4096,[0]*4095+[label+1]]
            rows.append({'id':f'g{group}_{variant}','twin_id':f'g{group}_{1-variant}','group':group,
                'split':'q' if group in Q_GROUPS else 'v','domain':domain,'variant':variant,'sources':[f's{group}'],
                'choice_ids':list(range(10,18)),'chunks':chunks,'context_sha256':digest(json.dumps(chunks).encode()),
                'queries':{'target':{'qa':{'ids':[2,3],'label':label,'answer_id':10+label}}}})
    return rows


class ConstructionTests(unittest.TestCase):
    def test_balanced_disjoint_twins(self):
        rows=scenes();validate_scenes(rows)
        broken=copy.deepcopy(rows);broken[0]['chunks'][0][0]=1;broken[0]['context_sha256']=digest(json.dumps(broken[0]['chunks']).encode())
        with self.assertRaisesRegex(ValueError,'twin token'):validate_scenes(broken)

    def test_policy_aggregation_and_pairing(self):
        rows=[]
        for group in V_GROUPS:
            for variant in (0,1):
                result={}
                for name,value in [('correct',1.),('wrong',2.),('zero',3.),('random0',4.),('random1',4.),('random2',4.)]:
                    result[name]={'own_nll':value,'twin_nll':value+1,'own_vs_twin_margin':1.,'correct':1.,'greedy_correct':1.,'eos_greedy':0.}
                rows.append({'group':group,'results':result})
        self.assertEqual(aggregate_v(rows,'random_mean')['own_nll'],4.)
        value=paired(rows,'wrong','correct','own_nll',samples=200,seed=1)
        self.assertEqual(value['mean'],1.);self.assertEqual(value['ci95'],[1.,1.])


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'torch available on experiment server')
class NativeTests(unittest.TestCase):
    def model(self):
        from tests.test_ttt_train_infer import TrainInferTests
        return TrainInferTests().pair()[1].requires_grad_(False)

    def test_one_write_and_random_norm(self):
        model=self.model();c=model.config.ttt_chunk;row={'chunks':[[1,2,3,4][:c],[4,3,2,1][:c]]}
        session,trace=rollout_one_write(model,row)
        self.assertTrue(all(v==0 for v in trace[0].values()));self.assertTrue(any(v>0 for v in trace[1].values()))
        parent=[tuple(None if x is None else x.clone() for x in state) for state in session.cache.ttt_states]
        _,ratios=intervention_weights(session,'random0',seed=9)
        self.assertTrue(all(abs(v-1)<1e-6 for v in ratios.values()))
        for a,b in zip(parent,session.cache.ttt_states):
            for x,y in zip(a,b):
                if x is not None:torch.testing.assert_close(x,y)


if __name__=='__main__':unittest.main()
