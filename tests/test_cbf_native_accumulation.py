"""Tests for native fast-weight write schedules and paired inference summaries."""
import copy
import unittest

from tasks.cbf_native_accumulation import aggregate, paired_bootstrap, rollout


def fake_rows(offset):
    rows=[]
    for group in range(8):
        query={"binding":{"answer_nll":group+offset,"eos_nll":group+offset+1,
                          "answer_eos_margin":1.,"correct":1.,"greedy_correct":1.,"eos_greedy":0.},
               "qa":{"answer_nll":group+offset,"eos_nll":group+offset+1,
                     "answer_eos_margin":1.,"correct":1.,"greedy_correct":1.,"eos_greedy":0.}}
        rows.append({"group":group,"queries":{"target":copy.deepcopy(query),"anchor":copy.deepcopy(query)}})
    return rows


class SummaryTests(unittest.TestCase):
    def test_aggregate_and_group_bootstrap(self):
        low,high=fake_rows(0.),fake_rows(2.)
        self.assertEqual(aggregate(low)['answer_eos_margin'],1.)
        result=paired_bootstrap(high,low,'answer_nll',samples=200,seed=3)
        self.assertEqual(result['mean'],2.)
        self.assertEqual(result['ci95'],[2.,2.])


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'torch available on experiment server')
class PolicyTests(unittest.TestCase):
    def model(self):
        from tests.test_ttt_train_infer import TrainInferTests
        return TrainInferTests().pair()[1].requires_grad_(False)

    def scene(self,model):
        c=model.config.ttt_chunk
        return {'chunks':[[1,2,3,4][:c] for _ in range(3)]}

    def test_zero_first_last_and_reset_schedules(self):
        model=self.model(); scene=self.scene(model)
        zero,ztrace=rollout(model,scene,'zero')
        self.assertTrue(all(v==0 for t in ztrace for v in t.values()))
        first,ftrace=rollout(model,scene,'first_only')
        self.assertTrue(any(v>0 for v in ftrace[0].values()))
        self.assertEqual(ftrace[0],ftrace[1]);self.assertEqual(ftrace[1],ftrace[2])
        last,ltrace=rollout(model,scene,'last_only')
        self.assertTrue(all(v==0 for t in ltrace[:2] for v in t.values()))
        self.assertTrue(any(v>0 for v in ltrace[2].values()))
        reset,rtrace=rollout(model,scene,'reset_each')
        cumulative,ctrace=rollout(model,scene,'cumulative')
        self.assertEqual(reset.cache.get_seq_length(),cumulative.cache.get_seq_length())
        self.assertTrue(any(abs(rtrace[-1][k]-ctrace[-1][k])>1e-8 for k in rtrace[-1]))
        self.assertEqual(zero.cache.get_seq_length(),first.cache.get_seq_length())


if __name__=='__main__':unittest.main()

