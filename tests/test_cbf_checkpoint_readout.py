"""Reject incomplete loading, non-comparable paths and mislabeled checkpoint scores."""
import copy
import unittest
from tasks.cbf_checkpoint_readout import check_loading,compare_scores,validate_scores


class CheckpointReadoutTests(unittest.TestCase):
    def test_loading_only_allows_expected_ttt_parameters(self):
        key='model.layers.0.mlp.ttt_conv.weight'
        check_loading({'unexpected_keys':[key]},[key])
        check_loading({},[])
        for info,expected in (({'missing_keys':['model.embed_tokens.weight']},[]),
                              ({'unexpected_keys':['lm_head.weight']},['lm_head.weight']),
                              ({'unexpected_keys':[]},[key]),
                              ({'mismatched_keys':['model.norm.weight']},[])):
            with self.assertRaises(ValueError):check_loading(info,expected)

    def fixture(self):
        scene=dict(id='a',cell='clean/stable',group=0,domain='fineweb',variant=0,regime='stable',is_bridge=False,context_sha256='x',
                   queries={'target':{'qa':{'label':0,'answer_id':99}}})
        profile=dict(seconds=1.,peak_allocated_gib=1.,peak_reserved_gib=2.)
        row={k:v for k,v in scene.items() if k!='queries'}
        row.update(data_sha256='data',parent_unchanged=True,rollout=profile,
            queries={'target':{'qa':dict(choice_nll=[1.]+[2.]*7,nll=1.,prediction=0,correct=1,greedy_id=99,greedy_correct=1,**profile)}})
        return scene,row

    def test_score_validation_rejects_tampering_and_duplicates(self):
        scene,row=self.fixture();validate_scores([row],[scene],'data')
        for field,value in (('nll',4.),('greedy_correct',0),('choice_nll',[1.]),('seconds',float('nan'))):
            bad=copy.deepcopy(row);bad['queries']['target']['qa'][field]=value
            with self.assertRaises(ValueError):validate_scores([bad],[scene],'data')
        with self.assertRaises(ValueError):validate_scores([row,row],[scene],'data')

    def test_bridge_checks_all_candidates_and_predictions(self):
        _,row=self.fixture();self.assertEqual(compare_scores([row],[row])['queries'],1)
        bad=copy.deepcopy(row);bad['queries']['target']['qa']['choice_nll'][7]+=.01
        with self.assertRaisesRegex(ValueError,'NLL'):compare_scores([bad],[row])
        bad=copy.deepcopy(row);bad['queries']['target']['qa']['greedy_id']=98
        with self.assertRaisesRegex(ValueError,'predictions'):compare_scores([bad],[row])


if __name__=='__main__':unittest.main()
