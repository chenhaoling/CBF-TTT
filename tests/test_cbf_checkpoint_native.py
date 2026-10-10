"""Preserve frozen records and exercise nonzero native writes plus query isolation."""
import copy
import unittest

from tasks.cbf_checkpoint_native import reblock, NativeSession, native_digest, score_difference


class ConstructionTests(unittest.TestCase):
    def test_exact_stream_records_and_queries(self):
        from tests.test_cbf_length_readout import LengthReadoutTests
        LengthReadoutTests.setUpClass(); old=LengthReadoutTests.rows
        rows=reblock(old); index={r['id']:r for r in old if r['segment_tokens']==2048}
        self.assertEqual(len(rows),64)
        for r in rows:
            src=index[r['id']]
            self.assertEqual([x for c in r['chunks'] for x in c],[x for c in src['chunks'] for x in c])
            self.assertEqual(r['queries'],src['queries']);self.assertEqual(r['choice_ids'],src['choice_ids'])
            for a,b in zip(r['records'],src['records']):
                count=len(src['chunks'][b['chunk']-1])-b['offset']
                self.assertEqual(r['chunks'][a['chunk']-1][a['offset']:a['offset']+count],src['chunks'][b['chunk']-1][b['offset']:])
            twin=next(x for x in rows if x['id']==r['twin_id'] and x['cell']==r['cell'])
            self.assertEqual(sum(a!=b for c,d in zip(r['chunks'],twin['chunks']) for a,b in zip(c,d)),1)
        with self.assertRaisesRegex(ValueError,'matrix'):reblock(old[:-1])

    def test_comparison_rejects_different_token_stream(self):
        r={'id':'a','cell':'b','context_sha256':'x','source_context_sha256':'old',
           'queries':{'target':{'binding':{'choice_nll':[0.,1.],'prediction':0,'greedy_id':4}}}}
        b=copy.deepcopy(r);b['context_sha256']='old'
        self.assertEqual(score_difference([r],[b],True)['max_choice_nll_error'],0.)
        with self.assertRaisesRegex(ValueError,'contexts'):score_difference([r],[b])


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'torch available on experiment server')
class NativeTests(unittest.TestCase):
    def model(self):
        from tests.test_ttt_train_infer import TrainInferTests
        return TrainInferTests().pair()[1].requires_grad_(False)

    def test_partial_buffer_nonzero_update_and_query_isolation(self):
        model=self.model();session=NativeSession(model)
        session._forward([1,2]);self.assertEqual(session.cache.ttt_states[0][0].shape[1],2)
        base=model.model.layers[0].mlp.down_proj.weight
        torch.testing.assert_close(session.cache.ttt_states[0][2],base)
        session._forward([3,4]);self.assertIsNone(session.cache.ttt_states[0][0])
        self.assertGreater(float((session.cache.ttt_states[0][2]-base).abs().max()),0.)
        before=native_digest(session.cache);branch=session.clone();w=branch.cache.ttt_states[0][2]
        branch._forward([5,6])
        self.assertIs(w,branch.cache.ttt_states[0][2]);self.assertEqual(native_digest(session.cache),before)
        branch._forward([7,8]);self.assertIsNot(w,branch.cache.ttt_states[0][2])
        self.assertEqual(native_digest(session.cache),before)
        self.assertNotEqual(native_digest(branch.cache),before)
        self.assertFalse(getattr(session.cache,'cbf_enabled',False))

    def test_zero_lr_plain_equivalence_fp32(self):
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from tasks.cbf_checkpoint_readout import PlainSession
        native=self.model();config=Qwen3Config(**native.config.to_dict());config.ttt_mode=False
        plain=Qwen3ForCausalLM(config).eval().requires_grad_(False)
        plain.load_state_dict({k:v for k,v in native.state_dict().items() if 'ttt_' not in k},strict=True)
        for layer in native.model.layers:layer.mlp.ttt_lr=0.
        a,b=NativeSession(native),PlainSession(plain)
        for ids in ([1,2,3,4],[5,6,7,8],[9,10]):
            torch.testing.assert_close(a._forward(ids),b._forward(ids),atol=2e-5,rtol=2e-5)


if __name__=='__main__':unittest.main()
