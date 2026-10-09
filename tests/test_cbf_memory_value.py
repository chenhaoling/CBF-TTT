"""Protocol, causal state isolation and negative-gate regressions."""
import copy
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

from tasks.build_cbf_memory_value import make_scenes, validate, domain, REGIMES
from tasks.cbf_memory_value import q_summary, value_metrics, read_results, digest


class Tokenizer:
    def __init__(self): self.words={}
    def encode(self,text,add_special_tokens=False):
        result=[]
        for word in re.findall(r'\w+|[^\w\s]',text):
            if word not in self.words: self.words[word]=len(self.words)+1
            result.append(self.words[word])
        return result


class ProtocolTests(unittest.TestCase):
    def scenes(self):
        groups=[{'domain':domain(g),'sources':[f'source-{g}'],
                 'chunks':[[1000+g*10000+c*100+i for i in range(96)] for c in range(6)]} for g in range(24)]
        return make_scenes(groups,Tokenizer(),96)

    def test_balanced_paired_truth_and_correction(self):
        rows=self.scenes();validate(rows)
        self.assertEqual(len(rows),192)
        for r in rows:
            if r['regime']=='correction': self.assertNotEqual(r['old_label'],r['queries']['target']['label'])
        bad=copy.deepcopy(rows);bad[8]['sources']=bad[0]['sources']
        with self.assertRaisesRegex(ValueError,'source'): validate(bad)
        bad=copy.deepcopy(rows);bad[0]['chunks'][0][0]+=1
        with self.assertRaisesRegex(ValueError,'one token'): validate(bad)
        bad=copy.deepcopy(rows);bad[0]['queries']['target']['answer_id']=-1
        with self.assertRaisesRegex(ValueError,'answer'): validate(bad)

    def test_readability_is_per_regime_and_question(self):
        rows=[{'regime':regime,'queries':{'full_kv':{'self':{'target':{'correct':1},'anchor':{'correct':1}}}}} for regime in REGIMES]
        self.assertTrue(q_summary(rows)['passed'])
        rows[2]['queries']['full_kv']['self']['anchor']['correct']=0
        self.assertFalse(q_summary(rows)['passed'])

    def test_nll_only_gain_does_not_pass_content_gate(self):
        rows=[]
        for g in range(8):
            for variant in (0,1):
                for policy in ('none','retain'):
                    good={'nll':1.,'correct':.125,'greedy_correct':0}
                    neutral={'nll':3.,'correct':.125,'greedy_correct':0}
                    interventions={'self':{'target':neutral if policy=='none' else good}}
                    if policy!='none':interventions.update(zero={'target':neutral},twin={'target':good})
                    rows.append({'id':f'{g}_{variant}','group':g,'domain':domain(g),'regime':'stable','policy':policy,'queries':{'memory_only':interventions}})
        result=value_metrics(rows,'memory_only','retain')
        self.assertGreater(result['means']['B_none'],1.)
        self.assertFalse(result['passed'])
        self.assertEqual(result['means']['B_twin'],0.)

    def test_missing_results_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'rows.jsonl';p.write_text('')
            Path(str(p)+'.audit.json').write_text(json.dumps({'rows':0,'backbone_unchanged':True,'rows_sha256':digest(b'')}))
            with self.assertRaisesRegex(ValueError,'incomplete'):read_results([str(p)],self.scenes(),'q')


class StateTests(unittest.TestCase):
    def test_scoring_interventions_isolate_parent_and_keep_kv(self):
        import torch
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from tasks.cbf_memory_value import build_session, score_query, state_digest, move_cache
        torch.manual_seed(208)
        config=Qwen3Config(vocab_size=32,hidden_size=8,intermediate_size=16,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,head_dim=4,
            max_position_embeddings=128,ttt_mode=True,ttt_layers=[0,1],ttt_chunk=4,ttt_lr=.1,ttt_proj=True)
        model=Qwen3ForCausalLM(config).eval().requires_grad_(False)
        with torch.no_grad():
            for layer in model.model.layers:layer.mlp.ttt_conv.weight.normal_(0,.02)
        scene={'chunks':[[1,2,3,4]]*6};query={'ids':[5,6],'label':0,'answer_id':7};choices=list(range(7,15))
        a=build_session(model,scene,'retain');b=build_session(model,scene,'none')
        before=state_digest(a.cache)
        score_query(a,query,choices,'full_kv',a.cache.cbf_memory)
        self.assertEqual(before,state_digest(a.cache))
        x=score_query(a,query,choices,'memory_only',{})
        y=score_query(b,query,choices,'memory_only',{})
        self.assertEqual(x,y)
        z=score_query(a,query,choices,'full_kv',{})
        self.assertTrue(abs(z['nll']-x['nll'])>1e-8)
        move_cache(a.cache,'cpu');self.assertEqual(before,state_digest(a.cache))
        self.assertEqual(a.cache.get_seq_length(),24)
        self.assertEqual(a.cache.cbf_candidates,{})


if __name__=='__main__':unittest.main()
