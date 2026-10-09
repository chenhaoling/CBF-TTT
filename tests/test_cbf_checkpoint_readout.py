"""Reject incomplete loading, non-comparable paths and mislabeled checkpoint scores."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
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

    def test_complete_summary_and_native_weight_mismatch(self):
        from tasks.cbf_checkpoint_readout import summarize, ARMS
        from tasks.cbf_disjoint_readout import IDENTITY
        from tasks.build_cbf_memory_value import file_digest
        from tests.test_cbf_disjoint_readout import DisjointReadoutTests
        DisjointReadoutTests.setUpClass();scenes=DisjointReadoutTests.rows
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir();source=Path(tmp)/'source';(source/'data').mkdir(parents=True)
            data=source/'data/scenes.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in scenes))
            sha=file_digest(data);(source/'summary.json').write_text('{}')
            config={k:1 for k in ('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads',
                'num_key_value_heads','head_dim','vocab_size','rope_theta','rope_scaling','rms_norm_eps',
                'tie_word_embeddings','max_position_embeddings','sliding_window','attention_bias','hidden_act')}
            profile=dict(seconds=1.,peak_allocated_gib=1.,peak_reserved_gib=2.)
            for arm in ARMS:
                for shard in (0,1):
                    rows=[]
                    for r in scenes:
                        if (r['group']//2+r['group']%2)%2!=shard or (arm=='original_native' and not r['is_bridge']):continue
                        queries={}
                        for name,fs in r['queries'].items():
                            queries[name]={}
                            for fmt,q in fs.items():
                                scores=[2.]*8;scores[q['label']]=1.
                                queries[name][fmt]=dict(choice_nll=scores,nll=1.,prediction=q['label'],correct=1,
                                    greedy_id=q['answer_id'],greedy_correct=1,**profile)
                        item={k:r[k] for k in IDENTITY};item.update(queries=queries,rollout=profile,arm=arm,
                            model='final' if arm=='final_repo' else 'original',weights_sha256='final' if arm=='final_repo' else 'original',
                            parent_unchanged=True,data_sha256=sha);rows.append(item)
                    p=root/f'{arm}_{shard}.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    adapters=['model.layers.0.mlp.ttt_proj.weight'] if arm=='final_repo' else []
                    a=dict(rows=len(rows),rows_sha256=file_digest(p),backbone_unchanged=True,arm=arm,shard=shard,
                        path=rows[0]['model'],weights_sha256=rows[0]['weights_sha256'],config=config,
                        source_files_sha256={},loading_info={'unexpected_keys':adapters},excluded_adapters=adapters)
                    a['class']='test';Path(str(p)+'.audit.json').write_text(json.dumps(a))
                    if arm=='final_repo':
                        ref=source/f'rows_{shard}.jsonl';ref.write_bytes(p.read_bytes())
                        Path(str(ref)+'.audit.json').write_text(json.dumps(a))
            with patch('tasks.cbf_checkpoint_readout.DATA_SHA',sha):
                result=summarize(root,source)
                self.assertEqual(result['bridges']['final_vs_previous_cbf']['queries'],288)
                self.assertEqual(result['bridges']['original_native_vs_repo']['queries'],32)
                self.assertTrue(result['models']['original_repo']['selection']['passed'])
                # Internally consistent native shards still cannot use different weights.
                for shard in (0,1):
                    p=root/f'original_native_{shard}.jsonl'
                    rows=list(map(json.loads,p.read_text().splitlines()))
                    for r in rows:r['weights_sha256']='wrong'
                    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    ap=Path(str(p)+'.audit.json');a=json.loads(ap.read_text());a.update(weights_sha256='wrong',rows_sha256=file_digest(p));ap.write_text(json.dumps(a))
                with self.assertRaisesRegex(ValueError,'native weights differ'):summarize(root,source)


if __name__=='__main__':unittest.main()
