"""New writer gradients, native update parity, answer alignment and packed boundaries."""
import tempfile
import unittest
from pathlib import Path

from tasks.pack_cbf_event_warmup import encode_answer,pack


class CharacterTokenizer:
    eos_token_id=1
    def encode(self,text,add_special_tokens=False):return [ord(c)+2 for c in text]
    def decode(self,ids):return ''.join(chr(i-2) for i in ids)


class PackingTests(unittest.TestCase):
    def test_answer_boundary_and_disjoint_fixed_blocks(self):
        from tasks.build_cbf_event_curriculum import materialize
        import json
        tok=CharacterTokenizer()
        q,a=encode_answer(tok,'Question?','code_12345')
        self.assertEqual(tok.decode(q+a),'Question?\nAnswer: code_12345')
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';out=Path(temp)/'packed'
            materialize(20261011,{'train':8,'dev':2,'test':2},source)
            # Packing must not require access to any test examples.
            for p in source.glob('test.*.jsonl'):p.unlink()
            m=pack(source,out,tok,str(source))
            rows=list(map(json.loads,(out/'rows.jsonl').read_text().splitlines()))
            self.assertEqual(len(rows),80);self.assertFalse(m['test_tokenized'])
            for r in rows:
                self.assertEqual(len(r['context_ids']),4096)
                self.assertIn(r['answer'],tok.decode(r['context_ids']))
                self.assertNotIn(r['answer'],tok.decode(r['query_ids']))
                self.assertNotEqual(r['group_id'],r['wrong_context_id'].split('.')[0])
                self.assertEqual(r['answer_start'],len(r['query_ids'])-1)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'torch tests run on experiment server')
class WriterTests(unittest.TestCase):
    def model(self):
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from cbf_ttt.event_writer import trainable_writer
        torch.manual_seed(301)
        config=Qwen3Config(vocab_size=32,hidden_size=8,intermediate_size=16,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=2,head_dim=4,max_position_embeddings=64,
            ttt_mode=True,ttt_layers=[0,1],ttt_chunk=4,ttt_lr=.3,ttt_proj=True)
        config._attn_implementation='eager'
        model=Qwen3ForCausalLM(config).eval();trainable_writer(model)
        return model

    def test_zero_initial_memory_and_live_writer_gradient(self):
        from cbf_ttt.event_writer import write_memory,answer_loss,backbone_digest
        model=self.model();before=backbone_digest(model)
        mem=write_memory(model,[1,2,3,4]);self.assertTrue(all(v.requires_grad for v in mem.values()))
        self.assertTrue(all(float(v.abs().sum())==0 for v in mem.values()))
        row={'query_ids':[6,7],'answer_ids':[8,9]}
        loss,_,_=answer_loss(model,mem,row,1);loss.backward()
        for n,p in model.named_parameters():
            if '.ttt_conv.' in n:self.assertGreater(float(p.grad.norm()),0.)
            elif '.ttt_proj.' in n:self.assertEqual(float(p.grad.norm()),0.)
            else:self.assertIsNone(p.grad)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        opt.step();opt.zero_grad(set_to_none=True)
        mem=write_memory(model,[1,2,3,4]);answer_loss(model,mem,row,1)[0].backward()
        for p in model.parameters():
            if p.requires_grad:self.assertGreater(float(p.grad.norm()),0.)
        self.assertEqual(backbone_digest(model),before)

    def test_nonzero_native_write_parity_and_fresh_kv(self):
        from cbf_ttt.event_writer import write_memory,cache_for,forward
        from inference_model.hf_qwen3.modeling_qwen3 import TTTDynamicCache
        model=self.model()
        with torch.no_grad():
            for l in model.model.layers:l.mlp.ttt_conv.weight.normal_(0,.02)
            memory=write_memory(model,[1,2,3,4]);native=TTTDynamicCache(config=model.config)
            forward(model,[1,2,3,4],native)
            fresh=TTTDynamicCache(config=model.config)
            for i in model.config.ttt_layers:
                base=model.model.layers[i].mlp.down_proj.weight
                torch.testing.assert_close(native.ttt_states[i][2]-base,memory[i],atol=1e-6,rtol=1e-5)
                fresh.ttt_states[i]=(None,None,base+memory[i])
            read=cache_for(model,memory);self.assertEqual(read.get_seq_length(),0)
            a,b=forward(model,[5,6],read),forward(model,[5,6],fresh)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            self.assertEqual(read.get_seq_length(),2)

    def test_answer_positions_include_terminal_only_after_answer(self):
        from cbf_ttt.event_writer import answer_logits,cache_for,forward
        model=self.model();q=[2,3,4];answer=[5,6,1]
        a=answer_logits(model,{},q,answer)
        b=model.lm_head(forward(model,q+answer[:-1],cache_for(model)))[:,2:].reshape(3,-1)
        torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
