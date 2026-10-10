"""Guard causal pair selection, numeric masks, and differentiable paired targets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tests.test_cbf_event_writer import CharacterTokenizer, torch
from tasks.train_cbf_event_content import select_rows, training_order, split_answer


class ContentDataTests(unittest.TestCase):
    def test_selected_worlds_pairs_and_masks(self):
        from tasks.build_cbf_event_curriculum import materialize
        from tasks.pack_cbf_event_warmup import pack
        tok=CharacterTokenizer()
        with tempfile.TemporaryDirectory() as t:
            source=Path(t)/'source';packed=Path(t)/'packed'
            materialize(20261011,{'train':8,'dev':2,'test':2},source)
            for p in source.glob('test.*.jsonl'):p.unlink()
            pack(source,packed,tok,str(source))
            original=list(map(json.loads,(packed/'rows.jsonl').read_text().splitlines()))
            untouched=copy.deepcopy(original)
            rows,groups=select_rows(original,tok)
            self.assertEqual(original,untouched);self.assertEqual(len(rows),32)
            order=training_order(rows)
            self.assertEqual(len(order),400)
            self.assertTrue(all(order.count(rid)==50 for rid in set(order)))
            byid={r['id']:r for r in rows}
            for rid in order:
                self.assertEqual(byid[rid]['split'],'train')
            for r in rows:
                self.assertEqual(tok.decode([r['answer_ids'][i] for i in r['digit_positions']]),r['answer'][5:])
                self.assertNotEqual(r['context_id'].split('.')[0],r['wrong_context_id'].split('.')[0])
                self.assertEqual(r['answer']!=byid[r['twin_row_id']]['answer'],r['anchor_query'])
            # Corrupt a selected row explicitly, independent of original order.
            bad=copy.deepcopy(original)
            r=next(r for r in bad if r['id']==rows[0]['id']);r['anchor_query']=not r['anchor_query']
            with self.assertRaises(ValueError):select_rows(bad,tok)

    def test_reject_non_numeric_or_partial_answer(self):
        tok=CharacterTokenizer()
        for answer in ('code_1234','code_abcde'):
            with self.assertRaises(ValueError):split_answer(tok,{'answer':answer,'answer_ids':tok.encode(' '+answer)})


@unittest.skipIf(torch is None,'torch tests run on experiment server')
class ContentGradientTests(unittest.TestCase):
    def test_contrast_gradient_and_unchanged_fact_mask(self):
        from tasks.train_cbf_event_content import objective
        own=[{k:torch.tensor(1.,requires_grad=True) for k in ('ce','digits','prefix','eos')} for _ in range(2)]
        cross=[{'digits':torch.tensor(1.,requires_grad=True)} for _ in range(2)]
        loss,_=objective(own,cross,True,'content_pair');loss.backward()
        self.assertTrue(all(v['digits'].grad<0 for v in cross))
        self.assertTrue(all(v['digits'].grad>0 for v in own))
        for v in cross:v['digits'].grad=None
        loss,_=objective(own,cross,False,'content_pair');loss.backward()
        self.assertTrue(all(v['digits'].grad==0 for v in cross))

    def test_live_two_memory_pair_reaches_native_writer(self):
        from cbf_ttt.event_writer import write_memory,backbone_digest
        from tasks.train_cbf_event_content import regions,objective
        from tests.test_cbf_event_writer import WriterTests
        model=WriterTests().model()
        with torch.no_grad():
            for l in model.model.layers:l.mlp.ttt_conv.weight.normal_(0,.01)
        before=backbone_digest(model)
        rows=[{'query_ids':[7,8],'answer_ids':[9,10,i],'digit_positions':[1,2],'prefix_positions':[0]} for i in (11,12)]
        mem=[write_memory(model,ids) for ids in ([1,2,3,4],[1,2,3,5])]
        own=[regions(model,m,r,1) for m,r in zip(mem,rows)]
        cross=[regions(model,m,r,1) for m,r in zip(reversed(mem),rows)]
        loss,_=objective(own,cross,True,'content_pair');loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0.)
            else:self.assertIsNone(p.grad)
        self.assertEqual(before,backbone_digest(model))


if __name__=='__main__':unittest.main()
