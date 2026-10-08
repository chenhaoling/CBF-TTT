"""Cohort provenance, baseline equivalence, causal data, and stage-gate checks."""

import copy
import hashlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from cbf_ttt.runtime import CBFSession
from cbf_ttt.selective import CohortSession, GRID, FIXED
from tasks.build_cbf_selective import make_scenes
from tasks.cbf_selective import action_name, summarize


class SelectiveTests(unittest.TestCase):
    def fake_session(self, dtype=torch.float32):
        session = object.__new__(CohortSession)
        session.layers = (0,)
        session.cache = SimpleNamespace(cbf_memory={}, cbf_candidates={}, cbf_collect=True)
        session.recent1, session.recent2 = {}, {}
        return session

    def test_uniform_matches_original_including_bf16(self):
        for dtype, retention in itertools.product((torch.float32, torch.bfloat16), (0., .5, 1.)):
            local = self.fake_session()
            baseline = copy.deepcopy(local)
            for delta in (1.13, -3.14, .015, 9.876, 1.004, .0021):
                for session in (local, baseline):
                    session.cache.cbf_candidates = {0: torch.tensor([[delta]], dtype=dtype)}
                local.commit_cohorts((retention,)*3)
                CBFSession.commit_both(baseline, retention, 1.)
                self.assertTrue(torch.equal(local.cache.cbf_memory[0], baseline.cache.cbf_memory[0]))

    def test_retained_cohorts_and_rotation(self):
        session = self.fake_session()
        contributions = []
        for delta, action in zip((1., 2., 3., 4., 5., 6.),
                                ((1.,)*3, (1.,)*3, (1.,)*3, (.5, 0., 1.), (0., .5, 1.), (1., 1., 0.))):
            ah, a2, a1 = action
            contributions = [value * (a1 if i == len(contributions)-1 else
                                      a2 if i == len(contributions)-2 else ah)
                             for i, value in enumerate(contributions)] + [delta]
            session.cache.cbf_candidates = {0: torch.tensor([[delta]])}
            session.commit_cohorts(action)
            self.assertAlmostEqual(session.cache.cbf_memory[0].item(), sum(contributions))
            self.assertAlmostEqual(session.recent1[0].item(), contributions[-1])
            if len(contributions) > 1:
                self.assertAlmostEqual(session.recent2[0].item(), contributions[-2])
        with self.assertRaises(RuntimeError):
            session.commit_cohorts((1.,)*3)
        with self.assertRaises(ValueError):
            session.commit_both(1., 0.)

    def test_model_branch_clone_and_readonly_query(self):
        from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
        from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
        torch.manual_seed(108)
        config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                             head_dim=4, max_position_embeddings=64, ttt_mode=True,
                             ttt_layers=[0, 1], ttt_chunk=4, ttt_lr=.1, ttt_proj=True)
        model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        for layer in model.model.layers:
            with torch.no_grad():
                layer.mlp.ttt_conv.weight.normal_(0, .02)
        session, baseline = CohortSession(model), CBFSession(model)
        for _ in range(4):
            session.advance([1, 2, 3, 4])
            baseline._forward([1, 2, 3, 4], collect=True)
            baseline.commit_both(1., 1.)
        for i in session.layers:
            self.assertTrue(torch.equal(session.cache.cbf_memory[i], baseline.cache.cbf_memory[i]))
        self.assertEqual(session.score_answer([5], [6, 7]), baseline.score_answer([5], [6, 7]))
        memory = {i: x.clone() for i, x in session.cache.cbf_memory.items()}
        seq_len = session.cache.get_seq_length()
        branch = session.clone()
        branch.advance([8, 9, 10, 11], (0., 1., 0.))
        for i in session.layers:
            self.assertNotEqual(branch.recent1[i].data_ptr(), session.recent1[i].data_ptr())
            self.assertTrue(torch.equal(memory[i], session.cache.cbf_memory[i]))
        session.score_answer([5], [6, 7])
        self.assertEqual(seq_len, session.cache.get_seq_length())

    def documents(self):
        return [{"sha256": str(i), "ids": list(range(i*1000, i*1000+188))} for i in range(16)]

    def test_data_groups_source_layout_and_causality(self):
        scenes = make_scenes(self.documents(), chunk_size=4)
        self.assertEqual(len(scenes), 32)
        self.assertFalse({x for s in scenes[:16] for x in s["sources"]} &
                         {x for s in scenes[16:] for x in s["sources"]})
        self.assertEqual(scenes[1]["prefix"][2], [1000, 1001, 1002, 1003])
        self.assertEqual(scenes[2]["prefix"][1], [1000, 1001, 1002, 1003])
        for scene in scenes:
            observed = sum(scene["prefix"], [])
            for future in scene["future"]:
                observed += future["ids"]
                self.assertFalse(set(future["answer_ids"]) & set(observed))
                self.assertEqual(len(future["answer_ids"]), 128)

    def test_summary_rejects_no_advantage_and_incomplete_labels(self):
        scenes = make_scenes(self.documents(), chunk_size=4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "scenes.jsonl"
            data.write_text("".join(json.dumps(s)+"\n" for s in scenes))
            Path(str(data)+".meta.json").write_text(json.dumps({"overlap_matching_rows": 0,
                "training_rows_scanned": 3, "documents": [{"sha256": str(i)} for i in range(16)]}))
            rows = []
            for scene in scenes:
                results = {name: {"losses": [2., 2., 2.], "nll": 2., "seconds": 1.,
                                  "peak_allocated_gib": 1., "peak_reserved_gib": 1.}
                           for name in [action_name(a) for a in GRID]+["fixed_"+n for n in FIXED]}
                rows.append({**{k: scene[k] for k in ("id", "group", "split", "regime", "sources", "protocol")},
                             "results": results, "smoke": False, "scene_seconds": 31., "model": "tiny",
                             "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest()})
            labels = root / "labels.jsonl"
            labels.write_text("".join(json.dumps(r)+"\n" for r in rows))
            args = SimpleNamespace(data=str(data), inputs=[str(labels)], output=str(root / "summary.json"))
            summarize(args)
            self.assertFalse(json.loads(Path(args.output).read_text())["passed_stage_a"])
            for row in rows:
                row["results"][action_name((1., 1., 0.))].update({"losses": [1.9]*3, "nll": 1.9})
            labels.write_text("".join(json.dumps(r)+"\n" for r in rows))
            summarize(args)
            self.assertTrue(json.loads(Path(args.output).read_text())["passed_stage_a"])
            labels.write_text("".join(json.dumps(r)+"\n" for r in rows[:-1]))
            with self.assertRaises(ValueError):
                summarize(args)


if __name__ == "__main__":
    unittest.main()
