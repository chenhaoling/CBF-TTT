import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from scripts.build_in_place_ttt_paper_inference_overlay import (
    EXPECTED_SOURCE_SHA256,
    PAPER_CLIP_NORM,
    PROTOCOL,
    UPSTREAM_COMMIT,
    build_overlay,
    patch_configuration,
    patch_modeling,
)


class PaperInferenceOverlayTests(unittest.TestCase):
    def _clean_upstream(self, root: pathlib.Path) -> pathlib.Path:
        upstream = root / "upstream"
        subprocess.run(
            [
                "git", "clone", "--no-hardlinks", "--quiet",
                str(pathlib.Path(__file__).resolve().parents[1]), str(upstream),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(upstream), "checkout", "--detach", "--quiet",
             UPSTREAM_COMMIT],
            check=True,
        )
        return upstream

    def test_builder_verifies_clean_upstream_and_changes_only_qwen_files(self):
        with tempfile.TemporaryDirectory(prefix="paper-inference-overlay-") as temp:
            root = pathlib.Path(temp)
            upstream = self._clean_upstream(root)
            output = root / "overlay"
            manifest = build_overlay(upstream, output)
            self.assertEqual(manifest["protocol"], PROTOCOL)
            self.assertEqual(manifest["clip_norm"], PAPER_CLIP_NORM)
            self.assertEqual(set(manifest["source_files"]), set(EXPECTED_SOURCE_SHA256))
            self.assertEqual(
                manifest["modified_files"],
                [
                    "inference_model/hf_qwen3/configuration_qwen3.py",
                    "inference_model/hf_qwen3/modeling_qwen3.py",
                ],
            )
            for relative in EXPECTED_SOURCE_SHA256:
                if relative not in manifest["modified_files"]:
                    self.assertEqual(
                        manifest["source_files"][relative],
                        manifest["overlay_files"][relative],
                    )
            stored = json.loads(
                (output / "paper_inference_overlay_manifest.json").read_text()
            )
            self.assertEqual(stored, manifest)
            with self.assertRaises(FileExistsError):
                build_overlay(upstream, output)

    def test_patch_is_fail_closed_and_default_is_disabled(self):
        with self.assertRaises(ValueError):
            patch_configuration("not upstream source")
        with self.assertRaises(ValueError):
            patch_modeling("not upstream source")

        source = subprocess.check_output(
            ["git", "show", f"{UPSTREAM_COMMIT}:"
             "inference_model/hf_qwen3/configuration_qwen3.py"],
            text=True,
        )
        patched = patch_configuration(source)
        self.assertIn("ttt_update_clip_norm=None", patched)
        self.assertIn("finite and positive", patched)
        with self.assertRaises(ValueError):
            patch_configuration(patched)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
        "requires the frozen inference environment",
    )
    def test_runtime_clips_each_delta_and_serializes_setting(self):
        with tempfile.TemporaryDirectory(prefix="paper-inference-runtime-") as temp:
            root = pathlib.Path(temp)
            upstream = self._clean_upstream(root)
            output = root / "overlay"
            build_overlay(upstream, output)
            program = r'''
import math
import torch
from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3MLP

def config(clip):
    return Qwen3Config(
        vocab_size=32, hidden_size=4, intermediate_size=6,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        head_dim=4, max_position_embeddings=16, ttt_layers=[0],
        ttt_mode=True, ttt_proj=False, ttt_lr=1.0, ttt_chunk=2,
        ttt_update_clip_norm=clip,
    )

torch.manual_seed(7)
plain = Qwen3MLP(config(None), layer_idx=0)
clipped = Qwen3MLP(config(1e-5), layer_idx=0)
clipped.load_state_dict(plain.state_dict())
with torch.no_grad():
    for module in (plain, clipped):
        module.ttt_conv.weight.zero_()
        module.ttt_conv.weight[:, 0, 2] = 1.0
x = torch.full((1, 2, 4), 2.0)
t = torch.full((1, 2, 4), 3.0)
past = torch.zeros_like(plain.down_proj.weight)
_, plain_w = plain(x, t=t, past_w=past)
_, clipped_w = clipped(x, t=t, past_w=past)
plain_norm = torch.linalg.matrix_norm(plain_w.float(), ord="fro").item()
clipped_norm = torch.linalg.matrix_norm(clipped_w.float(), ord="fro").item()
assert plain_norm > 1e-5, plain_norm
assert math.isclose(clipped_norm, 1e-5, rel_tol=2e-5, abs_tol=1e-10), clipped_norm
zero_t = torch.zeros_like(t)
_, zero_w = clipped(x, t=zero_t, past_w=past)
assert torch.equal(zero_w, past)
assert config(None).to_dict()["ttt_update_clip_norm"] is None
assert config(1e-5).to_dict()["ttt_update_clip_norm"] == 1e-5
for invalid in (0.0, -1.0, float("inf"), float("nan")):
    try:
        config(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted invalid clip: {invalid}")
'''
            env = os.environ.copy()
            env["PYTHONPATH"] = str(output)
            subprocess.run([sys.executable, "-c", program], env=env, check=True)


if __name__ == "__main__":
    unittest.main()
