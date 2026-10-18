#!/usr/bin/env python3
"""Validate the paper-clipped Qwen inference overlay on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import tempfile


PROTOCOL = "in_place_ttt_paper_inference_overlay_validation_v1"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def validate(overlay: pathlib.Path) -> dict:
    overlay = overlay.expanduser().resolve()
    manifest_path = overlay / "paper_inference_overlay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "in_place_ttt_paper_inference_overlay_v1":
        raise ValueError("overlay manifest protocol differs")
    sys.path.insert(0, str(overlay))

    import torch
    from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
    from inference_model.hf_qwen3.modeling_qwen3 import (
        Qwen3ForCausalLM,
        Qwen3MLP,
    )

    module_path = pathlib.Path(sys.modules[Qwen3MLP.__module__].__file__).resolve()
    if overlay not in module_path.parents:
        raise RuntimeError("Qwen3MLP was not imported from the requested overlay")

    def config(clip):
        return Qwen3Config(
            vocab_size=32,
            hidden_size=4,
            intermediate_size=6,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
            ttt_layers=[0],
            ttt_mode=True,
            ttt_proj=False,
            ttt_lr=1.0,
            ttt_chunk=2,
            ttt_update_clip_norm=clip,
        )

    torch.manual_seed(7)
    plain = Qwen3MLP(config(None), layer_idx=0).cpu()
    clipped = Qwen3MLP(config(1e-5), layer_idx=0).cpu()
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
        _, zero_w = clipped(x, t=torch.zeros_like(t), past_w=past)
    plain_norm = torch.linalg.matrix_norm(plain_w.float(), ord="fro").item()
    clipped_norm = torch.linalg.matrix_norm(clipped_w.float(), ord="fro").item()
    if not plain_norm > 1e-5:
        raise AssertionError(f"fixture delta is not above the boundary: {plain_norm}")
    if not math.isclose(clipped_norm, 1e-5, rel_tol=2e-5, abs_tol=1e-10):
        raise AssertionError(f"clipped delta norm differs: {clipped_norm}")
    if not torch.equal(zero_w, past):
        raise AssertionError("zero update changed the fast weight")
    if config(None).to_dict().get("ttt_update_clip_norm", "missing") is not None:
        raise AssertionError("default clip setting is not disabled")
    if config(1e-5).to_dict().get("ttt_update_clip_norm") != 1e-5:
        raise AssertionError("enabled clip setting does not serialize")
    with tempfile.TemporaryDirectory(prefix="paper-clip-from-pretrained-") as temp:
        tiny_model = Qwen3ForCausalLM(config(None)).cpu()
        tiny_model.save_pretrained(temp)
        loaded = Qwen3ForCausalLM.from_pretrained(
            temp, ttt_update_clip_norm=1e-5,
        ).cpu()
        if loaded.config.ttt_update_clip_norm != 1e-5:
            raise AssertionError("from_pretrained did not apply the clip override")
        if loaded.model.layers[0].mlp.ttt_update_clip_norm != 1e-5:
            raise AssertionError("Qwen3MLP did not receive the clip override")
    rejected = []
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        try:
            config(invalid)
        except ValueError:
            rejected.append(str(invalid))
        else:
            raise AssertionError(f"accepted invalid clip value: {invalid}")

    return {
        "protocol": PROTOCOL,
        "passed": True,
        "device": "cpu",
        "cuda_initialized": torch.cuda.is_initialized(),
        "overlay_root": str(overlay),
        "overlay_manifest_sha256": sha256(manifest_path),
        "modeling_module_path": str(module_path),
        "clip_norm": 1e-5,
        "unclipped_delta_frobenius_norm": plain_norm,
        "clipped_delta_frobenius_norm": clipped_norm,
        "zero_delta_unchanged": True,
        "config_default_disabled": True,
        "config_round_trip": True,
        "from_pretrained_override": True,
        "invalid_values_rejected": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    result = validate(args.overlay)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        partial = output.with_name(output.name + ".partial")
        if output.exists() or partial.exists():
            raise FileExistsError(f"refusing to overwrite {output} or {partial}")
        partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        partial.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
