#!/usr/bin/env python3
"""Build an immutable inference overlay with the paper's Qwen update clip.

The released repository does not expose the Qwen3 inference-time Frobenius
clip described in the paper.  This builder leaves the frozen upstream checkout
untouched, verifies every tracked ``inference_model`` input, and writes a
minimal, auditable copy that adds an opt-in ``ttt_update_clip_norm`` setting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
from typing import Final


UPSTREAM_COMMIT: Final = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
PAPER_CLIP_NORM: Final = 1e-5
PROTOCOL: Final = "in_place_ttt_paper_inference_overlay_v1"

EXPECTED_SOURCE_SHA256: Final = {
    "inference_model/__init__.py":
        "54ae124b5c993dff594b041dc4375407fbf1c016a689f9dac805e0043e86e52a",
    "inference_model/hf_llama3/__init__.py":
        "f084e24edbc2bb8cebe74f3ded368d40e1c6a848274f06a8077d09135567a1c9",
    "inference_model/hf_llama3/configuration_llama.py":
        "13e03dfcc525f809b215a0effae2837c82da612203f40d645b449e740650f8e8",
    "inference_model/hf_llama3/modeling_llama.py":
        "78e95c6cc3a64d4fc629ca6a9abb135883d1c0a6ca5184a2a0b8283f06802d85",
    "inference_model/hf_qwen3/__init__.py":
        "a38d76873e560e9c034258408e1c67fdfc08f2be846da1937cea1434f6e10e80",
    "inference_model/hf_qwen3/configuration_qwen3.py":
        "39ef69d7c79114ae9bd2710320c4260796fc5dbed9f725e18b5a0f5771114ec2",
    "inference_model/hf_qwen3/modeling_qwen3.py":
        "98ec94bcf1780635d7817b993a0bf4fa1f1a7e3a35415dce8a46c345a52a2858",
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one patch anchor, found {count}")
    return source.replace(old, new, 1)


def patch_configuration(source: str) -> str:
    source = replace_once(
        source,
        '        ttt_target="hidden_states",\n        **kwargs,\n',
        '        ttt_target="hidden_states",\n'
        '        ttt_update_clip_norm=None,\n'
        '        **kwargs,\n',
        "Qwen3Config signature",
    )
    source = replace_once(
        source,
        '        self.ttt_target = ttt_target\n'
        '        if self.ttt_target not in {"hidden_states", "input_embed"}:\n',
        '        self.ttt_target = ttt_target\n'
        '        self.ttt_update_clip_norm = ttt_update_clip_norm\n'
        '        if self.ttt_update_clip_norm is not None:\n'
        '            self.ttt_update_clip_norm = float(self.ttt_update_clip_norm)\n'
        '            if not (0.0 < self.ttt_update_clip_norm < float("inf")):\n'
        '                raise ValueError("ttt_update_clip_norm must be finite and positive")\n'
        '        if self.ttt_target not in {"hidden_states", "input_embed"}:\n',
        "Qwen3Config assignment",
    )
    return source


def patch_modeling(source: str) -> str:
    source = replace_once(
        source,
        '            self.ttt_lr = getattr(config, "ttt_lr", 0.3)\n'
        '            self.ttt_conv = nn.Conv1d(\n',
        '            self.ttt_lr = getattr(config, "ttt_lr", 0.3)\n'
        '            self.ttt_update_clip_norm = getattr(\n'
        '                config, "ttt_update_clip_norm", None\n'
        '            )\n'
        '            self.ttt_conv = nn.Conv1d(\n',
        "Qwen3MLP clip setting",
    )
    source = replace_once(
        source,
        '                current_w = current_w + dw\n',
        '                if self.ttt_update_clip_norm is not None:\n'
        '                    # Paper inference rule: clip each native update delta by\n'
        '                    # its full-matrix Frobenius norm before committing it.\n'
        '                    dw_norm = torch.linalg.matrix_norm(dw.float(), ord="fro")\n'
        '                    denominator = torch.clamp_min(\n'
        '                        dw_norm, torch.finfo(torch.float32).tiny\n'
        '                    )\n'
        '                    scale = torch.clamp(\n'
        '                        self.ttt_update_clip_norm / denominator, max=1.0\n'
        '                    ).to(dtype=dw.dtype)\n'
        '                    dw = dw * scale\n'
        '                current_w = current_w + dw\n',
        "Qwen3MLP update",
    )
    return source


def _git_output(root: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def build_overlay(
    upstream_root: pathlib.Path,
    output_root: pathlib.Path,
    clip_norm: float = PAPER_CLIP_NORM,
) -> dict:
    upstream_root = upstream_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    partial = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output_root} or {partial}")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    if clip_norm != PAPER_CLIP_NORM:
        raise ValueError(f"paper overlay requires clip_norm={PAPER_CLIP_NORM}")
    if _git_output(upstream_root, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError("upstream revision differs from the frozen commit")
    if _git_output(upstream_root, "status", "--short"):
        raise ValueError("upstream checkout is not clean")
    tracked = _git_output(upstream_root, "ls-files", "inference_model").splitlines()
    if set(tracked) != set(EXPECTED_SOURCE_SHA256):
        raise ValueError("tracked inference_model file set differs")

    source_manifest = {}
    for relative, expected in sorted(EXPECTED_SOURCE_SHA256.items()):
        path = upstream_root / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"upstream source hash differs: {relative}")
        source_manifest[relative] = {
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }

    partial.mkdir(parents=True)
    try:
        for relative in sorted(EXPECTED_SOURCE_SHA256):
            source_path = upstream_root / relative
            target_path = partial / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)

        config_path = partial / "inference_model/hf_qwen3/configuration_qwen3.py"
        model_path = partial / "inference_model/hf_qwen3/modeling_qwen3.py"
        config_path.write_text(
            patch_configuration(config_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        model_path.write_text(
            patch_modeling(model_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )

        overlay_manifest = {
            str(path.relative_to(partial)): {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(partial.rglob("*.py"))
        }
        unchanged = {
            relative for relative in EXPECTED_SOURCE_SHA256
            if relative not in {
                "inference_model/hf_qwen3/configuration_qwen3.py",
                "inference_model/hf_qwen3/modeling_qwen3.py",
            }
        }
        for relative in unchanged:
            if overlay_manifest[relative] != source_manifest[relative]:
                raise RuntimeError(f"unexpected overlay change: {relative}")

        manifest = {
            "protocol": PROTOCOL,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_root": str(upstream_root),
            "clip_scope": "each_native_qwen3_fast_weight_delta",
            "clip_norm_type": "frobenius",
            "clip_norm": clip_norm,
            "default_without_override": None,
            "source_files": source_manifest,
            "overlay_files": overlay_manifest,
            "modified_files": sorted(set(EXPECTED_SOURCE_SHA256) - unchanged),
            "builder_sha256": sha256(pathlib.Path(__file__).resolve()),
        }
        (partial / "paper_inference_overlay_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        partial.replace(output_root)
        return manifest
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--clip-norm", type=float, default=PAPER_CLIP_NORM)
    args = parser.parse_args()
    manifest = build_overlay(args.upstream_root, args.output, args.clip_norm)
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "protocol": manifest["protocol"],
        "clip_norm": manifest["clip_norm"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
