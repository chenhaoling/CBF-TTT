"""Audit S1 LoRA training before any calibration score is accepted."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_structured_fact_comparator_s1_train_v1"
CHECKPOINTS = {"checkpoint_epoch_0.25": 188, "checkpoint_epoch_0.50": 375,
               "checkpoint_epoch_1.00": 750, "checkpoint_epoch_2.00": 1500}
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj"}


def adapter_keys(path: str | Path) -> list[str]:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return sorted(handle.keys())


def audit(root: str | Path, data_root: str | Path) -> dict:
    root, data_root = Path(root), Path(data_root)
    manifest_path = root / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data_manifest = data_root / "manifest.json"
    data_audit = data_root / "audit.json"
    checks = {
        "protocol": manifest["protocol"] == PROTOCOL,
        "formal_complete": manifest["formal"] is True
            and manifest["terminal_status"] == "completed_training_before_calibration",
        "schedule": manifest["completed_steps"] == 1500
            and manifest["steps_per_epoch"] == 750
            and {int(key): value for key, value in manifest["checkpoint_steps"].items()}
            == {188: .25, 375: .5, 750: 1., 1500: 2.},
        "batch": manifest["world_size"] == 2 and manifest["effective_batch"] == 64
            and manifest["per_device_batch"] == 4 and manifest["gradient_accumulation"] == 8,
        "lengths": 0 < manifest["length_min"] <= manifest["length_mean"]
            <= manifest["length_max"] <= 384,
        "lora": manifest["lora"]["rank"] == 16 and manifest["lora"]["alpha"] == 32
            and manifest["lora"]["dropout"] == .05
            and set(manifest["lora"]["target_modules"]) == TARGETS
            and manifest["lora"]["trainable_parameters"] > 0,
        "optimizer": manifest["optimizer"] == {
            "name": "AdamW", "lr": 2e-4, "weight_decay": .01,
            "warmup_steps": 45, "schedule": "cosine", "grad_clip": 1.0},
        "data": manifest["data"]["manifest_sha256"] == sha(data_manifest)
            and manifest["data"]["audit_sha256"] == sha(data_audit)
            and json.loads(data_audit.read_text())["passed"] is True,
        "sealed": manifest["calibration_loaded"] is False
            and manifest["development_loaded"] is False
            and manifest["semeval_test_loaded"] is False,
        "profiles": len(manifest["profiles"]) == 2
            and {profile["rank"] for profile in manifest["profiles"]} == {0, 1}
            and all(profile["peak_allocated_gib"] < 32
                    and profile["peak_reserved_gib"] < 32
                    for profile in manifest["profiles"]),
    }
    checkpoint_checks, files = {}, {}
    for name, step in CHECKPOINTS.items():
        checkpoint = root / name
        meta_path = checkpoint / "checkpoint_meta.json"
        config_path = checkpoint / "adapter_config.json"
        weights_path = checkpoint / "adapter_model.safetensors"
        meta, config = json.loads(meta_path.read_text()), json.loads(config_path.read_text())
        keys = adapter_keys(weights_path)
        expected_epoch = float(name.removeprefix("checkpoint_epoch_"))
        checkpoint_checks[name] = {
            "step": meta["global_step"] == step,
            "epoch": abs(meta["epoch"] - expected_epoch) < 1e-12,
            "data": meta["data"] == manifest["data"],
            "config": config["r"] == 16 and config["lora_alpha"] == 32
                and config["lora_dropout"] == .05
                and set(config["target_modules"]) == TARGETS
                and config["bias"] == "none",
            "adapter_only": bool(keys) and all("lora_" in key for key in keys),
        }
        files[name] = {"meta_sha256": sha(meta_path), "config_sha256": sha(config_path),
                       "weights_sha256": sha(weights_path), "tensor_keys": len(keys)}
    checks["checkpoints"] = checkpoint_checks
    logs = {}
    for rank in range(2):
        rows = [json.loads(line) for line in (root / f"train_rank{rank}.jsonl").read_text().splitlines()
                if line]
        logs[str(rank)] = {"rows": len(rows), "last_step": rows[-1]["step"],
                           "sha256": sha(root / f"train_rank{rank}.jsonl")}
        checks[f"rank_{rank}_log"] = (rows[0]["step"] == 1 and rows[-1]["step"] == 1500
                                      and all(row["rank"] == rank and math.isfinite(row["loss"])
                                              and row["loss"] >= 0 for row in rows))

    def flatten(value):
        if isinstance(value, dict):
            return all(flatten(item) for item in value.values())
        return bool(value)
    passed = flatten(checks)
    result = {"protocol": "qwen_structured_fact_comparator_s1_training_audit_v1",
              "passed": passed, "checks": checks, "checkpoint_files": files,
              "logs": logs, "training_manifest_sha256": sha(manifest_path),
              "calibration_loaded": False, "development_loaded": False,
              "semeval_test_loaded": False}
    if not passed:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
