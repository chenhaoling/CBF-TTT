"""Audit the opt-in 2xRTX5090 100M matched-training configuration matrix."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

try:
    from scripts.audit_in_place_ttt_level_b_configs import read_config, sha256
except ModuleNotFoundError:  # Direct `python scripts/<name>.py` execution.
    from audit_in_place_ttt_level_b_configs import read_config, sha256


PROTOCOL = "in_place_ttt_2x5090_config_audit_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "baseline_reproduction"
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_2X5090_EXECUTION_ADDENDUM.md"
CONFIG_NAMES = {
    (1, False): "qwen3_4b_2x5090_100m_stage1_no_ttt.yaml",
    (1, True): "qwen3_4b_2x5090_100m_stage1_ttt.yaml",
    (2, False): "qwen3_4b_2x5090_100m_stage2_no_ttt.yaml",
    (2, True): "qwen3_4b_2x5090_100m_stage2_ttt.yaml",
}
STAGES = {
    1: {
        "seq_len": 32768,
        "steps": 27,
        "train_size": 56_623_104,
        "artifact": "stage1_32k.jsonl",
        "save_steps": 27,
        "max_position_embeddings": 32768,
        "rope_scaling": None,
    },
    2: {
        "seq_len": 131072,
        "steps": 5,
        "train_size": 41_943_040,
        "artifact": "stage2_128k.jsonl",
        "save_steps": 5,
        "max_position_embeddings": 131072,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
    },
}
FOUNDATION_COMMON = {
    "ttt_layers": [0, 6, 12, 18, 24, 30],
    "ttt_proj": True,
    "ttt_target": "hidden_states",
    "ttt_lr": 3,
    "ttt_chunk": 4096,
}
DATA_COMMON = {
    "dataloader_type": "native",
    "datasets_type": "iterable",
    "data_type": "plaintext",
    "text_keys": "content_split",
    "drop_last": True,
    "num_workers": 2,
    "prefetch_factor": 2,
    "pin_memory": True,
}
TRAIN_COMMON = {
    "seed": 42,
    "data_parallel_mode": "fsdp2",
    "ulysses_parallel_size": 1,
    "global_batch_size": 64,
    "micro_batch_size": 1,
    "rmpad": False,
    "use_wandb": False,
    "num_train_epochs": 1,
    "rmpad_with_pos_ids": False,
    "dyn_bsz_margin": 0,
    "dyn_bsz": False,
    "optimizer": "adamw",
    "lr": 5.0e-6,
    "lr_warmup_ratio": 0.02,
    "lr_decay_style": "cosine",
    "lr_decay_ratio": 0.90,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "enable_mixed_precision": True,
    "enable_gradient_checkpointing": True,
    "enable_activation_offload": True,
    "activation_gpu_limit": 0.0,
    "enable_full_shard": True,
    "enable_fsdp_offload": False,
    "init_device": "meta",
    "ckpt_manager": "dcp",
    "save_epochs": 0,
    "save_hf_weights": True,
}


def validate_config(config: dict[str, Any], stage: int, ttt_mode: bool) -> list[str]:
    errors: list[str] = []
    spec = STAGES[stage]
    if set(config) != {"model", "data", "train"}:
        return ["top-level sections must be exactly model/data/train"]
    model, data, train = config["model"], config["data"], config["train"]
    if not isinstance(model.get("model_path"), str) or not model["model_path"]:
        errors.append("model.model_path is missing")
    expected_foundation = {
        "max_position_embeddings": spec["max_position_embeddings"],
        **({"rope_scaling": spec["rope_scaling"]} if stage == 2 else {}),
        **FOUNDATION_COMMON,
        "ttt_mode": ttt_mode,
    }
    if model.get("foundation") != expected_foundation:
        errors.append("model.foundation differs from the frozen stage/arm specification")
    expected_data = {
        "train_path": data.get("train_path"),
        "train_size": spec["train_size"],
        **DATA_COMMON,
        "max_seq_len": spec["seq_len"],
    }
    if data != expected_data:
        errors.append("data section differs from the frozen specification")
    if not isinstance(data.get("train_path"), str) \
            or Path(data["train_path"]).name != spec["artifact"]:
        errors.append(f"data.train_path must end in {spec['artifact']}")
    expected_train = {
        "output_dir": train.get("output_dir"),
        **TRAIN_COMMON,
        "max_steps": spec["steps"],
        "save_steps": spec["save_steps"],
    }
    if train != expected_train:
        errors.append("train section differs from the frozen 2x5090 specification")
    if not isinstance(train.get("output_dir"), str) or not train["output_dir"]:
        errors.append("train.output_dir is missing")
    if train.get("global_batch_size", 0) * data.get("max_seq_len", 0) \
            * train.get("max_steps", 0) != data.get("train_size"):
        errors.append("global_batch_size * max_seq_len * max_steps != train_size")
    if train.get("enable_fsdp_offload") is not False:
        errors.append("parameter/optimizer CPU offload is forbidden")
    if train.get("enable_activation_offload") is not True:
        errors.append("activation offload must be enabled")
    if "load_checkpoint_path" in train:
        errors.append("Stage 2 must start from converted HF weights with fresh optimizer state")
    return errors


def without_runtime_paths(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["model"].pop("model_path", None)
    value["train"].pop("output_dir", None)
    return value


def audit(config_root: Path = CONFIG_ROOT) -> dict[str, Any]:
    configs: dict[tuple[int, bool], dict[str, Any]] = {}
    records: dict[str, Any] = {}
    errors: list[str] = []
    for key, name in CONFIG_NAMES.items():
        path = config_root / name
        if not path.is_file():
            errors.append(f"missing config: {name}")
            continue
        config = read_config(path)
        configs[key] = config
        config_errors = validate_config(config, *key)
        errors.extend(f"{name}: {item}" for item in config_errors)
        records[name] = {"sha256": sha256(path), "errors": config_errors}
    for stage in STAGES:
        if (stage, False) not in configs or (stage, True) not in configs:
            continue
        baseline = without_runtime_paths(configs[(stage, False)])
        ttt = without_runtime_paths(configs[(stage, True)])
        baseline["model"]["foundation"]["ttt_mode"] = True
        if baseline != ttt:
            errors.append(f"stage {stage} arms differ beyond ttt_mode and runtime paths")
    if (1, False) in configs and (1, True) in configs \
            and configs[(1, False)]["model"]["model_path"] \
            != configs[(1, True)]["model"]["model_path"]:
        errors.append("Stage-1 arms must start from the same model path")
    total_tokens = sum(spec["train_size"] for spec in STAGES.values())
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "plan_sha256": sha256(PLAN),
        "configs": records,
        "world_size": 2,
        "total_tokens": total_tokens,
        "stage1_fraction": STAGES[1]["train_size"] / total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", default=str(CONFIG_ROOT))
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(Path(args.config_root).expanduser().resolve())
    if args.output:
        output = Path(args.output).expanduser().resolve()
        partial = output.with_name(output.name + ".partial")
        partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        partial.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
