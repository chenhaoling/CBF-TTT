"""Audit the frozen four-config Level-B matched-training matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_level_b_config_audit_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "baseline_reproduction"
CONFIG_NAMES = {
    (1, False): "qwen3_4b_public_stage1_no_ttt.yaml",
    (1, True): "qwen3_4b_public_stage1_ttt.yaml",
    (2, False): "qwen3_4b_public_stage2_no_ttt.yaml",
    (2, True): "qwen3_4b_public_stage2_ttt.yaml",
}
STAGES = {
    1: {
        "seq_len": 32768,
        "train_size": 20_000_538_624,
        "steps": 9537,
        "artifact": "stage1_32k.jsonl",
        "max_position_embeddings": 32768,
        "rope_scaling": None,
    },
    2: {
        "seq_len": 131072,
        "train_size": 14_998_831_104,
        "steps": 1788,
        "artifact": "stage2_128k.jsonl",
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
}
TRAIN_COMMON = {
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
    "enable_full_shard": True,
    "enable_fsdp_offload": False,
    "init_device": "meta",
    "ckpt_manager": "dcp",
    "save_steps": 500,
    "save_epochs": 1,
    "save_hf_weights": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def parse_scalar(value: str) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    if value in {"null", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        return [] if not body else [parse_scalar(item.strip()) for item in body.split(",")]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value):
        return float(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_mapping_yaml(text: str) -> dict[str, Any]:
    """Parse the mapping/scalar YAML subset used by the frozen configs."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-2, root)]
    for line_number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if "\t" in raw[:indent] or indent % 2:
            raise ValueError(f"unsupported YAML indentation at line {line_number}")
        if ":" not in stripped:
            raise ValueError(f"unsupported YAML syntax at line {line_number}")
        key, raw_value = stripped.split(":", 1)
        if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"unsupported YAML key at line {line_number}")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if key in parent:
            raise ValueError(f"duplicate YAML key at line {line_number}: {key}")
        raw_value = raw_value.strip()
        if not raw_value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(raw_value)
    return root


def read_config(path: Path) -> dict[str, Any]:
    value = parse_mapping_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration is not a mapping: {path}")
    return value


def without_runtime_paths(config: dict[str, Any], stage: int) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["train"].pop("output_dir", None)
    if stage == 2:
        value["model"].pop("model_path", None)
    return value


def validate_config(config: dict[str, Any], stage: int, ttt_mode: bool) -> list[str]:
    errors: list[str] = []
    spec = STAGES[stage]
    if set(config) != {"model", "data", "train"}:
        errors.append("top-level sections must be exactly model/data/train")
        return errors
    model, data, train = config["model"], config["data"], config["train"]
    if not isinstance(model.get("model_path"), str) or not model["model_path"]:
        errors.append("model.model_path is missing")
    foundation = model.get("foundation")
    if not isinstance(foundation, dict):
        errors.append("model.foundation is missing")
        return errors
    expected_foundation = {
        "max_position_embeddings": spec["max_position_embeddings"],
        **({"rope_scaling": spec["rope_scaling"]} if stage == 2 else {}),
        **FOUNDATION_COMMON,
        "ttt_mode": ttt_mode,
    }
    if foundation != expected_foundation:
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
    }
    if train != expected_train:
        errors.append("train section differs from the frozen specification")
    if not isinstance(train.get("output_dir"), str) or not train["output_dir"]:
        errors.append("train.output_dir is missing")
    computed = train.get("global_batch_size", 0) * data.get("max_seq_len", 0) \
        * train.get("max_steps", 0)
    if computed != data.get("train_size"):
        errors.append("global_batch_size * max_seq_len * max_steps != train_size")
    if "load_checkpoint_path" in train:
        errors.append("Stage 2 must load converted HF weights and reset optimizer/scheduler")
    return errors


def validate_corpus(root: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    manifest_path, audit_path = root / "manifest.json", root / "independent_audit.json"
    if not manifest_path.is_file() or not audit_path.is_file():
        return ["full corpus manifest or independent audit is missing"], {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "in_place_ttt_public_corpus_v1" \
            or manifest.get("profile") != "full":
        errors.append("corpus is not the frozen full profile")
    if manifest.get("totals", {}).get("training_tokens") != 34_999_369_728:
        errors.append("corpus total token count is wrong")
    stages = manifest.get("stages", [])
    if len(stages) != 2 or any(
        stages[index].get("combined_training_tokens") != STAGES[index + 1]["train_size"]
        or stages[index].get("training_steps") != STAGES[index + 1]["steps"]
        for index in range(min(len(stages), 2))
    ):
        errors.append("corpus stage counts/steps are wrong")
    if not audit.get("passed") \
            or audit.get("bindings", {}).get("manifest_sha256") != sha256(manifest_path):
        errors.append("corpus independent audit failed or binds another manifest")
    for spec in STAGES.values():
        if not (root / spec["artifact"]).is_file():
            errors.append(f"missing corpus artifact: {spec['artifact']}")
    return errors, {
        "root": str(root),
        "manifest_sha256": sha256(manifest_path),
        "audit_sha256": sha256(audit_path),
    }


def audit(config_root: Path = CONFIG_ROOT, corpus_root: Path | None = None) -> dict[str, Any]:
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
        baseline = without_runtime_paths(configs[(stage, False)], stage)
        ttt = without_runtime_paths(configs[(stage, True)], stage)
        baseline["model"]["foundation"]["ttt_mode"] = True
        if baseline != ttt:
            errors.append(f"stage {stage} arms differ beyond ttt_mode and runtime paths")
    if (1, False) in configs and (1, True) in configs \
            and configs[(1, False)]["model"]["model_path"] \
            != configs[(1, True)]["model"]["model_path"]:
        errors.append("Stage-1 arms must start from the same model path")
    corpus_binding = None
    if corpus_root is not None:
        corpus_errors, corpus_binding = validate_corpus(corpus_root)
        errors.extend(corpus_errors)
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "configs": records,
        "corpus": corpus_binding,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", default=str(CONFIG_ROOT))
    parser.add_argument("--corpus-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(
        Path(args.config_root).expanduser().resolve(),
        Path(args.corpus_root).expanduser().resolve() if args.corpus_root else None,
    )
    if args.output:
        output = Path(args.output).expanduser().resolve()
        partial = output.with_name(output.name + ".partial")
        if output.exists() or partial.exists():
            raise FileExistsError(f"refusing to overwrite audit output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
