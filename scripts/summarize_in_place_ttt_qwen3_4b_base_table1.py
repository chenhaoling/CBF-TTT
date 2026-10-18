#!/usr/bin/env python3
"""Build the audited seven-context corrected-Base Table 1 conclusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

try:
    from scripts.summarize_in_place_ttt_2x5090_ruler_matrix import PAPER_TABLE1
    from scripts.summarize_in_place_ttt_level_b_ruler import TASK_PREFIXES
except ModuleNotFoundError:  # Direct script execution.
    from summarize_in_place_ttt_2x5090_ruler_matrix import PAPER_TABLE1
    from summarize_in_place_ttt_level_b_ruler import TASK_PREFIXES


PROTOCOL = "in_place_ttt_qwen3_4b_base_table1_summary_v1"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
CONTEXTS = (
    (4096, "4k"), (8192, "8k"), (16384, "16k"),
    (32768, "32k"), (65536, "64k"), (131072, "128k"),
    (262144, "256k"),
)
PRIMARY = tuple(suffix for _, suffix in CONTEXTS[:-1])
ARMS = ("no_ttt", "ttt")
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPS = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[math.floor(p * (len(ordered) - 1))]


def _summary_path(matrix_root: Path, formal_256k_root: Path,
                  arm: str, suffix: str) -> Path:
    return ((formal_256k_root / arm / "summary.json") if suffix == "256k"
            else (matrix_root / arm / suffix / "summary.json"))


def _normalized(record: dict[str, Any], suffix: str,
                field: str) -> dict[str, Any]:
    token = "_" + suffix
    values = record.get(field, {})
    return {
        name[:-len(token)]: value for name, value in values.items()
        if name.endswith(token)
    }


def summarize(
    *, matrix_root: Path, corrected_matrix_audit: Path,
    formal_256k_root: Path, corrected_formal_256k_audit: Path,
    corrected_primary_resource_audit: Path,
    corrected_256k_resource_audit: Path, plan: Path,
) -> dict[str, Any]:
    matrix, formal = load(corrected_matrix_audit), load(
        corrected_formal_256k_audit
    )
    primary_resource, resource_256k = load(corrected_primary_resource_audit), load(
        corrected_256k_resource_audit
    )
    if (matrix.get("protocol")
            != "in_place_ttt_qwen3_4b_base_ruler_matrix_audit_v1"
            or matrix.get("passed") is not True or matrix.get("errors") != []
            or matrix.get("model_revision") != MODEL_REVISION):
        raise ValueError("corrected primary matrix audit is invalid")
    if (formal.get("protocol")
            != "in_place_ttt_qwen3_4b_base_ruler_256k_audit_v1"
            or formal.get("passed") is not True or formal.get("errors") != []
            or formal.get("model_revision") != MODEL_REVISION):
        raise ValueError("corrected formal 256K audit is invalid")
    if (primary_resource.get("protocol")
            != "in_place_ttt_qwen3_4b_base_ruler_resource_audit_v1"
            or primary_resource.get("passed") is not True
            or primary_resource.get("errors") != []
            or resource_256k.get("protocol")
            != "in_place_ttt_qwen3_4b_base_ruler_256k_resource_audit_v1"
            or resource_256k.get("passed") is not True
            or resource_256k.get("errors") != []):
        raise ValueError("corrected resource evidence is invalid")

    audited_scores = dict(matrix.get("context_scores", {}))
    audited_scores["256k"] = formal.get("scores")
    if set(audited_scores) != {suffix for _, suffix in CONTEXTS}:
        raise ValueError("audited context score set differs")
    task_records: dict[str, dict[str, dict[str, float]]] = {}
    accounting: dict[str, Any] = {}
    source_bindings: dict[str, Any] = {}
    task_deltas: dict[str, dict[str, float]] = {}
    for context, suffix in CONTEXTS:
        arm_records: dict[str, Any] = {}
        arm_tasks: dict[str, dict[str, float]] = {}
        arm_inputs: dict[str, dict[str, str]] = {}
        source_bindings[suffix] = {}
        for arm in ARMS:
            path = _summary_path(matrix_root, formal_256k_root, arm, suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
            record = load(path)
            if (record.get("protocol")
                    != "in_place_ttt_2x5090_ruler_summary_v1"
                    or record.get("completed") is not True
                    or record.get("arm") != arm
                    or record.get("context_length") != context
                    or record.get("num_samples_per_task") != 100
                    or record.get("task_count") != 13
                    or record.get("exit_code") != 0
                    or record.get("tokenizer_regex_warning_count") != 0
                    or any(record.get(name) for name in (
                        "missing_result_tasks", "unexpected_result_tasks",
                        "missing_prediction_tasks", "unexpected_prediction_tasks",
                    ))):
                raise ValueError(f"{suffix}/{arm}: formal summary is invalid")
            counts = record.get("prediction_sample_counts", {})
            if len(counts) != 13 or set(counts.values()) != {100}:
                raise ValueError(f"{suffix}/{arm}: sample counts differ")
            tasks = {name: float(value) for name, value in
                     _normalized(record, suffix, "task_scores").items()}
            inputs = _normalized(record, suffix, "evaluation_input_sha256")
            if set(tasks) != set(TASK_PREFIXES) or set(inputs) != set(TASK_PREFIXES):
                raise ValueError(f"{suffix}/{arm}: task or input set differs")
            macro = sum(tasks.values()) / len(tasks)
            if abs(macro - float(record["macro_accuracy"])) >= 1e-12:
                raise ValueError(f"{suffix}/{arm}: macro differs from tasks")
            if abs(macro - float(audited_scores[suffix][arm])) >= 1e-12:
                raise ValueError(f"{suffix}/{arm}: corrected audit score differs")
            arm_records[arm], arm_tasks[arm], arm_inputs[arm] = record, tasks, inputs
            source_bindings[suffix][arm] = {"summary_sha256": sha256(path)}
        if arm_inputs["no_ttt"] != arm_inputs["ttt"]:
            raise ValueError(f"{suffix}: paired inputs differ")
        task_records[suffix] = {
            task: {
                "no_ttt": arm_tasks["no_ttt"][task],
                "ttt": arm_tasks["ttt"][task],
                "ttt_minus_no_ttt": (
                    arm_tasks["ttt"][task] - arm_tasks["no_ttt"][task]
                ),
            }
            for task in sorted(TASK_PREFIXES)
        }
        task_deltas[suffix] = {
            task: values["ttt_minus_no_ttt"]
            for task, values in task_records[suffix].items()
        }
        accounting[suffix] = {
            "tasks_per_arm": 13,
            "samples_per_task": 100,
            "samples_per_arm": 1300,
            "paired_inputs": True,
            "failed_examples": 0,
        }

    context_scores = {
        suffix: {
            "no_ttt": float(audited_scores[suffix]["no_ttt"]),
            "ttt": float(audited_scores[suffix]["ttt"]),
            "ttt_minus_no_ttt": float(
                audited_scores[suffix]["ttt_minus_no_ttt"]
            ),
        }
        for _, suffix in CONTEXTS
    }
    for suffix, values in context_scores.items():
        if abs(values["ttt"] - values["no_ttt"]
               - values["ttt_minus_no_ttt"]) >= 1e-12:
            raise ValueError(f"{suffix}: audited delta is inconsistent")
    primary_means = {
        arm: sum(context_scores[suffix][arm] for suffix in PRIMARY) / len(PRIMARY)
        for arm in ARMS
    }
    primary_means["ttt_minus_no_ttt"] = (
        primary_means["ttt"] - primary_means["no_ttt"]
    )
    comparison = {
        suffix: {
            "public_no_ttt_minus_paper_matched": (
                values["no_ttt"] - PAPER_TABLE1["matched_baseline"][suffix]
            ),
            "public_ttt_minus_paper_ttt": (
                values["ttt"] - PAPER_TABLE1["in_place_ttt"][suffix]
            ),
            "public_gain_minus_paper_gain": (
                values["ttt_minus_no_ttt"]
                - (PAPER_TABLE1["in_place_ttt"][suffix]
                   - PAPER_TABLE1["matched_baseline"][suffix])
            ),
        }
        for suffix, values in context_scores.items()
    }

    rng = random.Random(BOOTSTRAP_SEED)
    tasks = sorted(TASK_PREFIXES)
    context_samples = {suffix: [] for _, suffix in CONTEXTS}
    primary_samples: list[float] = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        for _, suffix in CONTEXTS:
            context_samples[suffix].append(
                sum(task_deltas[suffix][task] for task in sampled) / len(sampled)
            )
        primary_samples.append(sum(
            task_deltas[suffix][task] for suffix in PRIMARY for task in sampled
        ) / (len(PRIMARY) * len(sampled)))
    bootstrap = {
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_REPS,
        "unit": "paired_ruler_task",
        "interval": "percentile_95_floor_index",
        "scope": "task_set_uncertainty_not_example_level",
        "context_delta_ci95": {
            suffix: [percentile(values, 0.025), percentile(values, 0.975)]
            for suffix, values in context_samples.items()
        },
        "primary_mean_delta_ci95": [
            percentile(primary_samples, 0.025),
            percentile(primary_samples, 0.975),
        ],
    }

    def resources(payload: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, record in payload.get("boundaries", {}).items():
            gpu = record.get("gpu", {})
            result[key] = {
                "peak_memory_used_mib": gpu.get("peak_memory_used_mib"),
                "memory_total_mib": gpu.get("memory_total_mib"),
                "scores_discarded": True,
            }
        return result

    return {
        "protocol": PROTOCOL,
        "completed": True,
        "model_revision": MODEL_REVISION,
        "contexts": [suffix for _, suffix in CONTEXTS],
        "context_scores": context_scores,
        "task_scores": task_records,
        "sample_accounting": accounting,
        "registered_4k_to_128k_mean": primary_means,
        "paper_table1_reference": PAPER_TABLE1,
        "public_replication_minus_paper": comparison,
        "paired_task_bootstrap": bootstrap,
        "resource_measurements": {
            "primary": resources(primary_resource),
            "256k": resources(resource_256k),
        },
        "bindings": {
            "corrected_matrix_audit_sha256": sha256(corrected_matrix_audit),
            "corrected_formal_256k_audit_sha256": sha256(
                corrected_formal_256k_audit
            ),
            "corrected_primary_resource_audit_sha256": sha256(
                corrected_primary_resource_audit
            ),
            "corrected_256k_resource_audit_sha256": sha256(
                corrected_256k_resource_audit
            ),
            "plan_sha256": sha256(plan),
            "summarizer_sha256": sha256(Path(__file__).resolve()),
            "source_summaries": source_bindings,
        },
        "claim_boundary": (
            "public_data_fineweb_edu_longcrawl64_matched_replication_"
            "not_private_paper_corpus_or_scale"
        ),
        "baseline_evidence_complete": True,
        "gate_experiment_started": False,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Corrected Qwen3-4B-Base In-Place TTT Table 1 result", "",
        "This is a public-data matched replication using FineWeb-Edu and "
        "LongCrawl64. It is not an exact reproduction of the private paper "
        "corpus or its approximately 35B-token scale.", "",
        "| Context | no-TTT | In-Place TTT | TTT - no-TTT | Paper matched | "
        "Paper TTT | Delta 95% task bootstrap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    scores = summary["context_scores"]
    cis = summary["paired_task_bootstrap"]["context_delta_ci95"]
    paper = summary["paper_table1_reference"]
    for suffix in summary["contexts"]:
        row, ci = scores[suffix], cis[suffix]
        lines.append(
            f"| {suffix.upper()} | {row['no_ttt']:.4f} | {row['ttt']:.4f} | "
            f"{row['ttt_minus_no_ttt']:.4f} | "
            f"{paper['matched_baseline'][suffix]:.1f} | "
            f"{paper['in_place_ttt'][suffix]:.1f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] |"
        )
    mean = summary["registered_4k_to_128k_mean"]
    ci = summary["paired_task_bootstrap"]["primary_mean_delta_ci95"]
    lines += [
        "", "## Registered primary mean", "",
        f"4K--128K no-TTT: **{mean['no_ttt']:.4f}**; In-Place TTT: "
        f"**{mean['ttt']:.4f}**; delta: **{mean['ttt_minus_no_ttt']:.4f}**; "
        f"paired-task 95% interval: **[{ci[0]:.4f}, {ci[1]:.4f}]**.",
        "", "The interval resamples the 13 RULER tasks, not individual examples. "
        "It is descriptive and is not a directional pass criterion.",
    ]
    for suffix in summary["contexts"]:
        lines += ["", f"## {suffix.upper()} task scores", "",
                  "| Task | no-TTT | In-Place TTT | Delta |",
                  "|---|---:|---:|---:|"]
        for task, row in summary["task_scores"][suffix].items():
            lines.append(
                f"| {task} | {row['no_ttt']:.4f} | {row['ttt']:.4f} | "
                f"{row['ttt_minus_no_ttt']:.4f} |"
            )
    lines += [
        "", "## Accounting and claim boundary", "",
        "Every context contains 13 tasks and 100 examples per task and arm, "
        "with paired inputs and zero audited failures. The 256K point is a "
        "separate YaRN extrapolation and is excluded from the primary mean.",
        "", "This report completes the registered baseline evidence only. It "
        "does not start update-admission training or open a sealed gate test.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "matrix_root", "corrected_matrix_audit", "formal_256k_root",
        "corrected_formal_256k_audit", "corrected_primary_resource_audit",
        "corrected_256k_resource_audit", "plan", "output_json", "output_md",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("Table 1 reporter must hide all GPUs")
    paths = {key: value.expanduser().resolve() for key, value in vars(args).items()}
    output_json, output_md = paths.pop("output_json"), paths.pop("output_md")
    for output in (output_json, output_md):
        if output.exists() or output.with_name(output.name + ".partial").exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    result = summarize(**paths)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    json_partial = output_json.with_name(output_json.name + ".partial")
    md_partial = output_md.with_name(output_md.name + ".partial")
    json_partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    md_partial.write_text(render_markdown(result))
    os.replace(json_partial, output_json)
    os.replace(md_partial, output_md)
    print(json.dumps({"completed": True, "json": str(output_json),
                      "markdown": str(output_md)}, sort_keys=True))


if __name__ == "__main__":
    main()
