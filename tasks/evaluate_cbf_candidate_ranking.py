"""Localize dynamic-writer value binding with fixed 2/5/16 candidate ranking."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_dynamic_memory import load_rows, select_view
from tasks.train_cbf_event_content import regions
from tasks.train_cbf_event_writer import setup, write_json


ARMS = ("joint_ce", "joint_binding")
EXPOSURES = (0, 2048, 4096)
POLICIES = ("correct", "twin", "wrong", "empty")
SETS = ("pair2", "world5", "global16")
RANK_METRICS = ("top1", "mrr", "rank", "margin", "true_digit_nll", "true_answer_nll")


def _ordered(row_id: str, values: set[str], seed: int = 719) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(
        f"{seed}|{row_id}|{value}".encode()).hexdigest())


def build_candidate_sets(rows: list[dict], row: dict) -> dict[str, list[str]]:
    same_split = [item for item in rows if item["split"] == row["split"]]
    by_id = {item["id"]: item for item in same_split}
    world = {item["answer"] for item in same_split if item["group_id"] == row["group_id"]}
    if len(world) != 5 or row["answer"] not in world:
        raise ValueError("dynamic twin world does not contain exactly five unique values")
    wrong = {item["answer"] for item in same_split if item["context_id"] == row["wrong_context_id"]}
    if len(wrong) != 4:
        raise ValueError("wrong context does not contain four unique values")
    global_values = set(world) | wrong
    pool = _ordered(row["id"], {item["answer"] for item in same_split})
    for value in pool:
        if len(global_values) == 16:
            break
        global_values.add(value)
    if len(global_values) != 16:
        raise ValueError("insufficient unique values for global16")
    result = {
        "world5": _ordered(row["id"], world),
        "global16": _ordered(row["id"], global_values),
    }
    if row["anchor_query"]:
        twin = by_id[row["twin_row_id"]]["answer"]
        if twin == row["answer"]:
            raise ValueError("anchor twin did not change value")
        result["pair2"] = _ordered(row["id"], {row["answer"], twin})
    if not set(result["world5"]).issubset(result["global16"]):
        raise ValueError("candidate sets are not nested")
    return result


def answer_vocabulary(rows: list[dict], tokenizer) -> dict[str, dict]:
    from tasks.train_cbf_event_content import split_answer

    vocab = {}
    for row in rows:
        spec = copy.deepcopy(row)
        digits, prefix = split_answer(tokenizer, spec)
        item = {"ids": row["answer_ids"], "digits": digits, "prefix": prefix}
        previous = vocab.setdefault(row["answer"], item)
        if previous != item or len(digits) != 5:
            raise ValueError("candidate answer tokenization is inconsistent")
    return vocab


def score_candidates(model, memory: dict, row: dict, candidates: list[str], vocab: dict[str, dict]) -> list[dict]:
    """Batch query-only reads by functionally replacing the seven effective down projections."""
    import torch
    import torch.nn.functional as F
    from torch.func import functional_call

    lengths = {len(vocab[value]["ids"]) for value in candidates}
    if lengths != {7}:
        raise ValueError("candidate answers must share the audited seven-token layout")
    tokens = torch.tensor(
        [row["query_ids"] + vocab[value]["ids"][:-1] for value in candidates],
        device=next(model.parameters()).device,
    )
    overrides = {}
    for layer, delta in memory.items():
        base = model.model.layers[layer].mlp.down_proj.weight
        overrides[f"layers.{layer}.mlp.down_proj.weight"] = base + delta
    hidden = functional_call(
        model.model, overrides, (tokens,), {"use_cache": False}, strict=False
    ).last_hidden_state
    answer_length = next(iter(lengths))
    logits = model.lm_head(hidden[:, len(row["query_ids"]) - 1:
                                  len(row["query_ids"]) - 1 + answer_length]).float()
    targets = torch.tensor([vocab[value]["ids"] for value in candidates], device=logits.device)
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                             reduction="none").reshape(len(candidates), answer_length)
    output = []
    for index, value in enumerate(candidates):
        digits = vocab[value]["digits"]
        output.append({
            "answer": value,
            "digit_nll": float(losses[index, digits].mean()),
            "answer_nll": float(losses[index].mean()),
        })
    return output


def ranking(scores: list[dict], members: list[str], truth: str) -> dict:
    selected = [item for item in scores if item["answer"] in members]
    if len(selected) != len(members) or truth not in members:
        raise ValueError("ranking candidate coverage mismatch")
    ordered = sorted(selected, key=lambda item: (item["digit_nll"], item["answer"]))
    true = next(item for item in selected if item["answer"] == truth)
    rank = next(index for index, item in enumerate(ordered, 1) if item["answer"] == truth)
    negative = min(item["digit_nll"] for item in selected if item["answer"] != truth)
    return {
        "top1": int(rank == 1),
        "mrr": 1.0 / rank,
        "rank": rank,
        "margin": negative - true["digit_nll"],
        "true_digit_nll": true["digit_nll"],
        "true_answer_nll": true["answer_nll"],
        "prediction": ordered[0]["answer"],
    }


def _bootstrap(values: list[float], seed: int, draws: int = 5000) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(statistics.fmean(values[rng.randrange(size)] for _ in range(size))
                       for _ in range(draws))
    return [estimates[int(.025 * draws)], estimates[min(draws - 1, int(.975 * draws))]]


def aggregate(records: list[dict]) -> dict:
    result = {}
    for split in ("train_probe", "dev"):
        subset = [row for row in records if row["evaluation_split"] == split]
        if not subset:
            raise ValueError(f"missing {split} ranking rows")
        split_result = {"rows": len(subset), "groups": len({row["group_id"] for row in subset}), "sets": {}}
        for set_name in SETS:
            available = [row for row in subset if set_name in row["candidate_sets"]]
            policies = {}
            for policy in POLICIES:
                policies[policy] = {
                    metric: statistics.fmean(row["policies"][policy]["rankings"][set_name][metric]
                                             for row in available)
                    for metric in RANK_METRICS
                }
            comparisons = {}
            for policy_index, policy in enumerate(("twin", "wrong", "empty")):
                group_values = []
                group_accuracy = []
                for group in sorted({row["group_id"] for row in available}):
                    group_rows = [row for row in available if row["group_id"] == group]
                    group_values.append(statistics.fmean(
                        row["policies"][policy]["rankings"][set_name]["true_digit_nll"]
                        - row["policies"]["correct"]["rankings"][set_name]["true_digit_nll"]
                        for row in group_rows
                    ))
                    group_accuracy.append(statistics.fmean(
                        row["policies"]["correct"]["rankings"][set_name]["top1"]
                        - row["policies"][policy]["rankings"][set_name]["top1"]
                        for row in group_rows
                    ))
                comparisons[policy] = {
                    "mean_true_digit_nll_gain": statistics.fmean(group_values),
                    "digit_gain_bootstrap_95ci": _bootstrap(group_values, 727 + policy_index),
                    "top1_advantage": statistics.fmean(group_accuracy),
                    "top1_advantage_bootstrap_95ci": _bootstrap(group_accuracy, 739 + policy_index),
                }
            split_result["sets"][set_name] = {
                "rows": len(available), "policies": policies, "correct_comparisons": comparisons,
            }
        result[split] = split_result
    return result


def bridge(model, memory: dict, row: dict, vocab: dict, eos: int) -> dict:
    import torch

    candidate = score_candidates(model, memory, row, [row["answer"]], vocab)[0]
    established = regions(model, memory, row, eos)
    error = {
        "answer_nll": abs(candidate["answer_nll"] - float(established["answer"])),
        "digit_nll": abs(candidate["digit_nll"] - float(established["digits"])),
    }
    if max(error.values()) > 2e-5:
        raise ValueError(f"functional candidate scorer bridge failed: {error}")
    return error


def load_writer(model, params: dict, checkpoint: Path, manifest: dict) -> dict:
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["manifest"] != manifest or set(saved["writer"]) != set(params):
        raise ValueError("writer checkpoint provenance mismatch")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(saved["writer"][name].to(parameter.device))
    return saved


def evaluate(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from cbf_ttt.event_writer import write_memory
    from tasks.cbf_selective import measure

    if args.arm not in ARMS:
        raise ValueError("unknown ranking arm")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_rows, data_manifest = load_rows(args.data_root, tokenizer)
    rows, groups = select_view(all_rows, 128, 32)
    train_groups = groups["train"][:8]
    selected = [row for row in rows if row["split"] == "dev" or
                (row["split"] == "train" and row["group_id"] in train_groups)]
    vocab = answer_vocabulary(rows, tokenizer)
    source = Path(args.source) / args.arm
    manifest = json.loads((source / "training_manifest.json").read_text())
    if manifest["data_sha256"] != data_manifest["data_sha256"] or manifest["checkpoints"] != list(EXPOSURES):
        raise ValueError("ranking source manifest mismatch")
    model, params, model_meta = setup(args)
    if model_meta["backbone_sha256"] != manifest["backbone_sha256"]:
        raise ValueError("ranking backbone mismatch")
    output = Path(args.output) / args.arm
    output.mkdir(parents=True)
    source_hashes = {str(exposure): sha(source / f"writer_{exposure}.pt") for exposure in EXPOSURES}
    run_manifest = {
        "protocol": "candidate_ranking_v1", "arm": args.arm, "exposures": list(EXPOSURES),
        "source": str(source), "source_training_manifest_sha256": sha(source / "training_manifest.json"),
        "source_checkpoints_sha256": source_hashes, "data_sha256": data_manifest["data_sha256"],
        "backbone_sha256": model_meta["backbone_sha256"], "candidate_sets": {"pair2": 2, "world5": 5, "global16": 16},
        "policies": list(POLICIES), "ranking_metric": "digit_nll", "bridge_tolerance": 2e-5,
        "groups": {"train_probe": train_groups, "dev": groups["dev"]}, "test_scored": False,
    }
    write_json(output / "manifest.json", run_manifest)
    all_contexts = {row["context_id"]: row["context_ids"] for row in rows}
    checkpoint_summaries = {}
    for exposure in EXPOSURES:
        load_writer(model, params, source / f"writer_{exposure}.pt", manifest)
        fixed = next(row for row in selected if row["split"] == "dev")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            fixed_memory = write_memory(model, all_contexts[fixed["context_id"]])
            bridges = {
                "correct": bridge(model, fixed_memory, fixed, vocab, data_manifest["eos_token_id"]),
                "empty": bridge(model, {}, fixed, vocab, data_manifest["eos_token_id"]),
            }
        records = []
        write_profiles = {}
        raw_path = output / f"ranking_{exposure}.jsonl"
        with raw_path.open("x") as sink, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for split in ("train", "dev"):
                split_rows = [row for row in selected if row["split"] == split]
                for group in sorted({row["group_id"] for row in split_rows}):
                    group_rows = [row for row in split_rows if row["group_id"] == group]
                    needed = {context_id for row in group_rows for context_id in
                              (row["context_id"], row["twin_context_id"], row["wrong_context_id"])}
                    memories = {}
                    for context_id in sorted(needed):
                        memories[context_id], profile = measure(
                            lambda context_id=context_id: write_memory(model, all_contexts[context_id]))
                        write_profiles[f"{group}:{context_id}"] = profile
                    for row in group_rows:
                        candidate_sets = build_candidate_sets(rows, row)
                        candidates = candidate_sets["global16"]
                        policies = {}
                        for policy in POLICIES:
                            context_id = (row["context_id"] if policy == "correct" else
                                          row["twin_context_id"] if policy == "twin" else
                                          row["wrong_context_id"] if policy == "wrong" else None)
                            memory = memories[context_id] if context_id else {}
                            scores, profile = measure(lambda memory=memory: score_candidates(
                                model, memory, row, candidates, vocab))
                            policies[policy] = {
                                "scores": scores,
                                "rankings": {name: ranking(scores, members, row["answer"])
                                             for name, members in candidate_sets.items()},
                                **profile,
                            }
                        record = {
                            "id": row["id"], "group_id": row["group_id"], "split": row["split"],
                            "evaluation_split": "dev" if row["split"] == "dev" else "train_probe",
                            "anchor_query": row["anchor_query"], "truth": row["answer"],
                            "candidate_sets": candidate_sets, "policies": policies,
                        }
                        sink.write(json.dumps(record) + "\n")
                        sink.flush()
                        records.append(record)
                    del memories, memory
                    torch.cuda.empty_cache()
        summary = aggregate(records)
        score_profiles = [row["policies"][policy] for row in records for policy in POLICIES]
        summary.update({
            "exposure": exposure, "bridges": bridges, "rows_sha256": sha(raw_path),
            "source_checkpoint_sha256": sha(source / f"writer_{exposure}.pt"),
            "write_profiles": write_profiles,
            "resources": {
                "max_score_seconds": max(item["seconds"] for item in score_profiles),
                "peak_allocated_gib": max(
                    [item["peak_allocated_gib"] for item in score_profiles]
                    + [item["peak_allocated_gib"] for item in write_profiles.values()]
                ),
                "peak_reserved_gib": max(
                    [item["peak_reserved_gib"] for item in score_profiles]
                    + [item["peak_reserved_gib"] for item in write_profiles.values()]
                ),
            },
            "test_scored": False,
        })
        write_json(output / f"ranking_{exposure}.summary.json", summary)
        checkpoint_summaries[str(exposure)] = summary
    if source_hashes != {str(exposure): sha(source / f"writer_{exposure}.pt") for exposure in EXPOSURES}:
        raise RuntimeError("source writer checkpoints changed during ranking")
    complete = {
        "arm": args.arm, "rows_per_checkpoint": len(selected), "checkpoints": list(EXPOSURES),
        "source_checkpoints_unchanged": True,
        "max_bridge_error": max(value for summary in checkpoint_summaries.values()
                                for policy in summary["bridges"].values() for value in policy.values()),
        "peak_allocated_gib": max(summary["resources"]["peak_allocated_gib"]
                                  for summary in checkpoint_summaries.values()),
        "peak_reserved_gib": max(summary["resources"]["peak_reserved_gib"]
                                 for summary in checkpoint_summaries.values()),
        "test_scored": False,
    }
    write_json(output / "complete.json", complete)
    return complete


def interpretation(binding_dev: dict) -> dict:
    target = binding_dev["sets"]["global16"]
    correct = target["policies"]["correct"]["top1"]
    wrong = target["policies"]["wrong"]["top1"]
    twin = target["policies"]["twin"]["top1"]
    wrong_gain = target["correct_comparisons"]["wrong"]
    twin_gain = target["correct_comparisons"]["twin"]
    strong = (correct >= .50 and correct - wrong >= .20 and correct - twin >= .20
              and wrong_gain["digit_gain_bootstrap_95ci"][0] > 0)
    weak = correct >= .25 and (wrong_gain["mean_true_digit_nll_gain"] > 0
                               or twin_gain["mean_true_digit_nll_gain"] > 0)
    label = "strong_latent_binding" if strong else "weak_latent_binding" if weak else "no_localized_binding"
    return {"label": label, "strong": strong, "weak": weak, "rule_is_diagnostic_only": True}


def summarize(root: str | Path, formal_summary: str | Path, save: bool = True) -> dict:
    root, formal_summary = Path(root), Path(formal_summary)
    fullkv_source = json.loads(formal_summary.read_text())
    arms = {}
    for arm in ARMS:
        output = root / arm
        manifest = json.loads((output / "manifest.json").read_text())
        complete = json.loads((output / "complete.json").read_text())
        checkpoints = {}
        for exposure in EXPOSURES:
            raw_path = output / f"ranking_{exposure}.jsonl"
            records = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
            stored = json.loads((output / f"ranking_{exposure}.summary.json").read_text())
            if stored["rows_sha256"] != sha(raw_path) or aggregate(records) != {
                key: stored[key] for key in ("train_probe", "dev")
            }:
                raise ValueError("candidate-ranking summary mismatch")
            checkpoints[str(exposure)] = stored
        arms[arm] = {"manifest": manifest, "complete": complete, "checkpoints": checkpoints}
    if arms["joint_ce"]["checkpoints"]["0"]["train_probe"] != arms["joint_binding"]["checkpoints"]["0"]["train_probe"] or \
       arms["joint_ce"]["checkpoints"]["0"]["dev"] != arms["joint_binding"]["checkpoints"]["0"]["dev"]:
        raise ValueError("candidate-ranking initial arms differ")
    fullkv = fullkv_source["arms"]["joint_binding"]["checkpoints"]["4096"]["dev"]["means"]["full_kv"]
    result = {
        "protocol": "candidate_ranking_v1", "arms": arms,
        "primary_interpretation": interpretation(arms["joint_binding"]["checkpoints"]["4096"]["dev"]),
        "audited_full_kv_control": {
            "source": str(formal_summary), "source_sha256": sha(formal_summary),
            "dev_code_correct": fullkv["code_correct"], "dev_digit_nll": fullkv["digit_nll"],
        },
        "test_scored": False, "terminal_status": "completed_candidate_ranking",
    }
    if save:
        write_json(root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--source", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--model", default="/home/ctj/models/Qwen3-4B")
    summary = sub.add_parser("summarize")
    summary.add_argument("--root", required=True)
    summary.add_argument("--formal-summary", required=True)
    args = parser.parse_args()
    result = evaluate(args) if args.command == "evaluate" else summarize(args.root, args.formal_summary)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
