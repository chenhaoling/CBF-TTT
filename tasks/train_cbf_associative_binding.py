"""Matched all-fact permutation contrast pilot for dynamic writer binding."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from tasks.evaluate_cbf_candidate_ranking import (
    RANK_METRICS, answer_vocabulary, build_candidate_sets, ranking, score_candidates,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_dynamic_memory import load_rows, select_view
from tasks.train_cbf_event_content import METRICS, extra_score, regions
from tasks.train_cbf_event_writer import setup, write_json


ARMS = ("anchor_twin", "all_fact_permutation")
POLICIES = ("correct", "permuted", "twin", "wrong", "empty")


def schedule(rows: list[dict], rounds: int, seed: int = 811) -> list[list[str]]:
    by_context = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            by_context[row["context_id"]].append(row["id"])
    if rounds < 1 or any(len(ids) != 4 for ids in by_context.values()):
        raise ValueError("associative pilot requires four questions per context")
    rng = random.Random(seed)
    batches = []
    for _ in range(rounds):
        contexts = sorted(by_context)
        rng.shuffle(contexts)
        batches.extend([sorted(by_context[context]) for context in contexts])
    return batches


def _matches(sequence: list[int], pattern: list[int]) -> list[int]:
    return [index for index in range(len(sequence) - len(pattern) + 1)
            if sequence[index:index + len(pattern)] == pattern]


def permuted_contexts(rows: list[dict], tokenizer) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Cyclically permute four queried values while preserving all other tokens."""
    by_context = defaultdict(list)
    for row in rows:
        by_context[row["context_id"]].append(row)
    contexts = {}
    assigned = {}
    for context_id, context_rows in by_context.items():
        ordered = sorted(context_rows, key=lambda row: row["id"])
        if len(ordered) != 4 or len({row["answer"] for row in ordered}) != 4:
            raise ValueError("permutation requires four unique queried values")
        original = list(ordered[0]["context_ids"])
        suffix_start = len(original) - ordered[0]["fact_suffix_tokens"]
        patterns = [tokenizer.encode(" " + row["answer"], add_special_tokens=False) for row in ordered]
        if len({len(pattern) for pattern in patterns}) != 1:
            raise ValueError("permuted values must have equal token length")
        spans = []
        for row, pattern in zip(ordered, patterns):
            hits = [index for index in _matches(original, pattern) if index >= suffix_start]
            if len(hits) != 1:
                raise ValueError(f"value token boundary is ambiguous for {row['id']}")
            spans.append((hits[0], len(pattern)))
        if len({start for start, _ in spans}) != 4:
            raise ValueError("permutation spans overlap")
        rotated = patterns[1:] + patterns[:1]
        result = list(original)
        for (start, length), replacement in zip(spans, rotated):
            if len(replacement) != length:
                raise ValueError("permutation changed context length")
            result[start:start + length] = replacement
        for index, row in enumerate(ordered):
            assigned[row["id"]] = ordered[(index + 1) % len(ordered)]["answer"]
            if assigned[row["id"]] == row["answer"]:
                raise ValueError("permutation left a queried value fixed")
        if len(result) != len(original) or result == original:
            raise ValueError("invalid permuted context")
        contexts[context_id] = result
    return contexts, assigned


def objective(values: list[dict], negatives: list[dict], arm: str,
              margin: float = .2, contrast_weight: float = .5):
    import torch
    import torch.nn.functional as F

    if len(values) != 4:
        raise ValueError("objective requires four positive questions")
    balanced = sum(.75 * value["digits"] + .125 * value["prefix"] + .125 * value["eos"]
                   for value in values) / len(values)
    if arm == "anchor_twin":
        indexes = [index for index, value in enumerate(values) if value["anchor"]]
        if len(indexes) != 1 or len(negatives) != 1:
            raise ValueError("anchor arm requires one twin contrast")
        selected = [values[indexes[0]]]
    elif arm == "all_fact_permutation":
        if len(negatives) != len(values):
            raise ValueError("all-fact arm requires four permutation contrasts")
        selected = values
    else:
        raise ValueError("unknown associative arm")
    terms = [F.softplus(margin + positive["digits"] - negative["digits"])
             for positive, negative in zip(selected, negatives)]
    contrast = sum(terms) / len(terms)
    return balanced + contrast_weight * contrast, balanced, contrast, terms


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
            raise ValueError(f"missing {split} rows")
        generation = {
            policy: {metric: statistics.fmean(row["policies"][policy]["generation"][metric]
                                               for row in subset)
                     for metric in METRICS}
            for policy in POLICIES + ("full_kv",)
        }
        ranking_means = {
            policy: {metric: statistics.fmean(row["policies"][policy]["global16"][metric]
                                               for row in subset)
                     for metric in RANK_METRICS}
            for policy in POLICIES
        }
        comparisons = {}
        groups = sorted({row["group_id"] for row in subset})
        for index, policy in enumerate(("permuted", "twin", "wrong", "empty")):
            nll, accuracy = [], []
            for group in groups:
                group_rows = [row for row in subset if row["group_id"] == group]
                nll.append(statistics.fmean(
                    row["policies"][policy]["global16"]["true_digit_nll"]
                    - row["policies"]["correct"]["global16"]["true_digit_nll"]
                    for row in group_rows))
                accuracy.append(statistics.fmean(
                    row["policies"]["correct"]["global16"]["top1"]
                    - row["policies"][policy]["global16"]["top1"]
                    for row in group_rows))
            comparisons[policy] = {
                "mean_true_digit_nll_gain": statistics.fmean(nll),
                "digit_gain_bootstrap_95ci": _bootstrap(nll, 823 + index),
                "top1_advantage": statistics.fmean(accuracy),
                "top1_advantage_bootstrap_95ci": _bootstrap(accuracy, 839 + index),
            }
        result[split] = {
            "rows": len(subset), "groups": len(groups), "generation": generation,
            "global16": {"policies": ranking_means, "correct_comparisons": comparisons},
        }
    return result


def ranking_gate(dev: dict) -> dict:
    target = dev["global16"]
    correct = target["policies"]["correct"]["top1"]
    checks = {
        "full_kv_readable": dev["generation"]["full_kv"]["code_correct"] >= .80,
        "correct_top1": correct >= .25,
    }
    for policy in ("permuted", "wrong"):
        comparison = target["correct_comparisons"][policy]
        checks[f"{policy}_top1_separation"] = (
            correct - target["policies"][policy]["top1"] >= .10)
        checks[f"{policy}_nll_confidence"] = comparison["digit_gain_bootstrap_95ci"][0] > 0
    return {"checks": checks, "passed": all(checks.values())}


def evaluate(model, tokenizer, rows: list[dict], data_manifest: dict, output: Path,
             exposure: int, permutation: dict[str, list[int]], vocab: dict[str, dict]) -> dict:
    import torch
    from cbf_ttt.event_writer import write_memory
    from tasks.cbf_selective import measure

    train_groups = sorted({row["group_id"] for row in rows if row["split"] == "train"})[:8]
    selected = [copy.deepcopy(row) for row in rows if row["split"] == "dev" or
                (row["split"] == "train" and row["group_id"] in train_groups)]
    all_contexts = {row["context_id"]: row["context_ids"] for row in rows}
    records = []
    write_profiles = []
    score_profiles = []
    path = output / f"evaluation_{exposure}.jsonl"
    with path.open("x") as sink, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for split in ("train", "dev"):
            split_rows = [row for row in selected if row["split"] == split]
            for group in sorted({row["group_id"] for row in split_rows}):
                group_rows = [row for row in split_rows if row["group_id"] == group]
                needed = {context_id for row in group_rows for context_id in
                          (row["context_id"], row["twin_context_id"], row["wrong_context_id"])}
                memories = {}
                permuted_memories = {}
                for context_id in sorted(needed):
                    memories[context_id], profile = measure(
                        lambda context_id=context_id: write_memory(model, all_contexts[context_id]))
                    write_profiles.append(profile)
                for context_id in sorted({row["context_id"] for row in group_rows}):
                    permuted_memories[context_id], profile = measure(
                        lambda context_id=context_id: write_memory(model, permutation[context_id]))
                    write_profiles.append(profile)
                for row in group_rows:
                    candidates = build_candidate_sets(rows, row)["global16"]
                    policies = {}
                    for policy in POLICIES:
                        if policy == "correct":
                            memory = memories[row["context_id"]]
                        elif policy == "permuted":
                            memory = permuted_memories[row["context_id"]]
                        elif policy == "twin":
                            memory = memories[row["twin_context_id"]]
                        elif policy == "wrong":
                            memory = memories[row["wrong_context_id"]]
                        else:
                            memory = {}
                        def score_policy():
                            generation = extra_score(model, tokenizer, row, memory,
                                                     data_manifest["eos_token_id"], False)
                            scores = score_candidates(model, memory, row, candidates, vocab)
                            return generation, scores

                        (generation, scores), profile = measure(score_policy)
                        score_profiles.append(profile)
                        policies[policy] = {
                            "generation": generation,
                            "global16": ranking(scores, candidates, row["answer"]),
                            "profile": profile,
                        }
                    full_kv, profile = measure(lambda: extra_score(
                        model, tokenizer, row, {}, data_manifest["eos_token_id"], True))
                    score_profiles.append(profile)
                    policies["full_kv"] = {"generation": full_kv, "profile": profile}
                    record = {
                        "id": row["id"], "group_id": row["group_id"], "split": row["split"],
                        "evaluation_split": "dev" if row["split"] == "dev" else "train_probe",
                        "anchor_query": row["anchor_query"], "truth": row["answer"],
                        "policies": policies,
                    }
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                    records.append(record)
                del memories, permuted_memories, memory
                torch.cuda.empty_cache()
    result = aggregate(records)
    result.update({
        "exposure": exposure, "ranking_gate": ranking_gate(result["dev"]),
        "rows_sha256": sha(path), "checkpoint_sha256": sha(output / f"writer_{exposure}.pt"),
        "resources": {
            "peak_allocated_gib": max(item["peak_allocated_gib"]
                                      for item in write_profiles + score_profiles),
            "peak_reserved_gib": max(item["peak_reserved_gib"]
                                     for item in write_profiles + score_profiles),
        },
        "test_scored": False,
    })
    write_json(output / f"evaluation_{exposure}.summary.json", result)
    return result


def run(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from cbf_ttt.event_writer import backbone_digest, write_memory
    from tasks.cbf_selective import measure

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_rows, data_manifest = load_rows(args.data_root, tokenizer)
    rows, selected_groups = select_view(all_rows, args.train_groups, args.dev_groups)
    permutation, assigned = permuted_contexts(rows, tokenizer)
    vocab = answer_vocabulary(rows, tokenizer)
    batches = schedule(rows, args.rounds, args.order_seed)
    total_exposures = sum(map(len, batches))
    checkpoints = (0, total_exposures // 2, total_exposures)
    if total_exposures % 8 or any(point % 4 for point in checkpoints):
        raise ValueError("associative checkpoint alignment mismatch")
    output = Path(args.output)
    output.mkdir(parents=True)
    model, params, model_meta = setup(args)
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.)
    versions = {name: parameter._version for name, parameter in model.named_parameters()
                if not parameter.requires_grad}
    initial = {name: parameter.detach().cpu().clone() for name, parameter in params.items()}
    manifest = {
        **model_meta, "protocol": "associative_binding_v1", "arm": args.arm,
        "rounds": args.rounds, "question_exposures": total_exposures,
        "optimizer_updates": len(batches), "context_writes_per_update": 2,
        "checkpoints": list(checkpoints), "seed": 301, "order_seed": args.order_seed,
        "lr": args.lr, "clip": 1.0, "weight_decay": 0.0, "groups": selected_groups,
        "data_sha256": data_manifest["data_sha256"], "train_groups": args.train_groups,
        "dev_groups": args.dev_groups, "contrast_questions": 1 if args.arm == "anchor_twin" else 4,
        "permutation": "cyclic_by_sorted_row_id", "margin": .2, "contrast_weight": .5,
        "test_scored": False,
    }
    write_json(output / "training_manifest.json", manifest)
    by_id = {row["id"]: row for row in rows}
    by_context = {row["context_id"]: row["context_ids"] for row in rows}

    def save(exposure: int) -> None:
        path = output / f"writer_{exposure}.pt"
        temp = Path(str(path) + ".tmp")
        torch.save({"exposure": exposure,
                    "writer": {name: value.detach().cpu() for name, value in params.items()},
                    "manifest": manifest}, temp)
        temp.replace(path)

    save(0)
    evaluate(model, tokenizer, rows, data_manifest, output, 0, permutation, vocab)
    exposure = 0
    records = []
    with (output / "steps.jsonl").open("x") as sink:
        for update, batch in enumerate(batches, 1):
            selected = [by_id[sample_id] for sample_id in batch]

            def one_step() -> dict:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    context_id = selected[0]["context_id"]
                    if any(row["context_id"] != context_id for row in selected):
                        raise ValueError("joint batch mixes contexts")
                    memory = write_memory(model, by_context[context_id])
                    values = []
                    for row in selected:
                        value = regions(model, memory, row, data_manifest["eos_token_id"])
                        value["anchor"] = row["anchor_query"]
                        values.append(value)
                    if args.arm == "anchor_twin":
                        anchor = next(row for row in selected if row["anchor_query"])
                        negative_memory = write_memory(model, by_context[anchor["twin_context_id"]])
                        negatives = [regions(model, negative_memory, anchor,
                                             data_manifest["eos_token_id"])]
                    else:
                        negative_memory = write_memory(model, permutation[context_id])
                        negatives = [regions(model, negative_memory, row,
                                             data_manifest["eos_token_id"]) for row in selected]
                    loss, balanced, contrast, terms = objective(values, negatives, args.arm)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite associative loss")
                relative = max(float(delta.detach().float().norm() /
                                     model.model.layers[layer].mlp.down_proj.weight.float().norm())
                               for item in (memory, negative_memory) for layer, delta in item.items())
                if relative > 1:
                    raise RuntimeError("associative delta exceeds safety tripwire")
                loss.backward()
                gradients = {name: float(parameter.grad.norm()) if parameter.grad is not None else None
                             for name, parameter in params.items()}
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in params.values()):
                    raise RuntimeError("disconnected or nonfinite associative gradient")
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.,
                                                                      error_if_nonfinite=True))
                optimizer.step()
                return {
                    "update": update, "exposure_start": exposure,
                    "exposure_end": exposure + len(selected), "ids": batch,
                    "context_id": selected[0]["context_id"], "loss": float(loss.detach()),
                    "balanced_loss": float(balanced.detach()),
                    "contrast_loss": float(contrast.detach()),
                    "contrast_terms": [float(term.detach()) for term in terms],
                    "permuted_answers": [assigned[row["id"]] for row in selected],
                    "gradient_norms": gradients, "unclipped_gradient_norm": gradient_norm,
                    "max_relative_delta": relative,
                }

            record, profile = measure(one_step)
            exposure += len(selected)
            record.update(profile)
            records.append(record)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if exposure in checkpoints[1:]:
                save(exposure)
                evaluate(model, tokenizer, rows, data_manifest, output, exposure, permutation, vocab)
            if update % 32 == 0:
                print(json.dumps({key: record[key] for key in
                                  ("update", "exposure_end", "loss", "contrast_loss", "seconds")}),
                      flush=True)
    if exposure != total_exposures:
        raise RuntimeError("associative exposure accounting mismatch")
    if versions != {name: parameter._version for name, parameter in model.named_parameters()
                    if not parameter.requires_grad} or backbone_digest(model) != model_meta["backbone_sha256"]:
        raise RuntimeError("frozen backbone changed")
    changes = {name: float((parameter.detach().cpu() - initial[name]).norm())
               for name, parameter in params.items()}
    if any(value <= 0 for value in changes.values()):
        raise RuntimeError("associative parameter did not change")
    if not all(value.dtype == torch.float32 for state in optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("associative optimizer state is not FP32")
    torch.save(optimizer.state_dict(), output / f"optimizer_{total_exposures}.pt")
    complete = {
        "arm": args.arm, "question_exposures": total_exposures,
        "optimizer_updates": len(batches), "backbone_unchanged": True,
        "optimizer_state_fp32": True, "writer_update_norms": changes,
        "first32_loss": statistics.fmean(row["loss"] for row in records[:32]),
        "last32_loss": statistics.fmean(row["loss"] for row in records[-32:]),
        "mean_update_seconds": statistics.fmean(row["seconds"] for row in records),
        "peak_allocated_gib": max(row["peak_allocated_gib"] for row in records),
        "peak_reserved_gib": max(row["peak_reserved_gib"] for row in records),
        "max_relative_delta": max(row["max_relative_delta"] for row in records),
        "steps_sha256": sha(output / "steps.jsonl"), "test_scored": False,
    }
    write_json(output / "complete.json", complete)
    return complete


def summarize(root: str | Path, save: bool = True) -> dict:
    root = Path(root)
    arms = {}
    for arm in ARMS:
        output = root / arm
        manifest = json.loads((output / "training_manifest.json").read_text())
        complete = json.loads((output / "complete.json").read_text())
        checkpoints = {}
        for exposure in manifest["checkpoints"]:
            raw_path = output / f"evaluation_{exposure}.jsonl"
            records = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
            stored = json.loads((output / f"evaluation_{exposure}.summary.json").read_text())
            if stored["rows_sha256"] != sha(raw_path) or aggregate(records) != {
                    key: stored[key] for key in ("train_probe", "dev")}:
                raise ValueError("associative evaluation mismatch")
            checkpoints[str(exposure)] = stored
        arms[arm] = {"manifest": manifest, "complete": complete, "checkpoints": checkpoints,
                     "final_gate": checkpoints[str(manifest["question_exposures"])]["ranking_gate"]}
    for key in ("seed", "order_seed", "data_sha256", "backbone_sha256", "question_exposures"):
        if arms[ARMS[0]]["manifest"][key] != arms[ARMS[1]]["manifest"][key]:
            raise ValueError("associative arms are not matched")
    result = {
        "protocol": "associative_binding_v1", "arms": arms,
        "stage_passed": arms["all_fact_permutation"]["final_gate"]["passed"],
        "primary_arm": "all_fact_permutation", "test_scored": False,
        "terminal_status": "completed_associative_binding_pilot",
    }
    if save:
        write_json(root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("run")
    train.add_argument("--data-root", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--model", default="/home/ctj/models/Qwen3-4B")
    train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--rounds", type=int, default=4)
    train.add_argument("--train-groups", type=int, default=32)
    train.add_argument("--dev-groups", type=int, default=8)
    train.add_argument("--order-seed", type=int, default=811)
    train.add_argument("--lr", type=float, default=1e-7)
    summary = sub.add_parser("summarize")
    summary.add_argument("--root", required=True)
    args = parser.parse_args()
    result = run(args) if args.command == "run" else summarize(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
