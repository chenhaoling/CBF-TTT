"""Train a writer on counterfactually reassigned value-binding episodes."""

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
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_dynamic_memory import load_rows, select_view
from tasks.train_cbf_event_content import METRICS, extra_score, regions
from tasks.train_cbf_event_writer import setup, write_json
from tasks.train_cbf_listwise_binding import listwise_query


ARMS = ("fixed_assignment", "episodic_assignment")
POLICIES = ("correct", "counterfactual", "wrong", "empty")
SETS = ("context4", "global16")
EVAL_SHIFTS = (0, 2)


def _matches(sequence: list[int], pattern: list[int]) -> list[int]:
    return [index for index in range(len(sequence) - len(pattern) + 1)
            if sequence[index:index + len(pattern)] == pattern]


def rotation_episodes(rows: list[dict], tokenizer,
                      vocab: dict[str, dict]) -> dict[str, list[dict]]:
    """Create four coherent assignments for every fixed context and query set."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["context_id"]].append(row)
    episodes = {}
    for context_id, members in grouped.items():
        ordered = sorted(members, key=lambda row: row["id"])
        if len(ordered) != 4 or len({row["answer"] for row in ordered}) != 4:
            raise ValueError("rotation episodes require four unique values per context")
        original = list(ordered[0]["context_ids"])
        if any(row["context_ids"] != original for row in ordered):
            raise ValueError("context rows do not share packed input")
        suffix_start = len(original) - ordered[0]["fact_suffix_tokens"]
        patterns = [tokenizer.encode(" " + row["answer"], add_special_tokens=False)
                    for row in ordered]
        if len({len(pattern) for pattern in patterns}) != 1:
            raise ValueError("rotated values must have equal token lengths")
        spans = []
        for pattern in patterns:
            hits = [index for index in _matches(original, pattern) if index >= suffix_start]
            if len(hits) != 1:
                raise ValueError("rotated value boundary is ambiguous")
            spans.append((hits[0], len(pattern)))
        if len({start for start, _ in spans}) != 4:
            raise ValueError("rotated value spans overlap")
        variants = []
        for shift in range(4):
            context_ids = list(original)
            rotated_rows = []
            for index, ((start, length), pattern) in enumerate(zip(spans, patterns)):
                replacement = patterns[(index + shift) % 4]
                if len(replacement) != length:
                    raise ValueError("rotation changes packed context length")
                context_ids[start:start + length] = replacement
            for index, row in enumerate(ordered):
                target = ordered[(index + shift) % 4]["answer"]
                item = copy.deepcopy(row)
                item["answer"] = target
                item["answer_ids"] = list(vocab[target]["ids"])
                item["digit_positions"] = list(vocab[target]["digits"])
                item["prefix_positions"] = list(vocab[target]["prefix"])
                item["context_ids"] = context_ids
                item["assignment_shift"] = shift
                rotated_rows.append(item)
            variants.append({
                "context_id": context_id, "shift": shift,
                "context_ids": context_ids, "rows": rotated_rows,
                "targets": [row["answer"] for row in rotated_rows],
            })
        if len({tuple(variant["targets"]) for variant in variants}) != 4:
            raise ValueError("rotation did not create four distinct assignments")
        episodes[context_id] = variants
    return episodes


def episode_schedule(rows: list[dict], arm: str, rounds: int,
                     seed: int = 1021) -> list[tuple[str, int]]:
    if arm not in ARMS or rounds < 1:
        raise ValueError("invalid episodic schedule")
    contexts = sorted({row["context_id"] for row in rows if row["split"] == "train"})
    if not contexts:
        raise ValueError("empty episodic training split")
    rng = random.Random(seed)
    shift_orders = {}
    for context in contexts:
        order = list(range(4))
        rng.shuffle(order)
        shift_orders[context] = order
    batches = []
    for round_index in range(rounds):
        order = list(contexts)
        rng.shuffle(order)
        for context in order:
            shift = 0 if arm == "fixed_assignment" else shift_orders[context][round_index % 4]
            batches.append((context, shift))
    return batches


def _objective(items: list, balanced, lm_weight: float = .25):
    retrieval = sum(item[0] for item in items) / len(items)
    return retrieval + lm_weight * balanced, retrieval


def _prepare(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_rows, data_manifest = load_rows(args.data_root, tokenizer)
    rows, groups = select_view(all_rows, args.train_groups, args.dev_groups)
    vocab = answer_vocabulary(rows, tokenizer)
    episodes = rotation_episodes(rows, tokenizer, vocab)
    candidates = {
        context: sorted({row["answer"] for row in variants[0]["rows"]})
        for context, variants in episodes.items()
    }
    return tokenizer, rows, groups, data_manifest, vocab, episodes, candidates


def smoke(args) -> dict:
    import torch
    from cbf_ttt.event_writer import write_memory
    from tasks.cbf_selective import measure

    tokenizer, rows, _, data_manifest, vocab, episodes, candidates = _prepare(args)
    del tokenizer
    context, shift = episode_schedule(rows, args.arm, 1, args.order_seed)[0]
    episode = episodes[context][shift]
    model, params, model_meta = setup(args)
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.)

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            memory = write_memory(model, episode["context_ids"],
                                  max_relative_norm=args.memory_cap)
            values = [regions(model, memory, row, data_manifest["eos_token_id"])
                      for row in episode["rows"]]
            balanced = sum(.75 * value["digits"] + .125 * value["prefix"] + .125 * value["eos"]
                           for value in values) / len(values)
            items = [listwise_query(model, memory, row, candidates[context], vocab)
                     for row in episode["rows"]]
            loss, retrieval = _objective(items, balanced)
        loss.backward()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
               for parameter in params.values()):
            raise RuntimeError("invalid episodic smoke gradient")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.,
                                                              error_if_nonfinite=True))
        optimizer.step()
        relative = max(float(delta.detach().float().norm() /
                             model.model.layers[layer].mlp.down_proj.weight.float().norm())
                       for layer, delta in memory.items())
        return {
            "arm": args.arm, "shift": shift, "loss": float(loss.detach()),
            "retrieval_loss": float(retrieval.detach()),
            "balanced_loss": float(balanced.detach()),
            "listwise_accuracy": statistics.fmean(item[2] for item in items),
            "unclipped_gradient_norm": gradient_norm, "max_relative_delta": relative,
            "backbone_sha256": model_meta["backbone_sha256"],
            "memory_cap": args.memory_cap, "test_scored": False,
        }

    result, profile = measure(one_step)
    result.update(profile)
    write_json(args.output, result)
    return result


def _ranking_aggregate(records: list[dict]) -> dict:
    result = {}
    for split in ("train_probe", "dev"):
        subset = [row for row in records if row["evaluation_split"] == split]
        if not subset:
            raise ValueError(f"missing episodic {split} rows")
        generation = {
            policy: {metric: statistics.fmean(row["policies"][policy]["generation"][metric]
                                               for row in subset)
                     for metric in METRICS}
            for policy in POLICIES + ("full_kv",)
        }
        sets = {}
        for set_name in SETS:
            policies = {
                policy: {metric: statistics.fmean(
                    row["policies"][policy]["rankings"][set_name][metric]
                    for row in subset) for metric in RANK_METRICS}
                for policy in POLICIES
            }
            comparisons = {}
            groups = sorted({row["group_id"] for row in subset})
            for index, policy in enumerate(("counterfactual", "wrong", "empty")):
                nll, top1 = [], []
                for group in groups:
                    group_rows = [row for row in subset if row["group_id"] == group]
                    nll.append(statistics.fmean(
                        row["policies"][policy]["rankings"][set_name]["true_digit_nll"]
                        - row["policies"]["correct"]["rankings"][set_name]["true_digit_nll"]
                        for row in group_rows))
                    top1.append(statistics.fmean(
                        row["policies"]["correct"]["rankings"][set_name]["top1"]
                        - row["policies"][policy]["rankings"][set_name]["top1"]
                        for row in group_rows))
                comparisons[policy] = {
                    "mean_true_digit_nll_gain": statistics.fmean(nll),
                    "digit_gain_bootstrap_95ci": _bootstrap(nll, 1103 + index),
                    "top1_advantage": statistics.fmean(top1),
                    "top1_advantage_bootstrap_95ci": _bootstrap(top1, 1123 + index),
                }
            sets[set_name] = {"policies": policies, "correct_comparisons": comparisons}
        result[split] = {
            "rows": len(subset), "groups": len({row["group_id"] for row in subset}),
            "generation": generation, "sets": sets,
        }
    return result


def episodic_gate(dev: dict) -> dict:
    local = dev["sets"]["context4"]
    global_set = dev["sets"]["global16"]
    comparison = local["correct_comparisons"]["counterfactual"]
    correct = local["policies"]["correct"]["top1"]
    checks = {
        "full_kv_readable": dev["generation"]["full_kv"]["code_correct"] >= .80,
        "context4_correct_top1": correct >= .60,
        "context4_counterfactual_gap": (
            correct - local["policies"]["counterfactual"]["top1"] >= .25),
        "context4_counterfactual_nll_confidence": comparison["digit_gain_bootstrap_95ci"][0] > 0,
        "global16_correct_top1": global_set["policies"]["correct"]["top1"] >= .25,
    }
    return {"checks": checks, "passed": all(checks.values())}


def evaluate(model, tokenizer, rows: list[dict], data_manifest: dict, output: Path,
             exposure: int, vocab: dict[str, dict], episodes: dict[str, list[dict]],
             candidates: dict[str, list[str]], memory_cap: float) -> dict:
    import torch
    from cbf_ttt.event_writer import write_memory
    from tasks.cbf_selective import measure

    train_groups = sorted({row["group_id"] for row in rows if row["split"] == "train"})[:8]
    selected = [row for row in rows if row["split"] == "dev" or
                (row["split"] == "train" and row["group_id"] in train_groups)]
    records, profiles = [], []
    path = output / f"evaluation_{exposure}.jsonl"
    with path.open("x") as sink, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for split in ("train", "dev"):
            split_rows = [row for row in selected if row["split"] == split]
            for group in sorted({row["group_id"] for row in split_rows}):
                group_rows = [row for row in split_rows if row["group_id"] == group]
                contexts = sorted({row["context_id"] for row in group_rows})
                wrong_contexts = sorted({row["wrong_context_id"] for row in group_rows})
                memories = {}
                for context in contexts + wrong_contexts:
                    for shift in set(EVAL_SHIFTS) | {(item + 1) % 4 for item in EVAL_SHIFTS}:
                        key = (context, shift)
                        if key not in memories:
                            memories[key], profile = measure(lambda context=context, shift=shift:
                                write_memory(model, episodes[context][shift]["context_ids"],
                                             max_relative_norm=memory_cap))
                            profiles.append(profile)
                for base_row in group_rows:
                    context = base_row["context_id"]
                    row_index = next(index for index, row in enumerate(episodes[context][0]["rows"])
                                     if row["id"] == base_row["id"])
                    global_candidates = build_candidate_sets(rows, base_row)["global16"]
                    for shift in EVAL_SHIFTS:
                        row = episodes[context][shift]["rows"][row_index]
                        policy_memories = {
                            "correct": memories[(context, shift)],
                            "counterfactual": memories[(context, (shift + 1) % 4)],
                            "wrong": memories[(base_row["wrong_context_id"], shift)],
                            "empty": {},
                        }
                        policies = {}
                        for policy, memory in policy_memories.items():
                            def score_policy(memory=memory):
                                generation = extra_score(model, tokenizer, row, memory,
                                                         data_manifest["eos_token_id"], False)
                                scores = score_candidates(model, memory, row, global_candidates, vocab)
                                return generation, scores
                            (generation, scores), profile = measure(score_policy)
                            profiles.append(profile)
                            policies[policy] = {
                                "generation": generation,
                                "rankings": {
                                    "context4": ranking(scores, candidates[context], row["answer"]),
                                    "global16": ranking(scores, global_candidates, row["answer"]),
                                },
                            }
                        full_kv, profile = measure(lambda: extra_score(
                            model, tokenizer, row, {}, data_manifest["eos_token_id"], True))
                        profiles.append(profile)
                        policies["full_kv"] = {"generation": full_kv}
                        record = {
                            "id": row["id"], "group_id": row["group_id"], "split": row["split"],
                            "evaluation_split": "dev" if row["split"] == "dev" else "train_probe",
                            "context_id": context, "assignment_shift": shift,
                            "truth": row["answer"], "policies": policies,
                        }
                        sink.write(json.dumps(record) + "\n")
                        sink.flush()
                        records.append(record)
                del memories
                torch.cuda.empty_cache()
    result = _ranking_aggregate(records)
    result.update({
        "exposure": exposure, "episodic_gate": episodic_gate(result["dev"]),
        "rows_sha256": sha(path), "checkpoint_sha256": sha(output / f"writer_{exposure}.pt"),
        "resources": {
            "peak_allocated_gib": max(item["peak_allocated_gib"] for item in profiles),
            "peak_reserved_gib": max(item["peak_reserved_gib"] for item in profiles),
        },
        "test_scored": False,
    })
    write_json(output / f"evaluation_{exposure}.summary.json", result)
    return result


def run(args) -> dict:
    import torch
    from cbf_ttt.event_writer import backbone_digest, write_memory
    from tasks.cbf_selective import measure

    tokenizer, rows, groups, data_manifest, vocab, episodes, candidates = _prepare(args)
    batches = episode_schedule(rows, args.arm, args.rounds, args.order_seed)
    total_exposures = 4 * len(batches)
    checkpoints = (0, total_exposures)
    output = Path(args.output)
    output.mkdir(parents=True)
    model, params, model_meta = setup(args)
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.)
    versions = {name: parameter._version for name, parameter in model.named_parameters()
                if not parameter.requires_grad}
    initial = {name: parameter.detach().cpu().clone() for name, parameter in params.items()}
    manifest = {
        **model_meta, "protocol": "episodic_binding_v1", "arm": args.arm,
        "rounds": args.rounds, "question_exposures": total_exposures,
        "optimizer_updates": len(batches), "context_writes_per_update": 1,
        "query_reads_per_update": 4, "listwise_candidates": 4,
        "checkpoints": list(checkpoints), "evaluation_shifts": list(EVAL_SHIFTS),
        "seed": 301, "order_seed": args.order_seed, "lr": args.lr,
        "clip": 1.0, "weight_decay": 0.0, "groups": groups,
        "data_sha256": data_manifest["data_sha256"], "train_groups": args.train_groups,
        "dev_groups": args.dev_groups, "memory_cap": args.memory_cap,
        "lm_weight": .25, "assignment_policy": (
            "shift_zero_only" if args.arm == "fixed_assignment" else "all_four_rotations_per_context"),
        "test_scored": False,
    }
    write_json(output / "training_manifest.json", manifest)

    def save(exposure: int):
        path = output / f"writer_{exposure}.pt"
        temp = Path(str(path) + ".tmp")
        torch.save({"exposure": exposure,
                    "writer": {name: value.detach().cpu() for name, value in params.items()},
                    "manifest": manifest}, temp)
        temp.replace(path)

    save(0)
    evaluate(model, tokenizer, rows, data_manifest, output, 0, vocab, episodes,
             candidates, args.memory_cap)
    exposure, records = 0, []
    with (output / "steps.jsonl").open("x") as sink:
        for update, (context, shift) in enumerate(batches, 1):
            episode = episodes[context][shift]

            def one_step():
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    memory = write_memory(model, episode["context_ids"],
                                          max_relative_norm=args.memory_cap)
                    values = [regions(model, memory, row, data_manifest["eos_token_id"])
                              for row in episode["rows"]]
                    balanced = sum(.75 * value["digits"] + .125 * value["prefix"] + .125 * value["eos"]
                                   for value in values) / len(values)
                    items = [listwise_query(model, memory, row, candidates[context], vocab)
                             for row in episode["rows"]]
                    loss, retrieval = _objective(items, balanced)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite episodic loss")
                relative = max(float(delta.detach().float().norm() /
                                     model.model.layers[layer].mlp.down_proj.weight.float().norm())
                               for layer, delta in memory.items())
                if relative > 1:
                    raise RuntimeError("episodic delta exceeds safety tripwire")
                loss.backward()
                gradients = {name: float(parameter.grad.norm()) if parameter.grad is not None else None
                             for name, parameter in params.items()}
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in params.values()):
                    raise RuntimeError("disconnected or nonfinite episodic gradient")
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.,
                                                                      error_if_nonfinite=True))
                optimizer.step()
                return {
                    "update": update, "exposure_start": exposure,
                    "exposure_end": exposure + 4, "context_id": context,
                    "assignment_shift": shift, "targets": episode["targets"],
                    "loss": float(loss.detach()), "retrieval_loss": float(retrieval.detach()),
                    "balanced_loss": float(balanced.detach()),
                    "listwise_accuracy": statistics.fmean(item[2] for item in items),
                    "gradient_norms": gradients, "unclipped_gradient_norm": gradient_norm,
                    "max_relative_delta": relative,
                }

            record, profile = measure(one_step)
            exposure += 4
            record.update(profile)
            records.append(record)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if update % 32 == 0:
                print(json.dumps({key: record[key] for key in
                                  ("update", "exposure_end", "assignment_shift", "loss",
                                   "retrieval_loss", "listwise_accuracy", "seconds")}), flush=True)
    save(exposure)
    evaluate(model, tokenizer, rows, data_manifest, output, exposure, vocab, episodes,
             candidates, args.memory_cap)
    if exposure != total_exposures:
        raise RuntimeError("episodic exposure accounting mismatch")
    if versions != {name: parameter._version for name, parameter in model.named_parameters()
                    if not parameter.requires_grad} or backbone_digest(model) != model_meta["backbone_sha256"]:
        raise RuntimeError("frozen backbone changed")
    changes = {name: float((parameter.detach().cpu() - initial[name]).norm())
               for name, parameter in params.items()}
    if any(value <= 0 for value in changes.values()):
        raise RuntimeError("episodic writer parameter did not change")
    if not all(value.dtype == torch.float32 for state in optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("episodic optimizer state is not FP32")
    torch.save(optimizer.state_dict(), output / f"optimizer_{total_exposures}.pt")
    complete = {
        "arm": args.arm, "question_exposures": total_exposures,
        "optimizer_updates": len(batches), "backbone_unchanged": True,
        "optimizer_state_fp32": True, "writer_update_norms": changes,
        "first32_retrieval_loss": statistics.fmean(row["retrieval_loss"] for row in records[:32]),
        "last32_retrieval_loss": statistics.fmean(row["retrieval_loss"] for row in records[-32:]),
        "first32_listwise_accuracy": statistics.fmean(row["listwise_accuracy"] for row in records[:32]),
        "last32_listwise_accuracy": statistics.fmean(row["listwise_accuracy"] for row in records[-32:]),
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
            raw = output / f"evaluation_{exposure}.jsonl"
            records = [json.loads(line) for line in raw.read_text().splitlines() if line]
            stored = json.loads((output / f"evaluation_{exposure}.summary.json").read_text())
            if stored["rows_sha256"] != sha(raw) or _ranking_aggregate(records) != {
                    key: stored[key] for key in ("train_probe", "dev")}:
                raise ValueError("episodic evaluation mismatch")
            checkpoints[str(exposure)] = stored
        final = checkpoints[str(manifest["question_exposures"])]
        arms[arm] = {"manifest": manifest, "complete": complete,
                     "checkpoints": checkpoints, "final_gate": final["episodic_gate"]}
    for key in ("seed", "order_seed", "data_sha256", "backbone_sha256", "question_exposures"):
        if arms[ARMS[0]]["manifest"][key] != arms[ARMS[1]]["manifest"][key]:
            raise ValueError("episodic arms are not matched")
    result = {
        "protocol": "episodic_binding_v1", "arms": arms,
        "primary_arm": "episodic_assignment",
        "stage_passed": arms["episodic_assignment"]["final_gate"]["passed"],
        "test_scored": False, "terminal_status": "completed_episodic_binding",
    }
    if save:
        write_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "run"):
        item = sub.add_parser(name)
        item.add_argument("--data-root", required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--model", default="/home/ctj/models/Qwen3-4B")
        item.add_argument("--arm", choices=ARMS, required=True)
        item.add_argument("--train-groups", type=int, default=32)
        item.add_argument("--dev-groups", type=int, default=8)
        item.add_argument("--order-seed", type=int, default=1021)
        item.add_argument("--lr", type=float, default=1e-7)
        item.add_argument("--memory-cap", type=float, default=.3)
        if name == "run":
            item.add_argument("--rounds", type=int, default=4)
    summary = sub.add_parser("summarize")
    summary.add_argument("--root", required=True)
    args = parser.parse_args()
    result = smoke(args) if args.command == "smoke" else run(args) if args.command == "run" else summarize(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
