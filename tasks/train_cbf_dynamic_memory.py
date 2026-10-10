"""Matched-exposure sequential versus same-context writer training on dynamic worlds."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_content import METRICS, extra_score, regions, split_answer
from tasks.train_cbf_event_writer import setup, write_json


ARMS = ("sequential", "joint_context")


def load_rows(data_root: str | Path, tokenizer) -> tuple[list[dict], dict]:
    data_root = Path(data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    path = data_root / "rows.jsonl"
    if manifest["protocol"] != "dynamic_memory_writer_v1" or sha(path) != manifest["data_sha256"]:
        raise ValueError("dynamic packed data manifest mismatch")
    if manifest["chunk"] != 4096 or manifest["test_tokenized"]:
        raise ValueError("dynamic pilot requires 4096-token train/dev data and sealed test")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    expected = sum(manifest["rows"].values())
    if len(rows) != expected or any(len(row["context_ids"]) != 4096 for row in rows):
        raise ValueError("dynamic packed row matrix mismatch")
    for row in rows:
        row["digit_positions"], row["prefix_positions"] = split_answer(tokenizer, row)
    return rows, manifest


def select_view(rows: list[dict], train_groups: int | None = None,
                dev_groups: int | None = None) -> tuple[list[dict], dict[str, list[str]]]:
    selected = {}
    for split, limit in (("train", train_groups), ("dev", dev_groups)):
        groups = sorted({row["group_id"] for row in rows if row["split"] == split})
        if limit is not None:
            if limit < 2 or limit > len(groups):
                raise ValueError(f"invalid {split} group limit")
            groups = groups[:limit]
        selected[split] = groups
    output = []
    for original in rows:
        split = original["split"]
        if original["group_id"] not in selected[split]:
            continue
        row = copy.deepcopy(original)
        groups = selected[split]
        other = groups[(groups.index(row["group_id"]) + 1) % len(groups)]
        twin = row["context_id"].rsplit(".", 1)[1]
        row["wrong_context_id"] = f"{other}.{twin}"
        output.append(row)
    contexts = defaultdict(list)
    for row in output:
        contexts[(row["split"], row["context_id"])].append(row)
    if any(len(values) != 4 for values in contexts.values()):
        raise ValueError("every dynamic context must supervise exactly four questions")
    return output, selected


def schedule(rows: list[dict], arm: str, rounds: int, seed: int = 503) -> list[list[str]]:
    if arm not in ARMS or rounds < 1:
        raise ValueError("invalid arm or rounds")
    by_context = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            by_context[row["context_id"]].append(row["id"])
    if any(len(ids) != 4 for ids in by_context.values()):
        raise ValueError("training contexts need four questions")
    rng = random.Random(seed)
    batches = []
    for _ in range(rounds):
        contexts = sorted(by_context)
        rng.shuffle(contexts)
        if arm == "joint_context":
            batches.extend([sorted(by_context[context]) for context in contexts])
        else:
            ids = [sample_id for context in contexts for sample_id in sorted(by_context[context])]
            rng.shuffle(ids)
            batches.extend([[sample_id] for sample_id in ids])
    return batches


def _bootstrap(values: list[float], draws: int = 5000, seed: int = 509) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(statistics.mean(values[rng.randrange(size)] for _ in range(size)) for _ in range(draws))
    return [estimates[int(.025 * draws)], estimates[min(draws - 1, int(.975 * draws))]]


def aggregate(records: list[dict]) -> dict:
    result = {}
    for split in ("train_probe", "dev"):
        subset = [row for row in records if row["evaluation_split"] == split]
        if not subset:
            raise ValueError(f"missing {split} evaluation rows")
        means = {
            policy: {metric: statistics.mean(row["policies"][policy][metric] for row in subset)
                     for metric in METRICS}
            for policy in ("correct", "empty", "wrong", "full_kv")
        }
        groups = {}
        for group in sorted({row["group_id"] for row in subset}):
            group_rows = [row for row in subset if row["group_id"] == group]
            groups[group] = {
                policy: {metric: statistics.mean(row["policies"][policy][metric] for row in group_rows)
                         for metric in METRICS}
                for policy in means
            }
        anchor = [row for row in subset if row["anchor_query"]]
        result[split] = {
            "rows": len(subset),
            "groups": groups,
            "means": means,
            "anchor_twin_digit_gain": statistics.mean(
                row["policies"]["twin"]["digit_nll"] - row["policies"]["correct"]["digit_nll"]
                for row in anchor
            ),
        }
    return result


def content_gate(dev: dict) -> dict:
    correct = dev["means"]["correct"]
    group_gains = {}
    checks = {}
    for policy in ("wrong", "empty"):
        values = [group[policy]["digit_nll"] - group["correct"]["digit_nll"]
                  for group in dev["groups"].values()]
        group_gains[policy] = {
            "mean_digit_nll_gain": statistics.mean(values),
            "source_group_bootstrap_95ci": _bootstrap(values),
            "positive_group_fraction": sum(value > 0 for value in values) / len(values),
        }
        checks[policy] = (
            correct["code_correct"] - dev["means"][policy]["code_correct"] >= .20
            and group_gains[policy]["mean_digit_nll_gain"] >= .10
            and group_gains[policy]["source_group_bootstrap_95ci"][0] > 0
        )
    checks["full_kv_readable"] = dev["means"]["full_kv"]["code_correct"] >= .80
    checks["correct_accuracy"] = correct["code_correct"] >= .60
    return {"checks": checks, "group_gains": group_gains, "passed": all(checks.values())}


def evaluate(model, tokenizer, rows: list[dict], manifest: dict, output: Path, exposure: int) -> dict:
    import torch
    from cbf_ttt.event_writer import write_memory
    from tasks.cbf_selective import measure

    train_groups = sorted({row["group_id"] for row in rows if row["split"] == "train"})[:8]
    selected = [copy.deepcopy(row) for row in rows
                if row["split"] == "dev" or (row["split"] == "train" and row["group_id"] in train_groups)]
    records = []
    write_profiles = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        all_contexts = {row["context_id"]: row["context_ids"] for row in rows}
        path = output / f"evaluation_{exposure}.jsonl"
        with path.open("x") as sink:
            for split in ("train", "dev"):
                split_rows = [row for row in selected if row["split"] == split]
                for group in sorted({row["group_id"] for row in split_rows}):
                    group_rows = [row for row in split_rows if row["group_id"] == group]
                    # A Qwen3-4B memory contains seven dense down-projection deltas.
                    # Keep only this twin pair plus its wrong-world donor pair live.
                    needed_contexts = {
                        context_id
                        for row in group_rows
                        for context_id in (row["context_id"], row["wrong_context_id"], row["twin_context_id"])
                    }
                    memories = {}
                    for context_id in sorted(needed_contexts):
                        memories[context_id], profile = measure(
                            lambda context_id=context_id: write_memory(model, all_contexts[context_id])
                        )
                        write_profiles[f"{group}:{context_id}"] = profile
                    for row in group_rows:
                        policies = {}
                        names = ("correct", "empty", "wrong", "full_kv") + (("twin",) if row["anchor_query"] else ())
                        for name in names:
                            key = (row["context_id"] if name == "correct" else
                                   row["wrong_context_id"] if name == "wrong" else
                                   row["twin_context_id"] if name == "twin" else None)
                            memory = memories[key] if key else {}
                            value, profile = measure(lambda: extra_score(
                                model, tokenizer, row, memory, manifest["eos_token_id"], name == "full_kv"
                            ))
                            policies[name] = {**value, **profile}
                        record = {
                            "id": row["id"], "group_id": row["group_id"], "split": row["split"],
                            "evaluation_split": "dev" if row["split"] == "dev" else "train_probe",
                            "anchor_query": row["anchor_query"], "policies": policies,
                        }
                        sink.write(json.dumps(record) + "\n")
                        sink.flush()
                        records.append(record)
                    del memories, memory
                    torch.cuda.empty_cache()
    result = aggregate(records)
    result.update({
        "exposure": exposure,
        "content_gate": content_gate(result["dev"]),
        "rows_sha256": sha(output / f"evaluation_{exposure}.jsonl"),
        "checkpoint_sha256": sha(output / f"writer_{exposure}.pt"),
        "write_profiles": write_profiles,
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
    all_rows, manifest = load_rows(args.data_root, tokenizer)
    rows, selected_groups = select_view(all_rows, args.train_groups, args.dev_groups)
    batches = schedule(rows, args.arm, args.rounds, args.order_seed)
    total_exposures = sum(len(batch) for batch in batches)
    if total_exposures % 2:
        raise ValueError("total question exposures must be even")
    checkpoints = (0, total_exposures // 2, total_exposures)
    if any(point % (4 if args.arm == "joint_context" else 1) for point in checkpoints):
        raise ValueError("checkpoint does not align with an optimizer update")

    output = Path(args.output)
    output.mkdir(parents=True)
    model, params, model_meta = setup(args)
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.)
    versions = {name: parameter._version for name, parameter in model.named_parameters() if not parameter.requires_grad}
    initial = {name: parameter.detach().cpu().clone() for name, parameter in params.items()}
    training_manifest = {
        **model_meta,
        "protocol": "dynamic_memory_writer_v1",
        "arm": args.arm,
        "rounds": args.rounds,
        "question_exposures": total_exposures,
        "optimizer_updates": len(batches),
        "checkpoints": list(checkpoints),
        "seed": 301,
        "order_seed": args.order_seed,
        "lr": args.lr,
        "clip": 1.0,
        "weight_decay": 0.0,
        "groups": selected_groups,
        "data_sha256": manifest["data_sha256"],
        "test_scored": False,
    }
    write_json(output / "training_manifest.json", training_manifest)
    by_id = {row["id"]: row for row in rows}

    def save(exposure: int) -> None:
        path = output / f"writer_{exposure}.pt"
        temp = Path(str(path) + ".tmp")
        torch.save({"exposure": exposure, "writer": {name: value.detach().cpu() for name, value in params.items()},
                    "manifest": training_manifest}, temp)
        temp.replace(path)

    save(0)
    evaluate(model, tokenizer, rows, manifest, output, 0)
    exposure = 0
    records = []
    with (output / "steps.jsonl").open("x") as sink:
        for update, batch in enumerate(batches, 1):
            selected = [by_id[sample_id] for sample_id in batch]

            def one_step() -> dict:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    context_ids = selected[0]["context_ids"]
                    if any(row["context_ids"] != context_ids for row in selected):
                        raise ValueError("joint batch contains more than one write context")
                    memory = write_memory(model, context_ids)
                    values = [regions(model, memory, row, manifest["eos_token_id"]) for row in selected]
                    loss = sum(value["ce"] for value in values) / len(values)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite dynamic writer loss")
                relative = max(float(value.detach().float().norm() /
                                     model.model.layers[layer].mlp.down_proj.weight.float().norm())
                               for layer, value in memory.items())
                if relative > 1:
                    raise RuntimeError("dynamic writer delta exceeds ||delta||/||W|| safety tripwire")
                loss.backward()
                gradients = {name: float(parameter.grad.norm()) if parameter.grad is not None else None
                             for name, parameter in params.items()}
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in params.values()):
                    raise RuntimeError("disconnected or nonfinite dynamic writer gradient")
                if any(value == 0 for name, value in gradients.items()
                       if ".ttt_conv." in name or update > 1):
                    raise RuntimeError("zero dynamic writer gradient")
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True))
                optimizer.step()
                if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
                    raise RuntimeError("frozen backbone received a gradient")
                return {
                    "update": update,
                    "exposure_start": exposure,
                    "exposure_end": exposure + len(selected),
                    "ids": batch,
                    "context_id": selected[0]["context_id"],
                    "loss": float(loss.detach()),
                    "question_losses": [float(value["ce"].detach()) for value in values],
                    "gradient_norms": gradients,
                    "unclipped_gradient_norm": gradient_norm,
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
                evaluate(model, tokenizer, rows, manifest, output, exposure)
            if update % 50 == 0:
                print(json.dumps({key: record[key] for key in ("update", "exposure_end", "loss", "seconds")}), flush=True)

    if exposure != total_exposures:
        raise RuntimeError("dynamic writer exposure accounting mismatch")
    if versions != {name: parameter._version for name, parameter in model.named_parameters()
                    if not parameter.requires_grad} or backbone_digest(model) != model_meta["backbone_sha256"]:
        raise RuntimeError("frozen backbone changed")
    changes = {name: float((parameter.detach().cpu() - initial[name]).norm()) for name, parameter in params.items()}
    if any(value <= 0 for value in changes.values()):
        raise RuntimeError("dynamic writer parameter did not change")
    optimizer_fp32 = all(value.dtype == torch.float32 for state in optimizer.state.values()
                         for value in state.values() if isinstance(value, torch.Tensor))
    if not optimizer_fp32:
        raise RuntimeError("dynamic writer optimizer state is not FP32")
    torch.save(optimizer.state_dict(), output / f"optimizer_{total_exposures}.pt")
    complete = {
        "arm": args.arm,
        "question_exposures": total_exposures,
        "optimizer_updates": len(batches),
        "backbone_unchanged": True,
        "optimizer_state_fp32": True,
        "writer_update_norms": changes,
        "mean_update_seconds": statistics.mean(record["seconds"] for record in records),
        "peak_allocated_gib": max(record["peak_allocated_gib"] for record in records),
        "peak_reserved_gib": max(record["peak_reserved_gib"] for record in records),
        "steps_sha256": sha(output / "steps.jsonl"),
        "test_scored": False,
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
        if complete["question_exposures"] != manifest["question_exposures"] or complete["test_scored"]:
            raise ValueError("dynamic writer arm is incomplete")
        checkpoints = {}
        for exposure in manifest["checkpoints"]:
            raw_path = output / f"evaluation_{exposure}.jsonl"
            raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
            stored = json.loads((output / f"evaluation_{exposure}.summary.json").read_text())
            recomputed = aggregate(raw)
            if any(stored[key] != value for key, value in recomputed.items()):
                raise ValueError("dynamic writer evaluation summary mismatch")
            if stored["rows_sha256"] != sha(raw_path):
                raise ValueError("dynamic writer evaluation hash mismatch")
            checkpoints[str(exposure)] = stored
        for split in ("train_probe", "dev"):
            for exposure in manifest["checkpoints"][1:]:
                if checkpoints[str(exposure)][split]["means"]["empty"] != checkpoints["0"][split]["means"]["empty"]:
                    raise ValueError("frozen empty-memory control drift")
        arms[arm] = {"manifest": manifest, "complete": complete, "checkpoints": checkpoints,
                     "final_gate": checkpoints[str(manifest["question_exposures"])]["content_gate"]}
    for key in ("seed", "data_sha256", "backbone_sha256"):
        if arms["sequential"]["manifest"][key] != arms["joint_context"]["manifest"][key]:
            raise ValueError("dynamic writer arms have different initial provenance")
    result = {
        "protocol": "dynamic_memory_writer_v1",
        "arms": arms,
        "test_scored": False,
        "stage_passed": any(arms[arm]["final_gate"]["passed"] for arm in ARMS),
        "terminal_status": "completed_dynamic_memory_writer",
    }
    if save:
        write_json(root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--data-root", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--model", default="/home/ctj/models/Qwen3-4B")
    run_parser.add_argument("--arm", choices=ARMS, required=True)
    run_parser.add_argument("--rounds", type=int, default=4)
    run_parser.add_argument("--train-groups", type=int)
    run_parser.add_argument("--dev-groups", type=int)
    run_parser.add_argument("--order-seed", type=int, default=503)
    run_parser.add_argument("--lr", type=float, default=1e-7)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    args = parser.parse_args()
    result = run(args) if args.command == "run" else summarize(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
