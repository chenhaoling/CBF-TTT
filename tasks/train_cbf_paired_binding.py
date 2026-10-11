"""Train a writer with two counterfactual assignments in every update."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import setup, write_json
from tasks.train_cbf_episodic_binding import (
    _prepare, episodic_gate, evaluate,
)
from tasks.train_cbf_listwise_binding import listwise_query


ARMS = ("paired_listwise", "paired_swap")
PAIRINGS = ((0, 2), (1, 3))


def paired_schedule(rows: list[dict], rounds: int = 2,
                    seed: int = 1201) -> list[tuple[str, int, int]]:
    if rounds != 2:
        raise ValueError("paired protocol requires exactly two assignment-pair rounds")
    contexts = sorted({row["context_id"] for row in rows if row["split"] == "train"})
    if not contexts:
        raise ValueError("empty paired training split")
    rng = random.Random(seed)
    schedules = []
    pairing_order = list(PAIRINGS)
    rng.shuffle(pairing_order)
    for left, right in pairing_order:
        order = list(contexts)
        rng.shuffle(order)
        for context in order:
            if rng.randrange(2):
                schedules.append((context, right, left))
            else:
                schedules.append((context, left, right))
    return schedules


def paired_objective(left_items: list, right_items: list, left_rows: list[dict],
                     right_rows: list[dict], candidates: list[str], arm: str,
                     margin: float = .2, swap_weight: float = .5):
    import torch
    import torch.nn.functional as F

    if len(left_items) != 4 or len(right_items) != 4:
        raise ValueError("paired objective requires four queries per assignment")
    retrieval = (sum(item[0] for item in left_items) +
                 sum(item[0] for item in right_items)) / 8
    terms = []
    for left, right, left_row, right_row in zip(left_items, right_items,
                                                left_rows, right_rows):
        left_index = candidates.index(left_row["answer"])
        right_index = candidates.index(right_row["answer"])
        if left_index == right_index or left_row["query_ids"] != right_row["query_ids"]:
            raise ValueError("paired assignments must change the target but preserve the query")
        terms.extend((
            F.softplus(margin + left[1][left_index] - right[1][left_index]),
            F.softplus(margin + right[1][right_index] - left[1][right_index]),
        ))
    swap = sum(terms) / len(terms)
    if arm == "paired_listwise":
        loss = retrieval + 0. * swap
    elif arm == "paired_swap":
        loss = retrieval + swap_weight * swap
    else:
        raise ValueError("unknown paired arm")
    accuracy = statistics.fmean([item[2] for item in left_items + right_items])
    return loss, retrieval, swap, accuracy


def smoke(args) -> dict:
    import torch
    from cbf_ttt.event_writer import write_memory
    from tasks.cbf_selective import measure

    tokenizer, rows, _, _, vocab, episodes, candidates = _prepare(args)
    del tokenizer
    context, left_shift, right_shift = paired_schedule(rows, 2, args.order_seed)[0]
    left, right = episodes[context][left_shift], episodes[context][right_shift]
    model, params, model_meta = setup(args)
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.)

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            memories = [write_memory(model, item["context_ids"],
                                     max_relative_norm=args.memory_cap)
                        for item in (left, right)]
            left_items = [listwise_query(model, memories[0], row, candidates[context], vocab)
                          for row in left["rows"]]
            right_items = [listwise_query(model, memories[1], row, candidates[context], vocab)
                           for row in right["rows"]]
            loss, retrieval, swap, accuracy = paired_objective(
                left_items, right_items, left["rows"], right["rows"],
                candidates[context], args.arm)
        loss.backward()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
               for parameter in params.values()):
            raise RuntimeError("invalid paired smoke gradient")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.,
                                                              error_if_nonfinite=True))
        optimizer.step()
        relative = max(float(delta.detach().float().norm() /
                             model.model.layers[layer].mlp.down_proj.weight.float().norm())
                       for memory in memories for layer, delta in memory.items())
        return {
            "arm": args.arm, "shifts": [left_shift, right_shift],
            "loss": float(loss.detach()), "retrieval_loss": float(retrieval.detach()),
            "swap_loss": float(swap.detach()), "listwise_accuracy": accuracy,
            "unclipped_gradient_norm": gradient_norm, "max_relative_delta": relative,
            "backbone_sha256": model_meta["backbone_sha256"],
            "memory_cap": args.memory_cap, "test_scored": False,
        }

    result, profile = measure(one_step)
    result.update(profile)
    write_json(args.output, result)
    return result


def run(args) -> dict:
    import torch
    from cbf_ttt.event_writer import backbone_digest, write_memory
    from tasks.cbf_selective import measure

    tokenizer, rows, groups, data_manifest, vocab, episodes, candidates = _prepare(args)
    batches = paired_schedule(rows, args.rounds, args.order_seed)
    total_exposures = 8 * len(batches)
    checkpoints = (0, total_exposures)
    output = Path(args.output)
    output.mkdir(parents=True)
    model, params, model_meta = setup(args)
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.)
    versions = {name: parameter._version for name, parameter in model.named_parameters()
                if not parameter.requires_grad}
    initial = {name: parameter.detach().cpu().clone() for name, parameter in params.items()}
    manifest = {
        **model_meta, "protocol": "paired_episodic_binding_v1", "arm": args.arm,
        "rounds": args.rounds, "assignment_pairings": [list(item) for item in PAIRINGS],
        "question_exposures": total_exposures, "optimizer_updates": len(batches),
        "context_writes_per_update": 2, "query_reads_per_update": 8,
        "listwise_candidates": 4, "checkpoints": list(checkpoints),
        "evaluation_shifts": [0, 2], "seed": 301, "order_seed": args.order_seed,
        "lr": args.lr, "clip": 1.0, "weight_decay": 0.0, "groups": groups,
        "data_sha256": data_manifest["data_sha256"], "train_groups": args.train_groups,
        "dev_groups": args.dev_groups, "memory_cap": args.memory_cap,
        "lm_weight": 0.0, "swap_weight": 0.0 if args.arm == "paired_listwise" else .5,
        "swap_margin": .2, "test_scored": False,
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
        for update, (context, left_shift, right_shift) in enumerate(batches, 1):
            left, right = episodes[context][left_shift], episodes[context][right_shift]

            def one_step():
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    memories = [write_memory(model, item["context_ids"],
                                             max_relative_norm=args.memory_cap)
                                for item in (left, right)]
                    left_items = [listwise_query(model, memories[0], row, candidates[context], vocab)
                                  for row in left["rows"]]
                    right_items = [listwise_query(model, memories[1], row, candidates[context], vocab)
                                   for row in right["rows"]]
                    loss, retrieval, swap, accuracy = paired_objective(
                        left_items, right_items, left["rows"], right["rows"],
                        candidates[context], args.arm)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite paired loss")
                relative = max(float(delta.detach().float().norm() /
                                     model.model.layers[layer].mlp.down_proj.weight.float().norm())
                               for memory in memories for layer, delta in memory.items())
                if relative > 1:
                    raise RuntimeError("paired delta exceeds safety tripwire")
                loss.backward()
                gradients = {name: float(parameter.grad.norm()) if parameter.grad is not None else None
                             for name, parameter in params.items()}
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in params.values()):
                    raise RuntimeError("disconnected or nonfinite paired gradient")
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.,
                                                                      error_if_nonfinite=True))
                optimizer.step()
                return {
                    "update": update, "exposure_start": exposure,
                    "exposure_end": exposure + 8, "context_id": context,
                    "assignment_shifts": [left_shift, right_shift],
                    "left_targets": left["targets"], "right_targets": right["targets"],
                    "loss": float(loss.detach()), "retrieval_loss": float(retrieval.detach()),
                    "swap_loss": float(swap.detach()), "listwise_accuracy": accuracy,
                    "gradient_norms": gradients, "unclipped_gradient_norm": gradient_norm,
                    "max_relative_delta": relative,
                }

            record, profile = measure(one_step)
            exposure += 8
            record.update(profile)
            records.append(record)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if update % 16 == 0:
                print(json.dumps({key: record[key] for key in
                                  ("update", "exposure_end", "assignment_shifts", "loss",
                                   "retrieval_loss", "swap_loss", "listwise_accuracy", "seconds")}),
                      flush=True)
    save(exposure)
    evaluate(model, tokenizer, rows, data_manifest, output, exposure, vocab, episodes,
             candidates, args.memory_cap)
    if exposure != total_exposures:
        raise RuntimeError("paired exposure accounting mismatch")
    if versions != {name: parameter._version for name, parameter in model.named_parameters()
                    if not parameter.requires_grad} or backbone_digest(model) != model_meta["backbone_sha256"]:
        raise RuntimeError("frozen backbone changed")
    changes = {name: float((parameter.detach().cpu() - initial[name]).norm())
               for name, parameter in params.items()}
    if any(value <= 0 for value in changes.values()):
        raise RuntimeError("paired writer parameter did not change")
    if not all(value.dtype == torch.float32 for state in optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("paired optimizer state is not FP32")
    torch.save(optimizer.state_dict(), output / f"optimizer_{total_exposures}.pt")
    complete = {
        "arm": args.arm, "question_exposures": total_exposures,
        "optimizer_updates": len(batches), "backbone_unchanged": True,
        "optimizer_state_fp32": True, "writer_update_norms": changes,
        "first16_retrieval_loss": statistics.fmean(row["retrieval_loss"] for row in records[:16]),
        "last16_retrieval_loss": statistics.fmean(row["retrieval_loss"] for row in records[-16:]),
        "first16_swap_loss": statistics.fmean(row["swap_loss"] for row in records[:16]),
        "last16_swap_loss": statistics.fmean(row["swap_loss"] for row in records[-16:]),
        "first16_listwise_accuracy": statistics.fmean(row["listwise_accuracy"] for row in records[:16]),
        "last16_listwise_accuracy": statistics.fmean(row["listwise_accuracy"] for row in records[-16:]),
        "mean_update_seconds": statistics.fmean(row["seconds"] for row in records),
        "peak_allocated_gib": max(row["peak_allocated_gib"] for row in records),
        "peak_reserved_gib": max(row["peak_reserved_gib"] for row in records),
        "max_relative_delta": max(row["max_relative_delta"] for row in records),
        "steps_sha256": sha(output / "steps.jsonl"), "test_scored": False,
    }
    write_json(output / "complete.json", complete)
    return complete


def summarize(root: str | Path, save: bool = True) -> dict:
    # Reuse the evaluated checkpoint verifier after temporarily mapping arm names.
    root = Path(root)
    arms = {}
    from tasks.train_cbf_episodic_binding import _ranking_aggregate
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
                raise ValueError("paired evaluation mismatch")
            checkpoints[str(exposure)] = stored
        final = checkpoints[str(manifest["question_exposures"])]
        arms[arm] = {"manifest": manifest, "complete": complete,
                     "checkpoints": checkpoints, "final_gate": final["episodic_gate"]}
    for key in ("seed", "order_seed", "data_sha256", "backbone_sha256", "question_exposures"):
        if arms[ARMS[0]]["manifest"][key] != arms[ARMS[1]]["manifest"][key]:
            raise ValueError("paired arms are not matched")
    result = {
        "protocol": "paired_episodic_binding_v1", "arms": arms,
        "primary_arm": "paired_swap", "stage_passed": arms["paired_swap"]["final_gate"]["passed"],
        "test_scored": False, "terminal_status": "completed_paired_episodic_binding",
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
        item.add_argument("--order-seed", type=int, default=1201)
        item.add_argument("--lr", type=float, default=1e-7)
        item.add_argument("--memory-cap", type=float, default=.3)
        if name == "run":
            item.add_argument("--rounds", type=int, default=2)
    summary = sub.add_parser("summarize")
    summary.add_argument("--root", required=True)
    args = parser.parse_args()
    result = smoke(args) if args.command == "smoke" else run(args) if args.command == "run" else summarize(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
