"""Joint CE versus digit-weighted twin-contrast training on dynamic worlds."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_dynamic_memory import aggregate, evaluate, load_rows, select_view
from tasks.train_cbf_event_content import regions
from tasks.train_cbf_event_writer import setup, write_json


ARMS = ("joint_ce", "joint_binding")


def schedule(rows: list[dict], rounds: int, seed: int = 607) -> list[list[str]]:
    by_context = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            by_context[row["context_id"]].append(row["id"])
    if rounds < 1 or any(len(ids) != 4 for ids in by_context.values()):
        raise ValueError("invalid rounds or context supervision")
    rng = random.Random(seed)
    batches = []
    for _ in range(rounds):
        contexts = sorted(by_context)
        rng.shuffle(contexts)
        batches.extend([sorted(by_context[context]) for context in contexts])
    return batches


def objective(values: list[dict], cross: dict | None, arm: str,
              margin: float = .2, contrast_weight: float = .5):
    import torch
    import torch.nn.functional as F

    ce = sum(value["ce"] for value in values) / len(values)
    balanced = sum(.75 * value["digits"] + .125 * value["prefix"] + .125 * value["eos"]
                   for value in values) / len(values)
    if arm == "joint_ce":
        return ce, balanced, torch.zeros_like(ce)
    if arm != "joint_binding" or cross is None:
        raise ValueError("binding objective requires a cross-memory anchor")
    anchor = [value for value in values if value["anchor"]]
    if len(anchor) != 1:
        raise ValueError("every context needs exactly one anchor question")
    contrast = F.softplus(margin + anchor[0]["digits"] - cross["digits"])
    return balanced + contrast_weight * contrast, balanced, contrast


def run(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from cbf_ttt.event_writer import backbone_digest, write_memory
    from tasks.cbf_selective import measure

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_rows, data_manifest = load_rows(args.data_root, tokenizer)
    rows, selected_groups = select_view(all_rows, args.train_groups, args.dev_groups)
    batches = schedule(rows, args.rounds, args.order_seed)
    total_exposures = sum(map(len, batches))
    checkpoints = (0, total_exposures // 2, total_exposures)
    if total_exposures % 8 or any(point % 4 for point in checkpoints):
        raise ValueError("binding exposure/checkpoint alignment mismatch")

    output = Path(args.output)
    output.mkdir(parents=True)
    model, params, model_meta = setup(args)
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.)
    versions = {name: parameter._version for name, parameter in model.named_parameters() if not parameter.requires_grad}
    initial = {name: parameter.detach().cpu().clone() for name, parameter in params.items()}
    manifest = {
        **model_meta,
        "protocol": "dynamic_binding_writer_v1",
        "arm": args.arm,
        "rounds": args.rounds,
        "question_exposures": total_exposures,
        "optimizer_updates": len(batches),
        "context_writes_per_update": 1 if args.arm == "joint_ce" else 2,
        "checkpoints": list(checkpoints),
        "seed": 301,
        "order_seed": args.order_seed,
        "lr": args.lr,
        "clip": 1.0,
        "weight_decay": 0.0,
        "groups": selected_groups,
        "data_sha256": data_manifest["data_sha256"],
        "loss": ({"ce": 1.0} if args.arm == "joint_ce" else
                 {"digits": .75, "prefix": .125, "eos": .125,
                  "anchor_twin_contrast": .5, "contrast_margin": .2}),
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
    evaluate(model, tokenizer, rows, data_manifest, output, 0)
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
                        raise ValueError("joint batch mixes write contexts")
                    memory = write_memory(model, by_context[context_id])
                    values = []
                    for row in selected:
                        value = regions(model, memory, row, data_manifest["eos_token_id"])
                        value["anchor"] = row["anchor_query"]
                        values.append(value)
                    twin_memory = None
                    cross = None
                    if args.arm == "joint_binding":
                        anchor_row = next(row for row in selected if row["anchor_query"])
                        twin_memory = write_memory(model, by_context[anchor_row["twin_context_id"]])
                        cross = regions(model, twin_memory, anchor_row, data_manifest["eos_token_id"])
                    loss, balanced, contrast = objective(values, cross, args.arm)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite dynamic binding loss")
                memories = [memory] + ([twin_memory] if twin_memory is not None else [])
                relative = max(float(delta.detach().float().norm() /
                                     model.model.layers[layer].mlp.down_proj.weight.float().norm())
                               for item in memories for layer, delta in item.items())
                if relative > 1:
                    raise RuntimeError("dynamic binding delta exceeds ||delta||/||W|| safety tripwire")
                loss.backward()
                gradients = {name: float(parameter.grad.norm()) if parameter.grad is not None else None
                             for name, parameter in params.items()}
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in params.values()):
                    raise RuntimeError("disconnected or nonfinite dynamic binding gradient")
                if any(value == 0 for name, value in gradients.items()
                       if ".ttt_conv." in name or update > 1):
                    raise RuntimeError("zero dynamic binding gradient")
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
                    "balanced_loss": float(balanced.detach()),
                    "contrast_loss": float(contrast.detach()),
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
                evaluate(model, tokenizer, rows, data_manifest, output, exposure)
            if update % 50 == 0:
                print(json.dumps({key: record[key] for key in
                                  ("update", "exposure_end", "loss", "contrast_loss", "seconds")}), flush=True)

    if exposure != total_exposures:
        raise RuntimeError("dynamic binding exposure accounting mismatch")
    if versions != {name: parameter._version for name, parameter in model.named_parameters()
                    if not parameter.requires_grad} or backbone_digest(model) != model_meta["backbone_sha256"]:
        raise RuntimeError("frozen backbone changed")
    changes = {name: float((parameter.detach().cpu() - initial[name]).norm()) for name, parameter in params.items()}
    if any(value <= 0 for value in changes.values()):
        raise RuntimeError("dynamic binding parameter did not change")
    if not all(value.dtype == torch.float32 for state in optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("dynamic binding optimizer state is not FP32")
    torch.save(optimizer.state_dict(), output / f"optimizer_{total_exposures}.pt")
    complete = {
        "arm": args.arm,
        "question_exposures": total_exposures,
        "optimizer_updates": len(batches),
        "backbone_unchanged": True,
        "optimizer_state_fp32": True,
        "writer_update_norms": changes,
        "first50_loss": statistics.mean(record["loss"] for record in records[:50]),
        "last50_loss": statistics.mean(record["loss"] for record in records[-50:]),
        "mean_update_seconds": statistics.mean(record["seconds"] for record in records),
        "peak_allocated_gib": max(record["peak_allocated_gib"] for record in records),
        "peak_reserved_gib": max(record["peak_reserved_gib"] for record in records),
        "max_relative_delta": max(record["max_relative_delta"] for record in records),
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
        if manifest["arm"] != arm or complete["question_exposures"] != manifest["question_exposures"]:
            raise ValueError("dynamic binding arm incomplete")
        checkpoints = {}
        for exposure in manifest["checkpoints"]:
            raw_path = output / f"evaluation_{exposure}.jsonl"
            raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
            stored = json.loads((output / f"evaluation_{exposure}.summary.json").read_text())
            recomputed = aggregate(raw)
            if any(stored[key] != value for key, value in recomputed.items()) or stored["rows_sha256"] != sha(raw_path):
                raise ValueError("dynamic binding evaluation mismatch")
            checkpoints[str(exposure)] = stored
        for split in ("train_probe", "dev"):
            for exposure in manifest["checkpoints"][1:]:
                if checkpoints[str(exposure)][split]["means"]["empty"] != checkpoints["0"][split]["means"]["empty"]:
                    raise ValueError("dynamic binding empty control drift")
        arms[arm] = {"manifest": manifest, "complete": complete, "checkpoints": checkpoints,
                     "final_gate": checkpoints[str(manifest["question_exposures"])]["content_gate"]}
    for key in ("seed", "data_sha256", "backbone_sha256", "question_exposures"):
        if arms[ARMS[0]]["manifest"][key] != arms[ARMS[1]]["manifest"][key]:
            raise ValueError("dynamic binding arms are not matched")
    final_exposure = str(arms[ARMS[0]]["manifest"]["question_exposures"])
    comparison = {}
    for split in ("train_probe", "dev"):
        comparison[split] = {
            metric: (arms["joint_binding"]["checkpoints"][final_exposure][split]["means"]["correct"][metric]
                     - arms["joint_ce"]["checkpoints"][final_exposure][split]["means"]["correct"][metric])
            for metric in ("code_correct", "digit_nll", "answer_nll")
        }
    result = {
        "protocol": "dynamic_binding_writer_v1",
        "arms": arms,
        "binding_minus_ce": comparison,
        "stage_passed": any(arms[arm]["final_gate"]["passed"] for arm in ARMS),
        "test_scored": False,
        "terminal_status": "completed_dynamic_binding_writer",
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
    run_parser.add_argument("--order-seed", type=int, default=607)
    run_parser.add_argument("--lr", type=float, default=1e-7)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    args = parser.parse_args()
    result = run(args) if args.command == "run" else summarize(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
