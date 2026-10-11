"""Train dynamic writers with entity-conditioned listwise value retrieval."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from tasks.evaluate_cbf_candidate_ranking import answer_vocabulary
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import (
    aggregate, evaluate, permuted_contexts, ranking_gate, schedule,
)
from tasks.train_cbf_dynamic_memory import load_rows, select_view
from tasks.train_cbf_event_content import regions
from tasks.train_cbf_event_writer import setup, write_json


ARMS = ("retrieval_only", "retrieval_plus_lm")


def context_candidates(rows: list[dict]) -> dict[str, list[str]]:
    result = defaultdict(set)
    for row in rows:
        result[row["context_id"]].add(row["answer"])
    output = {context: sorted(values) for context, values in result.items()}
    if any(len(values) != 4 for values in output.values()):
        raise ValueError("listwise training requires four unique queried values per context")
    return output


def listwise_query(model, memory: dict, row: dict, candidates: list[str], vocab: dict[str, dict],
                   temperature: float = 1.0):
    import torch
    import torch.nn.functional as F
    from cbf_ttt.event_writer import cache_for

    if temperature <= 0 or row["answer"] not in candidates or len(candidates) != 4:
        raise ValueError("invalid listwise candidates or temperature")
    lengths = {len(vocab[value]["ids"]) for value in candidates}
    if lengths != {7}:
        raise ValueError("listwise answers must use the audited seven-token layout")
    tokens = torch.tensor(
        [row["query_ids"] + vocab[value]["ids"][:-1] for value in candidates],
        device=next(model.parameters()).device,
    )
    cache = cache_for(model, memory)
    hidden = model.model(input_ids=tokens, past_key_values=cache, use_cache=True).last_hidden_state
    answer_length = next(iter(lengths))
    logits = model.lm_head(hidden[:, len(row["query_ids"]) - 1:
                                  len(row["query_ids"]) - 1 + answer_length]).float()
    targets = torch.tensor([vocab[value]["ids"] for value in candidates], device=logits.device)
    token_losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                   reduction="none").reshape(len(candidates), answer_length)
    digit_nll = torch.stack([
        token_losses[index, vocab[value]["digits"]].mean()
        for index, value in enumerate(candidates)
    ])
    target = torch.tensor([candidates.index(row["answer"])], device=logits.device)
    retrieval_logits = (-digit_nll / temperature).unsqueeze(0)
    loss = F.cross_entropy(retrieval_logits, target)
    return loss, digit_nll, int(retrieval_logits.detach().argmax(dim=-1).item() == target.item())


def objective(retrieval_losses: list, balanced, arm: str):
    retrieval = sum(retrieval_losses) / len(retrieval_losses)
    if arm == "retrieval_only":
        lm_weight = 0.0
    elif arm == "retrieval_plus_lm":
        lm_weight = 0.25
    else:
        raise ValueError("unknown listwise arm")
    return retrieval + lm_weight * balanced, retrieval, lm_weight


def _prepare(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_rows, data_manifest = load_rows(args.data_root, tokenizer)
    rows, groups = select_view(all_rows, args.train_groups, args.dev_groups)
    permutation, _ = permuted_contexts(rows, tokenizer)
    vocab = answer_vocabulary(rows, tokenizer)
    candidates = context_candidates(rows)
    return tokenizer, rows, groups, data_manifest, permutation, vocab, candidates


def smoke(args) -> dict:
    import torch
    from cbf_ttt.event_writer import write_memory
    from tasks.cbf_selective import measure

    tokenizer, rows, _, data_manifest, _, vocab, candidates = _prepare(args)
    del tokenizer
    batch = schedule(rows, 1, args.order_seed)[0]
    by_id = {row["id"]: row for row in rows}
    selected = [by_id[sample_id] for sample_id in batch]
    model, params, model_meta = setup(args)
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.)

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context_id = selected[0]["context_id"]
            memory = write_memory(model, selected[0]["context_ids"])
            values = [regions(model, memory, row, data_manifest["eos_token_id"]) for row in selected]
            balanced = sum(.75 * value["digits"] + .125 * value["prefix"] + .125 * value["eos"]
                           for value in values) / len(values)
            retrieval_items = [listwise_query(model, memory, row, candidates[context_id], vocab)
                               for row in selected]
            loss, retrieval, lm_weight = objective(
                [item[0] for item in retrieval_items], balanced, args.arm)
        relative = max(float(delta.detach().float().norm() /
                             model.model.layers[layer].mlp.down_proj.weight.float().norm())
                       for layer, delta in memory.items())
        if not torch.isfinite(loss) or relative > 1:
            raise RuntimeError("unsafe listwise smoke")
        loss.backward()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
               for parameter in params.values()):
            raise RuntimeError("invalid listwise smoke gradient")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.,
                                                              error_if_nonfinite=True))
        optimizer.step()
        return {
            "arm": args.arm, "loss": float(loss.detach()),
            "retrieval_loss": float(retrieval.detach()), "balanced_loss": float(balanced.detach()),
            "lm_weight": lm_weight,
            "listwise_accuracy": statistics.fmean(item[2] for item in retrieval_items),
            "unclipped_gradient_norm": gradient_norm, "max_relative_delta": relative,
            "backbone_sha256": model_meta["backbone_sha256"], "test_scored": False,
        }

    result, profile = measure(one_step)
    result.update(profile)
    write_json(args.output, result)
    return result


def run(args) -> dict:
    import torch
    from cbf_ttt.event_writer import backbone_digest, write_memory
    from tasks.cbf_selective import measure

    tokenizer, rows, selected_groups, data_manifest, permutation, vocab, candidates = _prepare(args)
    batches = schedule(rows, args.rounds, args.order_seed)
    total_exposures = sum(map(len, batches))
    checkpoints = (0, total_exposures // 2, total_exposures)
    if total_exposures % 8 or any(point % 4 for point in checkpoints):
        raise ValueError("listwise checkpoint alignment mismatch")
    output = Path(args.output)
    output.mkdir(parents=True)
    model, params, model_meta = setup(args)
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.)
    versions = {name: parameter._version for name, parameter in model.named_parameters()
                if not parameter.requires_grad}
    initial = {name: parameter.detach().cpu().clone() for name, parameter in params.items()}
    manifest = {
        **model_meta, "protocol": "listwise_binding_v1", "arm": args.arm,
        "rounds": args.rounds, "question_exposures": total_exposures,
        "optimizer_updates": len(batches), "context_writes_per_update": 1,
        "query_reads_per_update": 8, "listwise_candidates": 4,
        "checkpoints": list(checkpoints), "seed": 301, "order_seed": args.order_seed,
        "lr": args.lr, "clip": 1.0, "weight_decay": 0.0, "groups": selected_groups,
        "data_sha256": data_manifest["data_sha256"], "train_groups": args.train_groups,
        "dev_groups": args.dev_groups, "temperature": 1.0,
        "lm_weight": 0.0 if args.arm == "retrieval_only" else 0.25,
        "test_scored": False,
    }
    write_json(output / "training_manifest.json", manifest)
    by_id = {row["id"]: row for row in rows}

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

            def one_step():
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    context_id = selected[0]["context_id"]
                    if any(row["context_id"] != context_id for row in selected):
                        raise ValueError("listwise batch mixes contexts")
                    memory = write_memory(model, selected[0]["context_ids"])
                    values = [regions(model, memory, row, data_manifest["eos_token_id"])
                              for row in selected]
                    balanced = sum(.75 * value["digits"] + .125 * value["prefix"] + .125 * value["eos"]
                                   for value in values) / len(values)
                    items = [listwise_query(model, memory, row, candidates[context_id], vocab)
                             for row in selected]
                    loss, retrieval, lm_weight = objective([item[0] for item in items], balanced, args.arm)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite listwise loss")
                relative = max(float(delta.detach().float().norm() /
                                     model.model.layers[layer].mlp.down_proj.weight.float().norm())
                               for layer, delta in memory.items())
                if relative > 1:
                    raise RuntimeError("listwise delta exceeds safety tripwire")
                loss.backward()
                gradients = {name: float(parameter.grad.norm()) if parameter.grad is not None else None
                             for name, parameter in params.items()}
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in params.values()):
                    raise RuntimeError("disconnected or nonfinite listwise gradient")
                if any(value == 0 for name, value in gradients.items()
                       if ".ttt_conv." in name or update > 1):
                    raise RuntimeError("zero listwise gradient")
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.,
                                                                      error_if_nonfinite=True))
                optimizer.step()
                return {
                    "update": update, "exposure_start": exposure,
                    "exposure_end": exposure + len(selected), "ids": batch,
                    "context_id": selected[0]["context_id"], "loss": float(loss.detach()),
                    "retrieval_loss": float(retrieval.detach()),
                    "balanced_loss": float(balanced.detach()), "lm_weight": lm_weight,
                    "listwise_accuracy": statistics.fmean(item[2] for item in items),
                    "candidate_values": candidates[context_id],
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
                                  ("update", "exposure_end", "loss", "retrieval_loss",
                                   "listwise_accuracy", "seconds")}), flush=True)
    if exposure != total_exposures:
        raise RuntimeError("listwise exposure accounting mismatch")
    if versions != {name: parameter._version for name, parameter in model.named_parameters()
                    if not parameter.requires_grad} or backbone_digest(model) != model_meta["backbone_sha256"]:
        raise RuntimeError("frozen backbone changed")
    changes = {name: float((parameter.detach().cpu() - initial[name]).norm())
               for name, parameter in params.items()}
    if any(value <= 0 for value in changes.values()):
        raise RuntimeError("listwise parameter did not change")
    if not all(value.dtype == torch.float32 for state in optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("listwise optimizer state is not FP32")
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


def optimization_gate(complete: dict, final_dev: dict, final_train: dict) -> dict:
    train_target = final_train["global16"]
    checks = {
        "last32_listwise_accuracy": complete["last32_listwise_accuracy"] > .50,
        "retrieval_below_chance": complete["last32_retrieval_loss"] < math.log(4),
        "train_permutation_top1_gap": (
            train_target["policies"]["correct"]["top1"]
            - train_target["policies"]["permuted"]["top1"] >= .10),
        "development_ranking_gate": ranking_gate(final_dev)["passed"],
    }
    return {"checks": checks, "passed": all(checks.values())}


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
            if stored["rows_sha256"] != sha(raw) or aggregate(records) != {
                    key: stored[key] for key in ("train_probe", "dev")}:
                raise ValueError("listwise evaluation mismatch")
            checkpoints[str(exposure)] = stored
        final = checkpoints[str(manifest["question_exposures"])]
        arms[arm] = {
            "manifest": manifest, "complete": complete, "checkpoints": checkpoints,
            "final_gate": optimization_gate(complete, final["dev"], final["train_probe"]),
        }
    for key in ("seed", "order_seed", "data_sha256", "backbone_sha256", "question_exposures"):
        if arms[ARMS[0]]["manifest"][key] != arms[ARMS[1]]["manifest"][key]:
            raise ValueError("listwise arms are not matched")
    result = {
        "protocol": "listwise_binding_v1", "arms": arms,
        "primary_arm": "retrieval_plus_lm",
        "stage_passed": arms["retrieval_plus_lm"]["final_gate"]["passed"],
        "test_scored": False, "terminal_status": "completed_listwise_binding",
    }
    if save:
        write_json(root / "summary.json", result)
    return result


def main() -> None:
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
        item.add_argument("--order-seed", type=int, default=919)
        item.add_argument("--lr", type=float, default=1e-7)
        if name == "run":
            item.add_argument("--rounds", type=int, default=4)
    summary = sub.add_parser("summarize")
    summary.add_argument("--root", required=True)
    args = parser.parse_args()
    result = smoke(args) if args.command == "smoke" else run(args) if args.command == "run" else summarize(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

