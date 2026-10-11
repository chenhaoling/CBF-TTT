"""Train an explicit candidate retrieval head with episodic writer memory."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from tasks.evaluate_cbf_candidate_ranking import (
    RANK_METRICS, build_candidate_sets, ranking,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import setup, write_json
from tasks.train_cbf_episodic_binding import _prepare
from tasks.train_cbf_paired_binding import paired_schedule


ARMS = ("head_only", "joint_head_writer")
POLICIES = ("correct", "counterfactual", "wrong", "empty")
SETS = ("context4", "global16")
EVAL_SHIFTS = (0, 2)


def make_head(hidden_size: int, projection_size: int = 256, temperature: float = .1):
    import torch
    import torch.nn.functional as F

    class RetrievalHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query = torch.nn.Linear(hidden_size, projection_size, bias=False)
            self.value = torch.nn.Linear(hidden_size, projection_size, bias=False)
            self.temperature = temperature

        def forward(self, query, values):
            query = F.normalize(self.query(query.float()), dim=-1)
            values = F.normalize(self.value(values.float()), dim=-1)
            return query @ values.transpose(-1, -2) / self.temperature

    return RetrievalHead()


def value_vectors(model, candidates: list[str], vocab: dict[str, dict]):
    import torch

    embedding = model.model.embed_tokens.weight.detach()
    vectors = []
    for value in candidates:
        ids = torch.tensor(vocab[value]["ids"], device=embedding.device)
        positions = torch.tensor(vocab[value]["digits"], device=embedding.device)
        vectors.append(embedding[ids][positions].float().mean(dim=0))
    return torch.stack(vectors)


def head_logits(model, head, memory: dict, row: dict, candidates: list[str],
                vocab: dict[str, dict], full_kv: bool = False):
    import torch
    from cbf_ttt.event_writer import cache_for, forward

    cache = cache_for(model, memory)
    if full_kv:
        forward(model, row["context_ids"], cache)
    hidden = forward(model, row["query_ids"], cache)[:, -1].float()
    values = value_vectors(model, candidates, vocab)
    with torch.autocast("cuda", enabled=False):
        return head(hidden, values)


def retrieval_item(model, head, memory: dict, row: dict, candidates: list[str],
                   vocab: dict[str, dict]):
    import torch
    import torch.nn.functional as F

    logits = head_logits(model, head, memory, row, candidates, vocab)
    target = torch.tensor([candidates.index(row["answer"])], device=logits.device)
    loss = F.cross_entropy(logits, target)
    correct = int(logits.detach().argmax(dim=-1).item() == target.item())
    return loss, correct


def head_schedule(rows: list[dict], cycles: int, seed: int = 1303):
    if cycles < 1:
        raise ValueError("head cycles must be positive")
    output = []
    for cycle in range(cycles):
        output.extend(paired_schedule(rows, 2, seed + cycle))
    return output


def _score(model, head, memory, row, candidates, vocab, full_kv=False):
    import torch
    import torch.nn.functional as F

    logits = head_logits(model, head, memory, row, candidates, vocab, full_kv).squeeze(0)
    nll = -F.log_softmax(logits.float(), dim=-1)
    return [{"answer": value, "digit_nll": float(nll[index]),
             "answer_nll": float(nll[index])}
            for index, value in enumerate(candidates)]


def _aggregate(records: list[dict]) -> dict:
    result = {}
    for split in ("train_probe", "dev"):
        subset = [row for row in records if row["evaluation_split"] == split]
        sets = {}
        for set_name in SETS:
            policies = {
                policy: {metric: statistics.fmean(
                    row["policies"][policy][set_name][metric] for row in subset)
                    for metric in RANK_METRICS}
                for policy in POLICIES
            }
            groups = sorted({row["group_id"] for row in subset})
            comparisons = {}
            for index, policy in enumerate(("counterfactual", "wrong", "empty")):
                nll, top1 = [], []
                for group in groups:
                    group_rows = [row for row in subset if row["group_id"] == group]
                    nll.append(statistics.fmean(
                        row["policies"][policy][set_name]["true_digit_nll"]
                        - row["policies"]["correct"][set_name]["true_digit_nll"]
                        for row in group_rows))
                    top1.append(statistics.fmean(
                        row["policies"]["correct"][set_name]["top1"]
                        - row["policies"][policy][set_name]["top1"]
                        for row in group_rows))
                comparisons[policy] = {
                    "mean_true_nll_gain": statistics.fmean(nll),
                    "nll_gain_bootstrap_95ci": _bootstrap(nll, 1321 + index),
                    "top1_advantage": statistics.fmean(top1),
                    "top1_advantage_bootstrap_95ci": _bootstrap(top1, 1331 + index),
                }
            sets[set_name] = {"policies": policies, "correct_comparisons": comparisons}
        full_kv = {metric: statistics.fmean(row["full_kv"][metric] for row in subset)
                   for metric in RANK_METRICS}
        result[split] = {"rows": len(subset),
                         "groups": len({row["group_id"] for row in subset}),
                         "sets": sets, "full_kv": full_kv}
    return result


def gate(dev: dict) -> dict:
    local = dev["sets"]["context4"]
    global_set = dev["sets"]["global16"]
    comparison = local["correct_comparisons"]["counterfactual"]
    correct = local["policies"]["correct"]["top1"]
    checks = {
        "context4_correct_top1": correct >= .60,
        "context4_counterfactual_gap": correct - local["policies"]["counterfactual"]["top1"] >= .25,
        "context4_counterfactual_nll_confidence": comparison["nll_gain_bootstrap_95ci"][0] > 0,
        "global16_correct_top1": global_set["policies"]["correct"]["top1"] >= .25,
        "full_kv_global16_top1": dev["full_kv"]["top1"] >= .50,
    }
    return {"checks": checks, "passed": all(checks.values())}


def evaluate(model, head, rows, vocab, episodes, candidates, output: Path,
             exposure: int, memory_cap: float):
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
                            scores, profile = measure(lambda memory=memory:
                                _score(model, head, memory, row, global_candidates, vocab))
                            profiles.append(profile)
                            policies[policy] = {
                                "context4": ranking(scores, candidates[context], row["answer"]),
                                "global16": ranking(scores, global_candidates, row["answer"]),
                            }
                        full_scores, profile = measure(lambda:
                            _score(model, head, {}, row, global_candidates, vocab, True))
                        profiles.append(profile)
                        record = {
                            "id": row["id"], "group_id": row["group_id"], "split": row["split"],
                            "evaluation_split": "dev" if row["split"] == "dev" else "train_probe",
                            "context_id": context, "assignment_shift": shift, "truth": row["answer"],
                            "policies": policies,
                            "full_kv": ranking(full_scores, global_candidates, row["answer"]),
                        }
                        sink.write(json.dumps(record) + "\n")
                        sink.flush()
                        records.append(record)
                del memories
                torch.cuda.empty_cache()
    result = _aggregate(records)
    result.update({"exposure": exposure, "gate": gate(result["dev"]),
                   "rows_sha256": sha(path), "test_scored": False,
                   "resources": {"peak_allocated_gib": max(x["peak_allocated_gib"] for x in profiles),
                                 "peak_reserved_gib": max(x["peak_reserved_gib"] for x in profiles)}})
    write_json(output / f"evaluation_{exposure}.summary.json", result)
    return result


def _run(args, smoke: bool):
    import torch
    from cbf_ttt.event_writer import backbone_digest, write_memory
    from tasks.cbf_selective import measure

    tokenizer, rows, groups, data_manifest, vocab, episodes, candidates = _prepare(args)
    del tokenizer
    batches = head_schedule(rows, 1 if smoke else args.cycles, args.order_seed)
    if smoke:
        batches = batches[:1]
    output = Path(args.output)
    if not smoke:
        output.mkdir(parents=True)
    model, writer_params, model_meta = setup(args)
    torch.manual_seed(1301)
    head = make_head(model.config.hidden_size, args.projection_size).to("cuda", dtype=torch.float32)
    head_initial = {name: value.detach().cpu().clone() for name, value in head.named_parameters()}
    if args.arm == "head_only":
        for parameter in writer_params.values():
            parameter.requires_grad_(False)
    groups_opt = [{"params": list(head.parameters()), "lr": args.head_lr}]
    if args.arm == "joint_head_writer":
        groups_opt.append({"params": list(writer_params.values()), "lr": args.lr})
    optimizer = torch.optim.AdamW(groups_opt, weight_decay=0.)
    frozen_versions = {name: parameter._version for name, parameter in model.named_parameters()
                       if not parameter.requires_grad}
    writer_initial = {name: parameter.detach().cpu().clone() for name, parameter in writer_params.items()}
    total_exposures = 8 * len(batches)
    manifest = {
        **model_meta, "protocol": "retrieval_head_v1", "arm": args.arm,
        "cycles": 1 if smoke else args.cycles, "question_exposures": total_exposures,
        "optimizer_updates": len(batches), "context_writes_per_update": 2,
        "projection_size": args.projection_size, "temperature": .1,
        "head_lr": args.head_lr, "writer_lr": args.lr,
        "writer_trainable": args.arm == "joint_head_writer", "memory_cap": args.memory_cap,
        "order_seed": args.order_seed, "writer_seed": 301, "head_seed": 1301,
        "groups": groups, "data_sha256": data_manifest["data_sha256"],
        "train_groups": args.train_groups, "dev_groups": args.dev_groups,
        "checkpoints": [0, total_exposures], "test_scored": False,
    }
    records = []

    def step(update, context, left_shift, right_shift, exposure):
        left, right = episodes[context][left_shift], episodes[context][right_shift]
        optimizer.zero_grad(set_to_none=True)
        grad_context = torch.enable_grad() if args.arm == "joint_head_writer" else torch.no_grad()
        with grad_context, torch.autocast("cuda", dtype=torch.bfloat16):
            memories = [write_memory(model, item["context_ids"], max_relative_norm=args.memory_cap)
                        for item in (left, right)]
        losses, correct = [], []
        for memory, episode in zip(memories, (left, right)):
            for row in episode["rows"]:
                if args.arm == "head_only":
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        from cbf_ttt.event_writer import cache_for, forward
                        hidden = forward(model, row["query_ids"], cache_for(model, memory))[:, -1].float()
                    values = value_vectors(model, candidates[context], vocab)
                    with torch.autocast("cuda", enabled=False):
                        logits = head(hidden.detach(), values)
                    target = torch.tensor([candidates[context].index(row["answer"])], device=logits.device)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    hit = int(logits.detach().argmax(dim=-1).item() == target.item())
                else:
                    loss, hit = retrieval_item(model, head, memory, row, candidates[context], vocab)
                losses.append(loss); correct.append(hit)
        loss = sum(losses) / len(losses)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite retrieval-head loss")
        loss.backward()
        head_grad = float(torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1., error_if_nonfinite=True))
        writer_grad = 0.0
        if args.arm == "joint_head_writer":
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                   for parameter in writer_params.values()):
                raise RuntimeError("invalid retrieval-head writer gradient")
            writer_grad = float(torch.nn.utils.clip_grad_norm_(list(writer_params.values()), 1.,
                                                                 error_if_nonfinite=True))
        optimizer.step()
        relative = max(float(delta.detach().float().norm() /
                             model.model.layers[layer].mlp.down_proj.weight.float().norm())
                       for memory in memories for layer, delta in memory.items())
        return {"update": update, "exposure_start": exposure, "exposure_end": exposure + 8,
                "context_id": context, "assignment_shifts": [left_shift, right_shift],
                "loss": float(loss.detach()), "accuracy": statistics.fmean(correct),
                "head_gradient_norm": head_grad, "writer_gradient_norm": writer_grad,
                "max_relative_delta": relative}

    if smoke:
        (context, left, right) = batches[0]
        result, profile = measure(lambda: step(1, context, left, right, 0))
        result.update(profile, arm=args.arm, backbone_sha256=model_meta["backbone_sha256"],
                      memory_cap=args.memory_cap, test_scored=False)
        write_json(output, result)
        return result

    def save(exposure):
        path = output / f"checkpoint_{exposure}.pt"; temp = Path(str(path) + ".tmp")
        torch.save({"exposure": exposure,
                    "writer": {name: value.detach().cpu() for name, value in writer_params.items()},
                    "head": {name: value.detach().cpu() for name, value in head.named_parameters()},
                    "manifest": manifest}, temp); temp.replace(path)

    write_json(output / "training_manifest.json", manifest)
    save(0); evaluate(model, head, rows, vocab, episodes, candidates, output, 0, args.memory_cap)
    exposure = 0
    with (output / "steps.jsonl").open("x") as sink:
        for update, (context, left, right) in enumerate(batches, 1):
            record, profile = measure(lambda: step(update, context, left, right, exposure))
            exposure += 8; record.update(profile); records.append(record)
            sink.write(json.dumps(record) + "\n"); sink.flush()
            if update % 32 == 0:
                print(json.dumps({key: record[key] for key in
                                  ("update", "exposure_end", "loss", "accuracy", "seconds")}), flush=True)
    save(exposure); evaluate(model, head, rows, vocab, episodes, candidates, output,
                             exposure, args.memory_cap)
    if backbone_digest(model) != model_meta["backbone_sha256"]:
        raise RuntimeError("retrieval-head backbone changed")
    writer_changes = {name: float((parameter.detach().cpu() - writer_initial[name]).norm())
                      for name, parameter in writer_params.items()}
    head_changes = {name: float((parameter.detach().cpu() - head_initial[name]).norm())
                    for name, parameter in head.named_parameters()}
    if any(value <= 0 for value in head_changes.values()):
        raise RuntimeError("retrieval head did not change")
    if args.arm == "head_only" and any(value != 0 for value in writer_changes.values()):
        raise RuntimeError("head-only writer changed")
    if args.arm == "joint_head_writer" and any(value <= 0 for value in writer_changes.values()):
        raise RuntimeError("joint writer did not change")
    if not all(value.dtype == torch.float32 for state in optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("retrieval-head optimizer state is not FP32")
    torch.save(optimizer.state_dict(), output / f"optimizer_{exposure}.pt")
    complete = {"arm": args.arm, "question_exposures": exposure,
                "optimizer_updates": len(batches), "backbone_unchanged": True,
                "optimizer_state_fp32": True, "writer_update_norms": writer_changes,
                "head_update_norms": head_changes,
                "first32_loss": statistics.fmean(row["loss"] for row in records[:32]),
                "last32_loss": statistics.fmean(row["loss"] for row in records[-32:]),
                "first32_accuracy": statistics.fmean(row["accuracy"] for row in records[:32]),
                "last32_accuracy": statistics.fmean(row["accuracy"] for row in records[-32:]),
                "mean_update_seconds": statistics.fmean(row["seconds"] for row in records),
                "peak_allocated_gib": max(row["peak_allocated_gib"] for row in records),
                "peak_reserved_gib": max(row["peak_reserved_gib"] for row in records),
                "max_relative_delta": max(row["max_relative_delta"] for row in records),
                "steps_sha256": sha(output / "steps.jsonl"), "test_scored": False}
    write_json(output / "complete.json", complete)
    return complete


def summarize(root: str | Path, save: bool = True):
    root = Path(root); arms = {}
    for arm in ARMS:
        output = root / arm
        manifest = json.loads((output / "training_manifest.json").read_text())
        complete = json.loads((output / "complete.json").read_text())
        checkpoints = {}
        for exposure in manifest["checkpoints"]:
            raw = output / f"evaluation_{exposure}.jsonl"
            records = [json.loads(line) for line in raw.read_text().splitlines() if line]
            stored = json.loads((output / f"evaluation_{exposure}.summary.json").read_text())
            if stored["rows_sha256"] != sha(raw) or _aggregate(records) != {
                    key: stored[key] for key in ("train_probe", "dev")}:
                raise ValueError("retrieval-head evaluation mismatch")
            checkpoints[str(exposure)] = stored
        final = checkpoints[str(manifest["question_exposures"])]
        arms[arm] = {"manifest": manifest, "complete": complete,
                     "checkpoints": checkpoints, "final_gate": final["gate"]}
    result = {"protocol": "retrieval_head_v1", "arms": arms,
              "primary_arm": "joint_head_writer",
              "stage_passed": arms["joint_head_writer"]["final_gate"]["passed"],
              "test_scored": False, "terminal_status": "completed_retrieval_head"}
    if save: write_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "run"):
        item = sub.add_parser(name)
        item.add_argument("--data-root", required=True); item.add_argument("--output", required=True)
        item.add_argument("--model", default="/home/ctj/models/Qwen3-4B")
        item.add_argument("--arm", choices=ARMS, required=True)
        item.add_argument("--train-groups", type=int, default=32)
        item.add_argument("--dev-groups", type=int, default=8)
        item.add_argument("--order-seed", type=int, default=1303)
        item.add_argument("--lr", type=float, default=1e-7)
        item.add_argument("--head-lr", type=float, default=1e-4)
        item.add_argument("--projection-size", type=int, default=256)
        item.add_argument("--memory-cap", type=float, default=.3)
        if name == "run": item.add_argument("--cycles", type=int, default=4)
    summary = sub.add_parser("summarize"); summary.add_argument("--root", required=True)
    args = parser.parse_args()
    result = _run(args, True) if args.command == "smoke" else _run(args, False) if args.command == "run" else summarize(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
