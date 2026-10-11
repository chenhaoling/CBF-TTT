"""Directly train the candidate head on frozen Full-KV query features."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from tasks.evaluate_cbf_candidate_ranking import RANK_METRICS, build_candidate_sets, ranking
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import setup, write_json
from tasks.train_cbf_episodic_binding import _prepare
from tasks.train_cbf_retrieval_head import head_schedule, make_head, value_vectors


ARMS = {"same_lr": 1e-4, "high_lr": 1e-3}
EVAL_SHIFTS = (0, 2)


def feature_key(context: str, shift: int, row_id: str) -> str:
    return f"{context}|{shift}|{row_id}"


def fullkv_hidden(model, row: dict):
    from cbf_ttt.event_writer import cache_for, forward

    cache = cache_for(model, {})
    forward(model, row["context_ids"], cache)
    return forward(model, row["query_ids"], cache)[:, -1].float()


def required_items(rows: list[dict], episodes: dict) -> list[tuple[str, int, dict]]:
    train_contexts = {row["context_id"] for row in rows if row["split"] == "train"}
    items = []
    for context in sorted(episodes):
        base = episodes[context][0]["rows"][0]
        if context in train_contexts:
            shifts = range(4)
        elif base["split"] == "dev":
            shifts = EVAL_SHIFTS
        else:
            continue
        for shift in shifts:
            for row in episodes[context][shift]["rows"]:
                items.append((context, shift, row))
    return items


def extract_features(model, items, vocab, output: Path):
    import torch
    from tasks.cbf_selective import measure

    def collect():
        features = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, (context, shift, row) in enumerate(items, 1):
                features[feature_key(context, shift, row["id"])] = fullkv_hidden(model, row).cpu()
                if index % 64 == 0:
                    print(json.dumps({"feature_rows": index, "feature_total": len(items)}), flush=True)
        values = sorted(vocab)
        vectors = value_vectors(model, values, vocab).cpu()
        return {"features": features, "values": values, "value_vectors": vectors}

    bundle, profile = measure(collect)
    temp = Path(str(output) + ".tmp")
    torch.save(bundle, temp); temp.replace(output)
    return bundle, {"rows": len(items), "sha256": sha(output), **profile}


def _scores(head, query, members, value_map):
    import torch.nn.functional as F

    vectors = value_map[members]
    logits = head(query, vectors).squeeze(0)
    nll = -F.log_softmax(logits.float(), dim=-1)
    return [{"answer": value, "digit_nll": float(nll[index]),
             "answer_nll": float(nll[index])} for index, value in enumerate(members)]


def aggregate(records: list[dict]) -> dict:
    result = {}
    for split in ("train_probe", "dev"):
        subset = [row for row in records if row["evaluation_split"] == split]
        sets = {}
        for name in ("context4", "global16"):
            metrics = {metric: statistics.fmean(row["sets"][name][metric] for row in subset)
                       for metric in RANK_METRICS}
            groups = sorted({row["group_id"] for row in subset})
            group_top1 = [statistics.fmean(row["sets"][name]["top1"] for row in subset
                                           if row["group_id"] == group) for group in groups]
            metrics["top1_group_bootstrap_95ci"] = _bootstrap(group_top1, 1401 if name == "context4" else 1403)
            sets[name] = metrics
        result[split] = {"rows": len(subset), "groups": len({row["group_id"] for row in subset}),
                         "sets": sets}
    return result


def gate(summary: dict, last32_accuracy: float) -> dict:
    dev = summary["dev"]["sets"]
    checks = {
        "last32_train_accuracy": last32_accuracy >= .80,
        "dev_context4_top1": dev["context4"]["top1"] >= .75,
        "dev_global16_top1": dev["global16"]["top1"] >= .50,
        "dev_global16_confidence": dev["global16"]["top1_group_bootstrap_95ci"][0] > .0625,
    }
    return {"checks": checks, "passed": all(checks.values())}


def evaluate(head, rows, episodes, candidates, features, value_map, output: Path, exposure: int):
    import torch

    train_groups = sorted({row["group_id"] for row in rows if row["split"] == "train"})[:8]
    selected = [row for row in rows if row["split"] == "dev" or
                (row["split"] == "train" and row["group_id"] in train_groups)]
    records = []
    raw = output / f"evaluation_{exposure}.jsonl"
    with raw.open("x") as sink, torch.no_grad():
        for base in selected:
            context = base["context_id"]
            row_index = next(index for index, row in enumerate(episodes[context][0]["rows"])
                             if row["id"] == base["id"])
            global_candidates = build_candidate_sets(rows, base)["global16"]
            for shift in EVAL_SHIFTS:
                row = episodes[context][shift]["rows"][row_index]
                query = features[feature_key(context, shift, row["id"])].to("cuda")
                scores = _scores(head, query, global_candidates, value_map)
                record = {"id": row["id"], "group_id": row["group_id"],
                          "evaluation_split": "dev" if row["split"] == "dev" else "train_probe",
                          "context_id": context, "assignment_shift": shift, "truth": row["answer"],
                          "sets": {"context4": ranking(scores, candidates[context], row["answer"]),
                                   "global16": ranking(scores, global_candidates, row["answer"])}}
                sink.write(json.dumps(record) + "\n"); records.append(record)
    summary = aggregate(records)
    summary.update({"exposure": exposure, "rows_sha256": sha(raw), "test_scored": False})
    write_json(output / f"evaluation_{exposure}.summary.json", summary)
    return summary


def run(args, smoke=False):
    import torch
    from cbf_ttt.event_writer import backbone_digest
    from tasks.cbf_selective import measure

    _, rows, groups, data_manifest, vocab, episodes, candidates = _prepare(args)
    batches = head_schedule(rows, 1 if smoke else args.cycles, args.order_seed)
    if smoke:
        batches = batches[:1]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not smoke:
        output.mkdir()
    model, writer, model_meta = setup(args)
    for parameter in writer.values():
        parameter.requires_grad_(False)
    feature_path = output.with_suffix(".features.pt") if smoke else output / "features.pt"
    items = required_items(rows, episodes)
    if smoke:
        context, left, right = batches[0]
        items = [(context, shift, row) for shift in (left, right)
                 for row in episodes[context][shift]["rows"]]
    bundle, feature_meta = extract_features(model, items, vocab, feature_path)
    features = bundle["features"]
    value_map = {value: bundle["value_vectors"][index].to("cuda")
                 for index, value in enumerate(bundle["values"])}
    torch.manual_seed(1301)
    head = make_head(model.config.hidden_size, args.projection_size).to("cuda", dtype=torch.float32)
    initial = {name: value.detach().cpu().clone() for name, value in head.named_parameters()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=ARMS[args.arm], weight_decay=0.)
    manifest = {**model_meta, "protocol": "fullkv_head_control_v1", "arm": args.arm,
                "head_lr": ARMS[args.arm], "cycles": 1 if smoke else args.cycles,
                "optimizer_updates": len(batches), "question_exposures": 8 * len(batches),
                "projection_size": args.projection_size, "temperature": .1,
                "order_seed": args.order_seed, "head_seed": 1301, "writer_frozen": True,
                "groups": groups, "data_sha256": data_manifest["data_sha256"],
                "feature_artifact": feature_meta, "test_scored": False}
    records = []

    def one_step(update, context, left_shift, right_shift, exposure):
        optimizer.zero_grad(set_to_none=True)
        queries, targets = [], []
        for shift in (left_shift, right_shift):
            for row in episodes[context][shift]["rows"]:
                queries.append(features[feature_key(context, shift, row["id"])])
                targets.append(candidates[context].index(row["answer"]))
        queries = torch.cat(queries).to("cuda")
        values = torch.stack([value_map[value] for value in candidates[context]])
        logits = head(queries, values)
        target = torch.tensor(targets, device="cuda")
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True))
        optimizer.step()
        return {"update": update, "exposure_start": exposure, "exposure_end": exposure + 8,
                "context_id": context, "assignment_shifts": [left_shift, right_shift],
                "loss": float(loss.detach()),
                "accuracy": float((logits.detach().argmax(-1) == target).float().mean()),
                "head_gradient_norm": grad}

    if smoke:
        context, left, right = batches[0]
        result, profile = measure(lambda: one_step(1, context, left, right, 0))
        result.update(profile, arm=args.arm, feature_artifact=feature_meta,
                      backbone_sha256=model_meta["backbone_sha256"], test_scored=False)
        write_json(output, result)
        return result

    write_json(output / "training_manifest.json", manifest)
    torch.save({"head": initial, "manifest": manifest}, output / "head_0.pt")
    evaluate(head, rows, episodes, candidates, features, value_map, output, 0)
    exposure = 0
    with (output / "steps.jsonl").open("x") as sink:
        for update, (context, left, right) in enumerate(batches, 1):
            record, profile = measure(lambda: one_step(update, context, left, right, exposure))
            exposure += 8; record.update(profile); records.append(record)
            sink.write(json.dumps(record) + "\n"); sink.flush()
            if update % 32 == 0:
                print(json.dumps({key: record[key] for key in ("update", "loss", "accuracy")}), flush=True)
    final_eval = evaluate(head, rows, episodes, candidates, features, value_map, output, exposure)
    last32_accuracy = statistics.fmean(row["accuracy"] for row in records[-32:])
    final_eval["gate"] = gate(final_eval, last32_accuracy)
    write_json(output / f"evaluation_{exposure}.summary.json", final_eval)
    checkpoint = {"head": {name: value.detach().cpu() for name, value in head.named_parameters()},
                  "manifest": manifest}
    torch.save(checkpoint, output / f"head_{exposure}.pt")
    changes = {name: float((value.detach().cpu() - initial[name]).norm())
               for name, value in head.named_parameters()}
    if any(value <= 0 for value in changes.values()) or backbone_digest(model) != model_meta["backbone_sha256"]:
        raise RuntimeError("fullkv control parameter invariant failed")
    complete = {"arm": args.arm, "optimizer_updates": len(records), "question_exposures": exposure,
                "first32_loss": statistics.fmean(row["loss"] for row in records[:32]),
                "last32_loss": statistics.fmean(row["loss"] for row in records[-32:]),
                "first32_accuracy": statistics.fmean(row["accuracy"] for row in records[:32]),
                "last32_accuracy": last32_accuracy, "head_update_norms": changes,
                "mean_update_seconds": statistics.fmean(row["seconds"] for row in records),
                "peak_reserved_gib": max(row["peak_reserved_gib"] for row in records),
                "steps_sha256": sha(output / "steps.jsonl"), "backbone_unchanged": True,
                "test_scored": False}
    write_json(output / "complete.json", complete)
    return complete


def summarize(root):
    root = Path(root); arms = {}
    for arm in ARMS:
        output = root / arm
        manifest = json.loads((output / "training_manifest.json").read_text())
        complete = json.loads((output / "complete.json").read_text())
        final = json.loads((output / f"evaluation_{manifest['question_exposures']}.summary.json").read_text())
        arms[arm] = {"manifest": manifest, "complete": complete, "final": final,
                     "final_gate": final["gate"]}
    result = {"protocol": "fullkv_head_control_v1", "arms": arms, "primary_arm": "high_lr",
              "stage_passed": arms["high_lr"]["final_gate"]["passed"],
              "test_scored": False, "terminal_status": "completed_fullkv_head_control"}
    write_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("smoke", "run"):
        item = sub.add_parser(command); item.add_argument("--data-root", required=True)
        item.add_argument("--output", required=True); item.add_argument("--model", required=True)
        item.add_argument("--arm", choices=tuple(ARMS), required=True)
        item.add_argument("--train-groups", type=int, default=32)
        item.add_argument("--dev-groups", type=int, default=8)
        item.add_argument("--order-seed", type=int, default=1303)
        item.add_argument("--projection-size", type=int, default=256)
        if command == "run": item.add_argument("--cycles", type=int, default=4)
    summary = sub.add_parser("summarize"); summary.add_argument("--root", required=True)
    args = parser.parse_args()
    result = run(args, args.command == "smoke") if args.command != "summarize" else summarize(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
