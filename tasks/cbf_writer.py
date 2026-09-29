"""Extract, train, and independently evaluate an optional current-content TTT writer."""

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch

from cbf_ttt.runtime import CBFSession
from cbf_ttt.writer import LowRankWriter, memory_nll
from tasks.cbf_ttt import _load_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def read_episodes(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if row.get("protocol") != "task_writer_v1" or row["id"] in seen or (
            row["id"] != row["group_id"] or row["split"] not in ("train", "dev", "test")
        ):
            raise ValueError("duplicate or incompatible writer episode")
        if not row["id"].replace(".", "").isalnum():
            raise ValueError("unsafe source ID")
        for field in ("context_ids", "query_ids", "answer_ids"):
            if not row[field] or any(not isinstance(x, int) or x < 0 for x in row[field]):
                raise ValueError("nonempty token lists required")
        seen.add(row["id"])
    if not rows or {row["split"] for row in rows} != {"train", "dev", "test"}:
        raise ValueError("all three disjoint splits required")
    return rows


def candidate_for(row, directory, model_path, device):
    value = torch.load(Path(directory) / (row["id"] + ".pt"), map_location=device, weights_only=True)
    context_hash = hashlib.sha256(json.dumps(row["context_ids"]).encode()).hexdigest()
    if value["id"] != row["id"] or value["context_sha256"] != context_hash or (
        value["model"] != str(Path(model_path).resolve())
        or value["config_sha256"] != digest(Path(model_path) / "config.json")
    ):
        raise ValueError("feature cache does not match source/model")
    return value["candidates"]


def start_profile(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def end_profile(device, started):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"time_s": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else None}


def extract(args, model, episodes):
    directory = Path(args.features)
    directory.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    records = []
    for index, row in enumerate(episodes):
        if index % args.shards != args.shard:
            continue
        path = directory / (row["id"] + ".pt")
        if path.exists():
            raise ValueError(f"refusing to overwrite {path}")
        started = start_profile(device)
        session = CBFSession(model)
        if len(row["context_ids"]) != session.chunk_size:
            raise ValueError("one complete chunk required")
        session.observe(row["context_ids"])
        # Queries and title labels never enter the feature extraction forward pass.
        candidates = {layer: value.detach().cpu() for layer, value in session.cache.cbf_candidates.items()}
        value = {"id": row["id"], "candidates": candidates,
                 "context_sha256": hashlib.sha256(json.dumps(row["context_ids"]).encode()).hexdigest(),
                 "model": str(Path(args.model).resolve()),
                 "config_sha256": digest(Path(args.model) / "config.json")}
        temporary = path.with_suffix(".incomplete")
        torch.save(value, temporary)
        temporary.replace(path)
        records.append({"id": row["id"], "feature_sha256": digest(path), **end_profile(device, started)})
        del session, candidates, value
        print(json.dumps({"extracted": len(records), "shard": args.shard}), flush=True)
    write_json(directory / f"manifest_shard{args.shard}.json",
               {"episodes_sha256": digest(args.data), "shard": args.shard, "shards": args.shards,
                "records": records})


@torch.no_grad()
def score(model, memory, row, gate=1.0):
    value = memory_nll(model, {layer: gate * tensor for layer, tensor in memory.items()},
                       row["query_ids"], row["answer_ids"])
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite writer evaluation loss")
    return result


@torch.no_grad()
def evaluate_dev(model, writer, rows, args, include_baselines=False):
    device = next(model.parameters()).device
    result = []
    for row in rows:
        delta = candidate_for(row, args.features, args.model, device)
        generated = writer(delta)
        losses = {f"writer_{gate}": score(model, generated, row, gate) for gate in (0.5, 1.0)}
        if include_baselines:
            losses["none"] = score(model, {}, row)
            losses.update({f"raw_{gate}": score(model, delta, row, gate) for gate in (0.5, 1.0)})
        result.append({"id": row["id"], **losses})
    return result


def means(rows):
    return {key: statistics.mean(row[key] for row in rows) for key in rows[0] if key != "id"}


def train(args, model, episodes):
    output = Path(args.output)
    if output.exists():
        raise ValueError("training output directory must be new")
    output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = next(model.parameters()).device
    training = [row for row in episodes if row["split"] == "train"]
    dev = [row for row in episodes if row["split"] == "dev"]
    if args.smoke:
        training, dev = training[:1], dev[:1]
    first = candidate_for(training[0], args.features, args.model, device)
    shapes = {layer: tuple(value.shape) for layer, value in first.items()}
    del first
    writer = LowRankWriter(shapes, args.rank).to(device)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=args.lr, weight_decay=0.01)
    baseline = evaluate_dev(model, writer, dev, args, include_baselines=True)
    baseline_means = means(baseline)
    raw_gate = min((0.5, 1.0), key=lambda gate: baseline_means[f"raw_{gate}"])
    baseline_by_id = {row["id"]: row for row in baseline}
    best = {"mean_nll": math.inf}
    history = []
    write_json(output / "baseline_dev.json", baseline)

    def select(epoch, evaluated):
        summary = means(evaluated)
        gate = min((0.5, 1.0), key=lambda value: summary[f"writer_{value}"])
        loss = summary[f"writer_{gate}"]
        history.append({"epoch": epoch, "dev_means": summary})
        if loss < best["mean_nll"]:
            best.update({"epoch": epoch, "gate": gate, "mean_nll": loss,
                         "gain_vs_none": baseline_means["none"] - loss,
                         "gain_vs_best_raw": baseline_means[f"raw_{raw_gate}"] - loss,
                         "improved_papers": sum(baseline_by_id[row["id"]]["none"] -
                                                row[f"writer_{gate}"] > 0.005 for row in evaluated)})
            torch.save({"protocol": "task_writer_v1", "shapes": shapes, "rank": args.rank,
                        "state_dict": {key: value.detach().cpu().clone() for key, value in writer.state_dict().items()},
                        "model": str(Path(args.model).resolve()),
                        "config_sha256": digest(Path(args.model) / "config.json"),
                        "episodes_sha256": digest(args.data)}, output / "best.pt")
            write_json(output / "selected_dev.json", evaluated)
        write_json(output / "history.json", history)
        print(json.dumps({"epoch": epoch, "dev_mean": loss, "selected": best}), flush=True)

    select(0, [{key: value for key, value in row.items() if key == "id" or key.startswith("writer")}
               for row in baseline])
    epochs = 1 if args.smoke else args.epochs
    with (output / "steps.jsonl").open("w") as sink:
        step = 0
        for epoch in range(1, epochs + 1):
            random.shuffle(training)
            for row in training:
                started = start_profile(device)
                delta = candidate_for(row, args.features, args.model, device)
                optimizer.zero_grad(set_to_none=True)
                generated = writer(delta)
                loss = memory_nll(model, generated, row["query_ids"], row["answer_ids"])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0, error_if_nonfinite=True)
                if not math.isfinite(float(loss.detach())) or float(norm) <= 0:
                    raise ValueError("nonfinite loss or disconnected writer gradients")
                optimizer.step()
                if any(parameter.grad is not None for parameter in model.parameters()):
                    raise RuntimeError("backbone unexpectedly received gradients")
                step += 1
                profile = end_profile(device, started)
                sink.write(json.dumps({"step": step, "epoch": epoch, "id": row["id"],
                                       "loss": float(loss.detach()), "grad_norm": float(norm), **profile}) + "\n")
                sink.flush()
                del loss, generated, delta
            select(epoch, evaluate_dev(model, writer, dev, args))
    passed = (not args.smoke and best["gain_vs_none"] > 0.005 and
              best["gain_vs_best_raw"] > 0.005 and best["improved_papers"] >= math.ceil(len(dev) / 2))
    selection = {"protocol": "task_writer_v1", "best": best, "raw_gate": raw_gate,
                 "passed_dev_gate": passed, "smoke": args.smoke,
                 "seed": args.seed, "lr": args.lr, "rank": args.rank, "epochs": epochs,
                 "train_papers": len(training), "dev_papers": len(dev), "optimizer_steps": step,
                 "baseline_dev_means": baseline_means, "checkpoint_sha256": digest(output / "best.pt"),
                 "episodes_sha256": digest(args.data)}
    write_json(output / "selection.json", selection)
    profiles = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    write_json(output / "profile_summary.json", {
        "steps": len(profiles), "trainable_parameters": sum(value.numel() for value in writer.parameters()),
        "mean_step_time_s": statistics.mean(row["time_s"] for row in profiles),
        "max_step_time_s": max(row["time_s"] for row in profiles),
        "max_peak_allocated_gib": max((row["peak_allocated_gib"] or 0) for row in profiles),
        "max_peak_reserved_gib": max((row["peak_reserved_gib"] or 0) for row in profiles),
    })
    print(json.dumps(selection), flush=True)


def evaluate_test(args, model, episodes):
    directory = Path(args.checkpoint_dir)
    selection = json.loads((directory / "selection.json").read_text())
    if not selection["passed_dev_gate"] or selection["smoke"]:
        raise ValueError("dev stage gate failed; test must remain unscored")
    if digest(directory / "best.pt") != selection["checkpoint_sha256"] or (
        digest(args.data) != selection["episodes_sha256"]
    ):
        raise ValueError("frozen checkpoint/data fingerprint mismatch")
    state = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    if state["model"] != str(Path(args.model).resolve()) or state["config_sha256"] != digest(Path(args.model) / "config.json"):
        raise ValueError("checkpoint backbone identity mismatch")
    output = Path(args.output)
    if output.exists():
        raise ValueError("test output directory must be new")
    output.mkdir(parents=True)
    device = next(model.parameters()).device
    writer = LowRankWriter(state["shapes"], state["rank"]).to(device)
    writer.load_state_dict(state["state_dict"])
    training = [row for row in episodes if row["split"] == "train"]
    testing = sorted((row for row in episodes if row["split"] == "test"), key=lambda row: row["id"])
    if len(testing) < 2:
        raise ValueError("at least two test sources required for mismatched writing")
    average = {}
    with torch.no_grad():
        for row in training:
            value = writer(candidate_for(row, args.features, args.model, device))
            for layer, tensor in value.items():
                if layer not in average:
                    average[layer] = torch.zeros_like(tensor)
                average[layer].add_(tensor / len(training))
        records = []
        gate = selection["best"]["gate"]
        for index, row in enumerate(testing):
            started = start_profile(device)
            delta = candidate_for(row, args.features, args.model, device)
            generated = writer(delta)
            wrong = writer(candidate_for(testing[(index + 1) % len(testing)], args.features, args.model, device))
            losses = {"none": score(model, {}, row),
                      "raw": score(model, delta, row, selection["raw_gate"]),
                      "writer": score(model, generated, row, gate),
                      "mismatched": score(model, wrong, row, gate),
                      "train_mean": score(model, average, row, gate)}
            losses.update({f"writer_{g}": score(model, generated, row, g) for g in (0.5, 1.0)})
            records.append({"id": row["id"], **losses, "profile": end_profile(device, started)})
    (output / "test_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    summary = {key: statistics.mean(row[key] for row in records)
               for key in ("none", "raw", "writer", "mismatched", "train_mean", "writer_0.5", "writer_1.0")}
    improved = sum(row["none"] - row["writer"] > 0.005 for row in records)
    gains = {key: summary[key] - summary["writer"] for key in ("none", "raw", "mismatched", "train_mean")}
    write_json(output / "summary.json", {"protocol": "task_writer_v1", "test_papers": len(records),
               "checkpoint_sha256": selection["checkpoint_sha256"], "gate": gate,
               "raw_gate": selection["raw_gate"], "mean_nll": summary,
               "writer_gain_vs_controls": gains, "improved_papers": improved,
               "passed_test_gate": improved >= math.ceil(len(records) / 2) and min(gains.values()) > 0.005,
               "mean_time_s": statistics.mean(row["profile"]["time_s"] for row in records),
               "max_peak_reserved_gib": max((row["profile"]["peak_reserved_gib"] or 0) for row in records)})
    print((output / "summary.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "train", "test"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards or args.epochs < 1 or args.lr <= 0:
        parser.error("invalid shard or training parameters")
    if args.command in ("train", "test") and not args.output:
        parser.error("--output required")
    if args.command == "test" and not args.checkpoint_dir:
        parser.error("--checkpoint-dir required")
    episodes = read_episodes(args.data)
    model = _load_model(args.model, args.device, args.dtype)
    if args.command == "extract":
        extract(args, model, episodes)
    elif args.command == "train":
        train(args, model, episodes)
    else:
        evaluate_test(args, model, episodes)


if __name__ == "__main__":
    main()
