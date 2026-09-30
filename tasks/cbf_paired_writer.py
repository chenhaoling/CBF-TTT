"""Train and gate a paired-fact writer against a matched NLL-only ablation."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import torch

from cbf_ttt.writer import LowRankWriter, memory_choice_nll
from tasks.build_cbf_paired_facts import PROTOCOL, validate
from tasks.cbf_ttt import _load_model
from tasks.cbf_writer import candidate_for, digest, end_profile, extract, start_profile, write_json


CONTROLS = ("none", "raw", "twin", "train_mean")


def paired_loss(correct, wrong, margin=0.1):
    # Both paths receive gradients; each twin also appears as a correct training example.
    return correct + torch.relu(margin + correct - wrong)


@torch.no_grad()
def prediction(model, memory, row, gate=1.0):
    nll = memory_choice_nll(model, {k: gate*v for k, v in memory.items()},
                            row["query_ids"], row["choice_ids"])
    if not torch.isfinite(nll).all():
        raise ValueError("nonfinite evaluation loss")
    return {"nll": float(nll[row["answer_index"]]),
            "correct": int(int(nll.argmin()) == row["answer_index"])}


def aggregate(records):
    policies = list(records[0]["policies"])
    return {"mean_nll": {p: statistics.mean(r["policies"][p]["nll"] for r in records) for p in policies},
            "accuracy": {p: statistics.mean(r["policies"][p]["correct"] for r in records) for p in policies}}


def decision(records, gate, raw_gate):
    summary = aggregate(records)
    controls = {"none": "none", "raw": f"raw_{raw_gate}", "twin": f"twin_{gate}",
                "train_mean": f"train_mean_{gate}"}
    policy = f"writer_{gate}"
    gains = {key: summary["mean_nll"][value] - summary["mean_nll"][policy] for key, value in controls.items()}
    accuracy_gains = {key: summary["accuracy"][policy] - summary["accuracy"][value] for key, value in controls.items()}
    groups = sorted({r["group_id"] for r in records})
    improved = sum(all(statistics.mean(r["policies"][control]["nll"] - r["policies"][policy]["nll"]
                                       for r in records if r["group_id"] == group) > 0.005
                       for control in ("none", f"twin_{gate}")) for group in groups)
    passed = (min(gains.values()) > 0.005 and improved >= math.ceil(len(groups)/2)
              and summary["accuracy"][policy] >= 0.25 and min(accuracy_gains.values()) >= 0.125)
    return {"gate": gate, "mean_nll": summary["mean_nll"][policy], "min_gain": min(gains.values()),
            "gain_vs_controls": gains, "accuracy": summary["accuracy"][policy],
            "accuracy_gain_vs_controls": accuracy_gains, "improved_groups": improved,
            "groups": len(groups), "passed": passed}


@torch.no_grad()
def evaluate(model, writer, rows, training, args):
    device = next(model.parameters()).device
    average = {}
    for row in training:
        generated = writer(candidate_for(row, args.features, args.model, device))
        for layer, tensor in generated.items():
            if layer not in average:
                average[layer] = torch.zeros_like(tensor)
            average[layer].add_(tensor / len(training))
    del generated
    lookup = {r["id"]: r for r in rows}
    result = []
    for row in rows:
        started = start_profile(device)
        delta = candidate_for(row, args.features, args.model, device)
        generated = writer(delta)
        twin = writer(candidate_for(lookup[row["twin_id"]], args.features, args.model, device))
        policies = {"none": prediction(model, {}, row)}
        for gate in (0.5, 1.0):
            for name, memory in (("writer", generated), ("raw", delta), ("twin", twin), ("train_mean", average)):
                policies[f"{name}_{gate}"] = prediction(model, memory, row, gate)
        result.append({"id": row["id"], "group_id": row["group_id"], "twin_id": row["twin_id"],
                       "policies": policies, "profile": end_profile(device, started)})
    return result


def train(args, model, episodes):
    output = Path(args.output)
    if output.exists():
        raise ValueError("training output directory must be new")
    output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = next(model.parameters()).device
    training = sorted((r for r in episodes if r["split"] == "train"), key=lambda r: r["id"])
    dev = sorted((r for r in episodes if r["split"] == "dev"), key=lambda r: r["id"])
    if args.smoke:
        training = [r for r in training if r["group_id"] == training[0]["group_id"]]
        dev = [r for r in dev if r["group_id"] == dev[0]["group_id"]]
    lookup = {r["id"]: r for r in training}
    first = candidate_for(training[0], args.features, args.model, device)
    shapes = {layer: tuple(tensor.shape) for layer, tensor in first.items()}
    del first
    writer = LowRankWriter(shapes, args.rank).to(device)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=args.lr, weight_decay=0.01)
    initial = evaluate(model, writer, dev, training, args)
    raw_gate = min((0.5, 1.0), key=lambda g: aggregate(initial)["mean_nll"][f"raw_{g}"])
    if any(abs(r["policies"][f"writer_{g}"]["nll"]-r["policies"][f"raw_{g}"]["nll"]) > 1e-6
           for r in initial for g in (0.5, 1.0)):
        raise ValueError("initial writer does not match original candidate")
    write_json(output / "initial_dev.json", initial)
    best, history = None, []

    def select(epoch, records):
        nonlocal best
        candidates = [{"epoch": epoch, **decision(records, g, raw_gate)} for g in (0.5, 1.0)]
        history.append({"epoch": epoch, **aggregate(records), "decisions": candidates})
        for candidate in candidates:
            key = lambda item: (item["min_gain"], -item["mean_nll"], -item["epoch"], -item["gate"])
            if best is None or key(candidate) > key(best):
                best = candidate
                torch.save({"protocol": PROTOCOL, "objective": args.objective, "shapes": shapes,
                            "rank": args.rank, "state_dict": {k: v.detach().cpu().clone() for k, v in writer.state_dict().items()},
                            "model": str(Path(args.model).resolve()), "config_sha256": digest(Path(args.model)/"config.json"),
                            "episodes_sha256": digest(args.data)}, output/"best.pt")
                write_json(output/"selected_dev.json", records)
        write_json(output/"history.json", history)
        print(json.dumps({"epoch": epoch, "objective": args.objective, "best": best}), flush=True)

    select(0, initial)
    profiles = []
    with (output/"steps.jsonl").open("w") as sink:
        for epoch in range(1, (1 if args.smoke else args.epochs)+1):
            rng.shuffle(training)
            for row in training:
                started = start_profile(device)
                optimizer.zero_grad(set_to_none=True)
                delta = candidate_for(row, args.features, args.model, device)
                generated = writer(delta)
                correct = memory_choice_nll(model, generated, row["query_ids"], row["choice_ids"])[row["answer_index"]]
                wrong = None
                if args.objective == "paired":
                    twin_delta = candidate_for(lookup[row["twin_id"]], args.features, args.model, device)
                    twin = writer(twin_delta)
                    wrong = memory_choice_nll(model, twin, row["query_ids"], row["choice_ids"])[row["answer_index"]]
                    loss = paired_loss(correct, wrong, args.margin)
                else:
                    loss = correct
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0, error_if_nonfinite=True)
                if not math.isfinite(float(loss.detach())) or float(norm) <= 0:
                    raise ValueError("nonfinite loss or disconnected gradients")
                optimizer.step()
                if any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError("backbone received gradients")
                record = {"step": len(profiles)+1, "epoch": epoch, "id": row["id"],
                          "twin_id": row["twin_id"], "loss": float(loss.detach()),
                          "correct_nll": float(correct.detach()), "twin_nll": float(wrong.detach()) if wrong is not None else None,
                          "grad_norm": float(norm), **end_profile(device, started)}
                sink.write(json.dumps(record)+"\n")
                sink.flush()
                profiles.append(record)
                del correct, loss, wrong, generated, delta
                if args.objective == "paired":
                    del twin, twin_delta
            select(epoch, evaluate(model, writer, dev, training, args))
    selection = {"protocol": PROTOCOL, "objective": args.objective, "seed": args.seed,
                 "rank": args.rank, "lr": args.lr, "margin": args.margin, "epochs": 1 if args.smoke else args.epochs,
                 "best": best, "raw_gate": raw_gate, "smoke": args.smoke,
                 "passed_dev_gate": bool(best["passed"] and not args.smoke), "optimizer_steps": len(profiles),
                 "checkpoint_sha256": digest(output/"best.pt"), "episodes_sha256": digest(args.data)}
    write_json(output/"selection.json", selection)
    write_json(output/"profile_summary.json", {"steps": len(profiles),
        "trainable_parameters": sum(p.numel() for p in writer.parameters()),
        "mean_step_time_s": statistics.mean(r["time_s"] for r in profiles),
        "max_peak_allocated_gib": max((r["peak_allocated_gib"] or 0) for r in profiles),
        "max_peak_reserved_gib": max((r["peak_reserved_gib"] or 0) for r in profiles)})
    print(json.dumps(selection), flush=True)


def load_selected(directory, args, device):
    directory = Path(directory)
    selection = json.loads((directory/"selection.json").read_text())
    if selection["smoke"] or selection["episodes_sha256"] != digest(args.data) or selection["checkpoint_sha256"] != digest(directory/"best.pt"):
        raise ValueError("checkpoint/data mismatch or smoke run")
    state = torch.load(directory/"best.pt", map_location="cpu", weights_only=True)
    if state["model"] != str(Path(args.model).resolve()) or state["config_sha256"] != digest(Path(args.model)/"config.json"):
        raise ValueError("backbone mismatch")
    writer = LowRankWriter(state["shapes"], state["rank"]).to(device)
    writer.load_state_dict(state["state_dict"])
    return writer, selection


def test(args, model, episodes):
    paired_selection = json.loads((Path(args.paired_dir)/"selection.json").read_text())
    if not paired_selection["passed_dev_gate"] or paired_selection["smoke"] or paired_selection["objective"] != "paired":
        raise ValueError("paired dev gate failed; test must remain unscored")
    output = Path(args.output)
    if output.exists():
        raise ValueError("test output directory must be new")
    device = next(model.parameters()).device
    # Freeze and validate both identities before any test scoring.
    models = {name: load_selected(directory, args, device)
              for name, directory in (("paired", args.paired_dir), ("nll", args.nll_dir))}
    if any(selection["objective"] != name for name, (_, selection) in models.items()):
        raise ValueError("objective/checkpoint mismatch")
    output.mkdir(parents=True)
    training = [r for r in episodes if r["split"] == "train"]
    testing = [r for r in episodes if r["split"] == "test"]
    summary = {"protocol": PROTOCOL, "episodes_sha256": digest(args.data), "test_groups": len(testing)//2}
    for name, (writer, selection) in models.items():
        records = evaluate(model, writer, testing, training, args)
        (output/f"{name}_rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        summary[name] = {"checkpoint_sha256": selection["checkpoint_sha256"],
                         "decision": decision(records, selection["best"]["gate"], selection["raw_gate"]),
                         **aggregate(records),
                         "mean_time_s": statistics.mean(r["profile"]["time_s"] for r in records),
                         "max_peak_reserved_gib": max((r["profile"]["peak_reserved_gib"] or 0) for r in records)}
    write_json(output/"summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "train", "test"))
    for name in ("model", "data", "features"):
        parser.add_argument("--"+name, required=True)
    for name in ("output", "paired-dir", "nll-dir"):
        parser.add_argument("--"+name)
    parser.add_argument("--objective", choices=("paired", "nll"), default="paired")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=134)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if args.rank < 1 or args.lr <= 0 or args.margin < 0 or args.epochs < 1 or not 0 <= args.shard < args.shards:
        parser.error("invalid numeric parameters")
    if args.command != "extract" and not args.output:
        parser.error("--output required")
    if args.command == "test" and not (args.paired_dir and args.nll_dir):
        parser.error("both frozen model directories required")
    episodes = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    validate(episodes)
    model = _load_model(args.model, args.device, args.dtype)
    {"extract": extract, "train": train, "test": test}[args.command](args, model, episodes)


if __name__ == "__main__":
    main()
