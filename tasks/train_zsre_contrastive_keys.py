"""Train calibration-only contrastive key alignment for ZsRE contextual memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_qwen_lowrank_forgetting import QwenEmbeddings
from tasks.evaluate_zsre_contextual_readout import (
    SEEDS, UNKNOWN_GRID, aggregate, apply, evaluate, extract, gate, load_split,
    summarize, texts,
)


ARMS = ("identity", "positive_only", "relation_aware")


class AlignmentHead:
    def __init__(self, dim=256, rank=16, seed=0):
        import torch

        torch.manual_seed(seed)
        self.down = torch.nn.Linear(dim, rank, bias=False)
        self.up = torch.nn.Linear(rank, dim, bias=False)
        torch.nn.init.zeros_(self.up.weight)
        self.model = torch.nn.Sequential(self.down, torch.nn.Tanh(), self.up)

    def parameters(self):
        return self.model.parameters()

    def __call__(self, values):
        import torch

        output = values + self.model(values)
        return torch.nn.functional.normalize(output, dim=-1)


def calibration_tensors(sessions, embeddings):
    import torch

    facts = [fact for session in sessions for fact in session["facts"]]
    def stack(field, index=None):
        names = [fact[field] if index is None else fact[field][index] for fact in facts]
        return torch.stack([embeddings.get("entity", name) for name in names])
    return stack("prompt"), stack("rephrase_prompt"), stack("locality_prompts", 0), \
        stack("locality_prompts", 1)


def train_head(sessions, embeddings, arm, seed, epochs=300):
    import torch

    if arm not in ("positive_only", "relation_aware"):
        raise ValueError("alignment head arm")
    prompt, rephrase, local1, local2 = calibration_tensors(sessions, embeddings)
    head = AlignmentHead(embeddings.dim, 16, seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=1e-4)
    labels = torch.arange(prompt.shape[0]); curve = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        p, r = head(prompt), head(rephrase)
        logits = p @ r.T / .07
        info = (torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)) / 2
        locality = torch.tensor(0.)
        if arm == "relation_aware":
            positive = (p * r).sum(-1)
            negatives = torch.stack(((p * head(local1)).sum(-1),
                                     (p * head(local2)).sum(-1)), dim=1)
            locality = torch.nn.functional.softplus(.2 + negatives - positive[:, None]).mean()
        loss = info + .5 * locality
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.)
        optimizer.step()
        if epoch in (1, 10, 50, 100, 250, epochs):
            with torch.no_grad():
                p, r = head(prompt), head(rephrase)
                retrieval = float((p @ r.T).argmax(-1).eq(labels).float().mean())
                local_margin = float(((p * r).sum(-1) - torch.maximum(
                    (p * head(local1)).sum(-1), (p * head(local2)).sum(-1))).mean())
            curve.append({"epoch": epoch, "loss": float(loss.detach()),
                          "info_nce": float(info.detach()), "locality_loss": float(locality.detach()),
                          "prompt_rephrase_top1": retrieval, "locality_margin": local_margin})
    return head, curve


def aligned_embeddings(base, head):
    import torch

    values = {}
    with torch.no_grad():
        for key, value in base.values.items():
            values[key] = head(value.unsqueeze(0))[0] if key[0] == "entity" else value
    return QwenEmbeddings(base.seed, base.dim, values)


def select_unknown(calibration, embeddings_by_arm):
    grid, selected = {}, {}
    for arm in ARMS:
        grid[arm] = {}
        for threshold in UNKNOWN_GRID:
            scores = {}
            for seed in SEEDS:
                records = evaluate(calibration, embeddings_by_arm[arm][str(seed)], threshold, "calibration")
                scores[str(seed)] = aggregate(records)["accuracy"]
            grid[arm][str(threshold)] = scores
        best = max(statistics.fmean(values.values()) for values in grid[arm].values())
        candidates = [float(threshold) for threshold, values in grid[arm].items()
                      if statistics.fmean(values.values()) == best]
        selected[arm] = min(candidates, key=lambda value: (abs(value - .35), value))
    return grid, selected


def comparison(records, primary="relation_aware", baseline="identity", seed=6201):
    categories = {"overall": lambda q: True,
                  "rephrase": lambda q: q["type"].endswith("_rephrase"),
                  "locality": lambda q: q["type"].endswith("_locality")}
    output = {}
    session_ids = sorted({row["session_id"] for rows in records[primary].values() for row in rows})
    for category, predicate in categories.items():
        differences = []
        for session_id in session_ids:
            arm_scores = {}
            for arm in (primary, baseline):
                seed_scores = []
                for rep_seed in SEEDS:
                    row = next(item for item in records[arm][str(rep_seed)]
                               if item["session_id"] == session_id)
                    selected = [q["correct"] for q in row["queries"] if predicate(q)]
                    seed_scores.append(statistics.fmean(selected))
                arm_scores[arm] = statistics.fmean(seed_scores)
            differences.append(arm_scores[primary] - arm_scores[baseline])
        output[category] = {"accuracy_gain": statistics.fmean(differences),
                            "bootstrap_95ci": _bootstrap(differences, seed),
                            "sessions": len(differences)}
        seed += 1
    return output


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    if source_summary["protocol"] != "zsre_contextual_readout_v1" or source_summary["stage_passed"]:
        raise ValueError("contrastive stage requires failed audited ZsRE v1")
    if source_summary["dev_gate"]["checks"]["locality_unknown"]:
        raise ValueError("expected locality-only gate failure")
    config = source_summary["selected_config"]
    artifact = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("source transform mismatch")
    transforms = artifact["transforms"]
    calibration = load_split(args.data_root, "calibration")
    dev_sessions = load_split(args.data_root, "dev")
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    calibration_names = texts(calibration)
    calibration_raw = extract(model, tokenizer, calibration_names, args.batch_size)
    calibration_base, calibration_hashes = apply(
        calibration_raw, calibration_names, transforms, config["regularizer"])
    if calibration_hashes != source_training["calibration_embedding_hashes"][config["internal"]]:
        raise ValueError("calibration embedding bridge mismatch")

    heads, curves, calibration_arms = {}, {}, {arm: {} for arm in ARMS}
    for seed in SEEDS:
        base = calibration_base[str(seed)]; calibration_arms["identity"][str(seed)] = base
        heads[str(seed)], curves[str(seed)] = {}, {}
        for arm in ("positive_only", "relation_aware"):
            head, curve = train_head(calibration, base, arm, args.seed + seed, args.epochs)
            heads[str(seed)][arm] = head; curves[str(seed)][arm] = curve
            calibration_arms[arm][str(seed)] = aligned_embeddings(base, head)
    calibration_grid, thresholds = select_unknown(calibration, calibration_arms)

    dev_names = texts(dev_sessions); dev_raw = extract(model, tokenizer, dev_names, args.batch_size)
    dev_base, dev_hashes = apply(dev_raw, dev_names, transforms, config["regularizer"])
    dev, dev_records, raw_hashes = {}, {arm: {} for arm in ARMS}, {}
    for arm in ARMS:
        per_seed = {}
        for seed in SEEDS:
            embeddings = dev_base[str(seed)] if arm == "identity" else aligned_embeddings(
                dev_base[str(seed)], heads[str(seed)][arm])
            records = evaluate(dev_sessions, embeddings, thresholds[arm], "dev")
            dev_records[arm][str(seed)] = records; per_seed[str(seed)] = aggregate(records)
            path = root / f"dev_{arm}_seed{seed}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            raw_hashes[path.name] = sha(path)
        dev[arm] = summarize(per_seed)
    dev_gate = gate(dev["relation_aware"])
    identity_bridge = all(raw_hashes[f"dev_identity_seed{seed}.jsonl"]
                          == source_training["raw_rows_sha256"][f"dev_seed{seed}.jsonl"]
                          for seed in SEEDS)
    if not identity_bridge:
        raise ValueError("identity arm does not exactly reproduce ZsRE v1")

    test, test_records, test_comparison, test_hashes = None, None, None, None
    test_scored = False
    if dev_gate["passed"]:
        test_sessions = load_split(args.data_root, "test")
        test_names = texts(test_sessions); test_raw = extract(model, tokenizer, test_names, args.batch_size)
        test_base, test_hashes = apply(test_raw, test_names, transforms, config["regularizer"])
        test, test_records = {}, {arm: {} for arm in ARMS}
        for arm in ARMS:
            per_seed = {}
            for seed in SEEDS:
                embeddings = test_base[str(seed)] if arm == "identity" else aligned_embeddings(
                    test_base[str(seed)], heads[str(seed)][arm])
                records = evaluate(test_sessions, embeddings, thresholds[arm], "test")
                test_records[arm][str(seed)] = records; per_seed[str(seed)] = aggregate(records)
                path = root / f"test_{arm}_seed{seed}.jsonl"
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
                raw_hashes[path.name] = sha(path)
            test[arm] = summarize(per_seed)
        test_comparison = comparison(test_records); test_scored = True
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()
    checkpoint = root / "alignment_heads.pt"
    torch.save({"heads": {seed: {arm: head.model.state_dict() for arm, head in values.items()}
                                for seed, values in heads.items()},
                "thresholds": thresholds, "source_config": config, "epochs": args.epochs,
                "seed": args.seed, "rank": 16}, checkpoint)
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": peak_allocated, "gpu_peak_reserved_gib": peak_reserved,
                "curves": curves, "calibration_grid": calibration_grid,
                "selected_unknown": thresholds, "source_config": config,
                "calibration_embedding_hashes": calibration_hashes,
                "dev_embedding_hashes": dev_hashes, "test_embedding_hashes": test_hashes,
                "identity_bridge_exact": identity_bridge, "raw_rows_sha256": raw_hashes,
                "checkpoint_sha256": sha(checkpoint),
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt")}
    manifest = {"protocol": "zsre_contrastive_keys_v1",
                "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"], "seeds": list(SEEDS),
                "arms": list(ARMS), "rank": 16, "epochs": args.epochs,
                "temperature": .07, "locality_margin": .2, "locality_weight": .5,
                "learning_rate": .01, "weight_decay": 1e-4,
                "fit_split": "calibration", "gate_split": "dev",
                "test_release_rule": "primary relation-aware arm passes unchanged ZsRE v1 gate",
                "test_scored": test_scored}
    result = {"protocol": "zsre_contrastive_keys_v1", "selected_unknown": thresholds,
              "dev": dev, "dev_gate": dev_gate, "test": test,
              "test_comparison": test_comparison, "test_scored": test_scored,
              "stage_passed": dev_gate["passed"], "training_seconds": training["seconds"],
              "terminal_status": "completed_zsre_contrastive_keys"}
    write_json(root / "training.json", training); write_json(root / "training_manifest.json", manifest)
    write_json(root / "summary.json", result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True); parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=6151); parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__": main()
