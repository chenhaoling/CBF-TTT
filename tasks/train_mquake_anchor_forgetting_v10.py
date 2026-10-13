"""Train causal forgetting with the frozen MQuAKE anchor-4 candidate resolver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossfit_calibrated_forgetting_v5 import CalibratedEnsemble, GATE_SEEDS, calibration_threshold
from tasks.train_factorized_causal_forgetting_v4 import PairEvidence, aggregate, evaluate, oracle_rows, paired_bootstrap, payload
from tasks.train_factorized_scope_candidate_v3 import scope_key
from tasks.train_long_session_candidate_v2 import Ranker
from tasks.train_long_session_sparse_gate_v1 import embedding_matrices, fit_snapshots, matched_random_schedules
from tasks.train_mquake_temporal_adaptation_v8 import load_split, load_texts


PROTOCOL = "mquake_anchor_forgetting_v10"
EPOCH = 50
SHORTLIST = 4


class AnchorResolver:
    def __init__(self, evidence, artifacts):
        self.evidence = evidence; self.members = []; self.anchor_tables = {}
        for artifact in artifacts:
            ranker = Ranker(artifact["dim"], artifact["seed"])
            ranker.model.load_state_dict(artifact["state"]); ranker.model.eval()
            self.members.append((ranker, artifact))

    def prepare(self, episode):
        import torch
        identifier = episode["episode_id"]
        if identifier in self.anchor_tables:
            return self.anchor_tables[identifier]
        indices = torch.tensor([self.evidence.index[row["units"][0]]
                                for row in episode["sessions"]])
        matrices = []
        for matrix in self.evidence.scope_matrices.values():
            values = matrix[indices]; matrices.append(values @ values.T)
        result = torch.stack(matrices).mean(0)
        self.anchor_tables[identifier] = result
        return result

    def resolve(self, memory, incoming, step, length, episode_scores):
        import torch
        del length, episode_scores
        # Session id prefix is shared by every event in one episode.
        episode_id = incoming["episode_id"]
        anchor = self.anchor_tables[episode_id]
        positions = [row["position"] for row in memory]
        scores = anchor[torch.tensor(positions), step]
        shortlist = sorted(range(len(memory)), key=lambda i: (-float(scores[i]), i))[:SHORTLIST]
        pairs = [self.evidence.pair(memory[i]["units"], incoming["units"]) for i in shortlist]
        features = torch.stack([row["scope_feature"] for row in pairs])
        member_scores = []
        with torch.no_grad():
            for ranker, artifact in self.members:
                member_scores.append(ranker((features - artifact["mean"]) / artifact["std"]))
        learned = torch.stack(member_scores).mean(0)
        order = sorted(range(len(shortlist)), key=lambda i: (-float(learned[i]), shortlist[i]))
        selected_local = order[0]; selected = shortlist[selected_local]
        score_order = sorted(range(len(memory)), key=lambda i: (-float(scores[i]), i))
        second = float(scores[score_order[1]]) if len(score_order) > 1 else -1.
        pair = pairs[selected_local]
        feature = torch.cat((pair["scope_feature"], torch.tensor([
            float(scores[selected]), float(scores[score_order[0]]) - second,
        ]), pair["payload_feature"]))
        return selected, feature, {"shortlist_positions": [memory[i]["position"] for i in shortlist],
                                   "selected_position": memory[selected]["position"],
                                   "scope_score": float(scores[selected])}


def attach_episode_id(episodes):
    for episode in episodes:
        for row in episode["sessions"]:
            row["episode_id"] = episode["episode_id"]


def candidate_metrics(rows):
    positives = [row for row in rows if row["label"]]
    return {"shortlist_recall": statistics.fmean(row["shortlist_correct"] for row in positives),
            "candidate_top1": statistics.fmean(row["candidate_correct"] for row in positives)}


def tune_gate(metric, candidate):
    checks = {"shortlist_recall_at_least_99": candidate["shortlist_recall"] >= .99,
              "candidate_top1_at_least_99": candidate["candidate_top1"] >= .99,
              "current_retention_at_least_97": metric["current_retention"] >= .97,
              "non_target_retention_at_least_95": metric["non_target_retention"] >= .95,
              "hard_retain_at_least_95": metric["by_event_kind"]["hard_retain"]["accuracy"] >= .95,
              "pair_state_at_least_75": metric["pair_state_accuracy"] >= .75}
    return {"checks": checks, "passed": all(checks.values())}


def development_gate(metric, candidate, versus_random):
    checks = {"shortlist_recall_at_least_99": candidate["shortlist_recall"] >= .99,
              "candidate_top1_at_least_99": candidate["candidate_top1"] >= .99,
              "current_retention_at_least_97": metric["current_retention"] >= .97,
              "non_target_retention_at_least_95": metric["non_target_retention"] >= .95,
              "hard_retain_at_least_95": metric["by_event_kind"]["hard_retain"]["accuracy"] >= .95,
              "stale_deletion_at_least_85": metric["stale_deletion"] >= .85,
              "pair_state_at_least_85": metric["pair_state_accuracy"] >= .85,
              "every_length_pair_at_least_80": min(
                  row["pair_state_accuracy"] for row in metric["by_length"].values()) >= .80,
              "random_gain_lower_bound_above_zero": versus_random["bootstrap_95ci"][0] > 0}
    return {"checks": checks, "passed": all(checks.values())}


def compact(rows):
    keys = ("episode_id", "subject", "length", "step", "event_kind", "label",
            "candidate_correct", "shortlist_correct")
    return [{key: row[key] for key in keys} for row in rows]


def prepare_episodes(resolver, episodes):
    attach_episode_id(episodes)
    for episode in episodes:
        resolver.prepare(episode)


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    train, data_manifest = load_split(args.data_root, "train"); tune, _ = load_split(args.data_root, "tune")
    texts = load_texts(args.data_root, data_manifest)
    scopes = {identifier: scope_key(text) for identifier, text in texts.items()}
    payloads = sorted({value for text in texts.values() if (value := payload(text))})
    source = Path(args.source_root); source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)["transforms"]
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
    v8 = Path(args.v8_root); v8_training = json.loads((v8 / "training.json").read_text())
    v8_checkpoint = v8 / "mquake_adapted_forgetting.pt"
    if v8_training["checkpoint_sha256"] != sha(v8_checkpoint):
        raise ValueError("V8 checkpoint mismatch")
    candidate_artifacts = torch.load(v8_checkpoint, map_location="cpu", weights_only=False)["candidate_artifacts"]

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    names = {"entity": sorted(set(scopes.values())), "value": sorted({"unknown", *payloads})}
    raw = extract(model, tokenizer, names, args.batch_size)
    embeddings, embedding_hashes = apply(raw, names, transforms,
                                         source_summary["selected_config"]["regularizer"])
    scope_matrices, index = embedding_matrices(embeddings, scopes, scopes)
    del model, raw; torch.cuda.empty_cache()
    evidence = PairEvidence(scope_matrices, embeddings, index, texts, scopes)
    resolver = AnchorResolver(evidence, list(candidate_artifacts.values()))
    prepare_episodes(resolver, train); prepare_episodes(resolver, tune)
    train_rows = oracle_rows(train, evidence, resolver); tune_rows = oracle_rows(tune, evidence, resolver)
    tune_candidate = candidate_metrics(tune_rows)

    members, calibration = [], {}
    for seed in GATE_SEEDS:
        artifact = fit_snapshots(train_rows, seed, EPOCH)[EPOCH]
        threshold, grid, feasible = calibration_threshold(artifact, tune, evidence, resolver)
        members.append((artifact, threshold)); calibration[str(seed)] = {
            "threshold": threshold, "calibration_feasible": feasible, "grid": grid}
        print(f"stage=gate_calibrated seed={seed} threshold={threshold}", flush=True)
    controller = CalibratedEnsemble(members)
    tune_records = evaluate(tune, evidence, "controller", resolver, controller, seed=35101)
    tune_metric = aggregate(tune_records); prerequisite = tune_gate(tune_metric, tune_candidate)
    (root / "tune_controller.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tune_records))
    (root / "tune_candidate_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in compact(tune_rows)))
    summary = {"protocol": PROTOCOL, "candidate_policy": "anchor_4_then_v8_ranker",
               "tune": tune_metric, "tune_candidate": tune_candidate,
               "tune_prerequisite": prerequisite, "development_scored": False,
               "mquake_cf_used": False, "longmemeval_used": False, "locomo_scored": False,
               "next_stage": "stop_anchor_tune",
               "terminal_status": "completed_mquake_anchor_forgetting_v10_tune"}
    deletion_count = None
    if prerequisite["passed"]:
        dev, _ = load_split(args.data_root, "dev"); prepare_episodes(resolver, dev)
        dev_rows = oracle_rows(dev, evidence, resolver); dev_candidate = candidate_metrics(dev_rows)
        records = evaluate(dev, evidence, "controller", resolver, controller, seed=35102)
        metric = aggregate(records); deletion_count = sum(
            decision["deleted"] is not None for row in records for decision in row["decisions"])
        schedules = matched_random_schedules(dev, deletion_count, 35103)
        controls = {"retain_all": evaluate(dev, evidence, "retain_all"),
                    "random_matched": evaluate(dev, evidence, "random_matched", schedules=schedules, seed=35104),
                    "oracle": evaluate(dev, evidence, "oracle")}
        versus_random = paired_bootstrap(records, controls["random_matched"], 35105)
        gate = development_gate(metric, dev_candidate, versus_random)
        (root / "dev_controller.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        (root / "dev_candidate_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in compact(dev_rows)))
        for name, rows in controls.items():
            (root / f"dev_{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary.update({"development_scored": True, "development": metric,
                        "development_candidate": dev_candidate,
                        "controls": {name: aggregate(rows) for name, rows in controls.items()},
                        "primary_vs_random": versus_random, "development_gate": gate,
                        "next_stage": "preregister_locomo_confirmation" if gate["passed"] else "stop_anchor_dev_no_retuning",
                        "terminal_status": "completed_mquake_anchor_forgetting_v10"})
    checkpoint = root / "mquake_anchor_forgetting.pt"
    torch.save({"candidate_artifacts": candidate_artifacts, "gate_members": members}, checkpoint)
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "feature_dim": train_rows[0]["feature"].numel(), "calibration": calibration,
                "embedding_hashes": embedding_hashes, "gate_train_rows": len(train_rows),
                "matched_random_deletions": deletion_count,
                "raw_rows_sha256": {path.name: sha(path) for path in root.glob("*.jsonl")},
                "checkpoint_sha256": sha(checkpoint),
                "v8_hashes": {"summary": sha(v8 / "summary.json"), "training": sha(v8 / "training.json"),
                              "checkpoint": sha(v8_checkpoint)}}
    manifest = {"protocol": PROTOCOL, "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt"),
                "model": args.model, "tokenizer": args.tokenizer, "loading_info": model_meta["loading_info"],
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "candidate_policy": "anchor_4_then_v8_ranker", "shortlist": SHORTLIST,
                "gate_epoch": EPOCH, "gate_seeds": list(GATE_SEEDS), "refit_after_calibration": False,
                "development_labels_loaded_conditionally": True,
                "payload_excluded_from_candidate_ranking": True,
                "candidate_age_feature": False, "unit_position_feature": False,
                "mquake_cf_used": False, "longmemeval_used": False, "locomo_used": False}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--v8-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__": main()
