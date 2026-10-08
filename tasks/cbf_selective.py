"""Collect and audit the preregistered selective-forgetting oracle pilot."""

import argparse
import collections
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from pathlib import Path


def mean(values):
    return statistics.fmean(values)


def action_name(action):
    return "cohort_" + "_".join(f"{a:g}" for a in action)


def ci(values):
    rng = random.Random(108)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [draws[249], draws[9749]]


def audit_sources(metadata, scenes):
    """Check source-level rejection and exact chunk independence separately."""
    candidates = {d["sha256"]: d for d in metadata["candidate_sources"]}
    eligible = [d["sha256"] for d in metadata["candidate_sources"] if d["matching_rows"] == 0]
    selected = [d["sha256"] for d in metadata["documents"]]
    if len(candidates) != len(metadata["candidate_sources"]) or selected != eligible[:16]:
        raise ValueError("source filtering/selection mismatch")
    owners = {}
    for scene in scenes:
        for ids in scene["prefix"]+[f["ids"] for f in scene["future"]]:
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            owners.setdefault(digest, set()).add(scene["group"])
    duplicates = sum(len(groups) > 1 for groups in owners.values())
    if duplicates:
        raise ValueError("identical full chunks across source groups")
    return {"training_rows_scanned": metadata["training_rows_scanned"],
            "candidate_sources": len(candidates), "rejected_sources": len(candidates)-len(eligible),
            "selected_sources": len(selected), "selected_sources_with_matches": 0,
            "cross_group_identical_full_chunks": duplicates}


def measure(fn):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, {"seconds": time.perf_counter() - start,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}


def future_loss(session, scene, action=(1., 1., 1.)):
    losses = []
    for future in scene["future"]:
        session.advance(future["ids"], action)
        losses.append(session.score_answer(future["query_ids"], future["answer_ids"]))
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError("nonfinite loss")
    return losses


def collect(args):
    import torch
    from cbf_ttt.selective import CohortSession, GRID, FIXED
    from tasks.cbf_ttt import _load_model

    if not 0 <= args.shard < args.shards:
        raise ValueError("invalid shard")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = Path(args.data).read_bytes()
    scenes = [json.loads(line) for line in raw.splitlines()]
    scenes = [s for s in scenes if s["group"] % args.shards == args.shard]
    if args.smoke:
        scenes = scenes[:1]
    torch.manual_seed(108)
    model = _load_model(args.model, "cuda", "bfloat16")
    actions = ((0., 0., 0.), (1., 1., 0.), (1., 1., 1.)) if args.smoke else GRID
    with output.open("x") as sink:
        for scene in scenes:
            start = time.perf_counter()
            if scene["chunk_size"] != model.config.ttt_chunk:
                raise ValueError("scene/checkpoint chunk mismatch")
            torch.cuda.empty_cache()
            def prefix():
                session = CohortSession(model)
                for ids in scene["prefix"][:-1]:
                    session.advance(ids)
                session._forward(scene["prefix"][-1], collect=True)
                return session
            session, prefix_profile = measure(prefix)
            results = {}
            for action in actions:
                name = action_name(action)
                def counterfactual():
                    branch = session.clone()
                    branch.commit_cohorts(action)
                    return future_loss(branch, scene)
                losses, profile = measure(counterfactual)
                results[name] = {"retention": action, "losses": losses, "nll": mean(losses), **profile}
                print(json.dumps({"scene": scene["id"], "action": name, **profile}), flush=True)
            # Release common prefix before full-session fixed-policy baselines.
            del session
            for name, action in FIXED.items():
                def fixed():
                    branch = CohortSession(model)
                    for ids in scene["prefix"]:
                        branch.advance(ids, action)
                    return future_loss(branch, scene, action)
                losses, profile = measure(fixed)
                results["fixed_"+name] = {"retention": action, "losses": losses,
                                          "nll": mean(losses), **profile}
                print(json.dumps({"scene": scene["id"], "action": "fixed_"+name, **profile}), flush=True)
            record = {k: scene[k] for k in ("id", "group", "split", "regime", "sources", "protocol")}
            record.update({"results": results, "prefix_profile": prefix_profile,
                           "scene_seconds": time.perf_counter()-start,
                           "data_sha256": hashlib.sha256(raw).hexdigest(), "model": args.model,
                           "smoke": args.smoke, "shard": args.shard, "shards": args.shards})
            sink.write(json.dumps(record)+"\n")
            sink.flush()
            print(json.dumps({"completed_scene": scene["id"], "seconds": record["scene_seconds"]}), flush=True)


def summarize(args):
    from tasks.build_cbf_selective import REGIMES
    metadata = json.loads(Path(args.data+".meta.json").read_text())
    if metadata["overlap_matching_rows"] or metadata["training_rows_scanned"] < 1:
        raise ValueError("source overlap audit failed")
    raw = Path(args.data).read_bytes()
    expected = {s["id"]: s for s in (json.loads(line) for line in raw.splitlines())}
    source_audit = audit_sources(metadata, expected.values())
    grid_names = [action_name(a) for a in itertools.product((0., .5, 1.), repeat=3)]
    global_names = [action_name((a,)*3) for a in (0., .5, 1.)]
    fixed_names = ["fixed_"+n for n in ("global_clear", "global_half", "window2", "window3")]
    rows = [json.loads(line) for path in args.inputs for line in Path(path).read_text().splitlines()]
    if len(rows) != 32 or len(expected) != 32 or {r["id"] for r in rows} != set(expected):
        raise ValueError("incomplete or duplicate scenes")
    if len({row["model"] for row in rows}) != 1:
        raise ValueError("shards used different checkpoints")
    doc_hashes = [doc["sha256"] for doc in metadata["documents"]]
    if len(doc_hashes) != 16 or len(set(doc_hashes)) != 16:
        raise ValueError("source groups are not independent")
    gains, all_profiles = [], []
    for row in rows:
        ref = expected[row["id"]]
        if row["smoke"] or row["data_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("smoke or incompatible data hash")
        if any(row[k] != ref[k] for k in ("group", "split", "regime", "sources", "protocol")):
            raise ValueError("scene identity mismatch")
        group = row["group"]
        if row["sources"] != doc_hashes[2*group:2*group+2]:
            raise ValueError("source assignment mismatch")
        if set(row["results"]) != set(grid_names+fixed_names):
            raise ValueError("missing or extra candidate policy")
        for result in row["results"].values():
            if len(result["losses"]) != 3 or not all(math.isfinite(x) for x in result["losses"]):
                raise ValueError("invalid future loss")
            if abs(mean(result["losses"])-result["nll"]) > 1e-10:
                raise ValueError("invalid aggregate loss")
            if any(not math.isfinite(result[k]) or result[k] <= 0 for k in
                   ("seconds", "peak_allocated_gib", "peak_reserved_gib")):
                raise ValueError("missing resource profile")
        all_profiles.extend(row["results"][name] for name in grid_names)
        values = {name: result["nll"] for name, result in row["results"].items()}
        best_local = min(grid_names, key=values.get)
        best_global = min(global_names, key=values.get)
        best_strong = min(global_names+fixed_names, key=values.get)
        # A single action is chosen on mean future loss; do not reselect by horizon.
        gains.append({"group": group, "split": row["split"], "regime": row["regime"],
                      "local_action": best_local, "strong_action": best_strong,
                      "local_nll": values[best_local], "global_nll": values[best_global],
                      "strong_nll": values[best_strong],
                      "vs_global": values[best_global]-values[best_local],
                      "vs_strong": values[best_strong]-values[best_local],
                      "horizon_gain": [b-a for a, b in zip(row["results"][best_local]["losses"],
                                                          row["results"][best_strong]["losses"])]})
    report = {"protocol": "selective_forgetting_v1", "audit_passed": True,
              "source_audit": source_audit,
              "scenes": len(rows), "groups": 8, "data_sha256": hashlib.sha256(raw).hexdigest(),
              "model": sorted(set(r["model"] for r in rows)), "splits": {}}
    for split in ("pilot", "confirm"):
        selected = [g for g in gains if g["split"] == split]
        group_values = [mean(g["vs_strong"] for g in selected if g["group"] == group)
                        for group in sorted({g["group"] for g in selected})]
        by_regime = {regime: mean(g["vs_strong"] for g in selected if g["regime"] == regime)
                     for regime in REGIMES}
        interval = ci(group_values)
        passed = (mean(group_values) > .005 and sum(x > .005 for x in group_values) >= 3
                  and sum(x > .005 for x in by_regime.values()) >= 2 and interval[0] > 0)
        report["splits"][split] = {
            "oracle_nll": {key: mean(g[key] for g in selected)
                           for key in ("local_nll", "global_nll", "strong_nll")},
            "gain_vs_global_oracle": mean(g["vs_global"] for g in selected),
            "gain_vs_strong_controls": mean(group_values), "group_gains": group_values,
            "group_bootstrap_95ci": interval, "regime_gains": by_regime,
            "regime_gains_vs_global": {regime: mean(g["vs_global"] for g in selected if g["regime"] == regime)
                                       for regime in REGIMES},
            "regime_oracle_action_counts": {
                regime: dict(collections.Counter(g["local_action"] for g in selected if g["regime"] == regime))
                for regime in REGIMES},
            "horizon_gains_fixed_selected_action": [mean(g["horizon_gain"][i] for g in selected)
                                                     for i in range(3)],
            "oracle_action_counts": dict(collections.Counter(g["local_action"] for g in selected)),
            "strong_control_counts": dict(collections.Counter(g["strong_action"] for g in selected)),
            "policy_mean_nll": {name: mean(r["results"][name]["nll"] for r in rows if r["split"] == split)
                                for name in grid_names+fixed_names}, "passed": passed}
    report["passed_stage_a"] = all(s["passed"] for s in report["splits"].values())
    # Diagnostic specified before reading real NLL: freeze the pilot's best
    # constant policy, then report its confirm loss without reselecting there.
    report["pilot_selected_constant_policies"] = {}
    for family, names in (("same_state_cohort", grid_names), ("all_fixed_policies", grid_names+fixed_names)):
        selected_policy = min(names, key=report["splits"]["pilot"]["policy_mean_nll"].get)
        report["pilot_selected_constant_policies"][family] = {
            "policy": selected_policy,
            "pilot_nll": report["splits"]["pilot"]["policy_mean_nll"][selected_policy],
            "confirm_nll": report["splits"]["confirm"]["policy_mean_nll"][selected_policy],
            "confirm_gain_of_local_oracle": report["splits"]["confirm"]["policy_mean_nll"][selected_policy]
                                            - report["splits"]["confirm"]["oracle_nll"]["local_nll"]}
    report["next_action"] = ("preregister stage B on fresh source groups" if report["passed_stage_a"]
                             else "stop scaling; no multi-decision or controller training")
    seconds = sorted(r["seconds"] for r in all_profiles)
    report["profile"] = {"counterfactual_labels": len(all_profiles), "mean_label_seconds": mean(seconds),
                         "p95_label_seconds": seconds[math.ceil(.95*len(seconds))-1],
                         "max_peak_allocated_gib": max(r["peak_allocated_gib"] for r in all_profiles),
                         "max_peak_reserved_gib": max(r["peak_reserved_gib"] for r in all_profiles),
                         "summed_scene_seconds": sum(r["scene_seconds"] for r in rows),
                         "mean_scene_seconds": mean(r["scene_seconds"] for r in rows)}
    Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("collect")
    run.add_argument("--data", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--shard", type=int, default=0)
    run.add_argument("--shards", type=int, default=1)
    run.add_argument("--smoke", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("--data", required=True)
    summary.add_argument("--inputs", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    args = parser.parse_args()
    (collect if args.command == "collect" else summarize)(args)


if __name__ == "__main__":
    main()
