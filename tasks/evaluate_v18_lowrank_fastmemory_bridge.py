"""Evaluate V18 selective forgetting in decomposable Qwen low-rank fast memory.

This stage is hard-gated by an audited positive V18 formal textual-memory
headroom result.  Session notes are never placed in the query KV context.
Each observed fact contributes five position-specific rank-one key/value
factors and a memory intervention retains or removes those factors directly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_controlled_interference_v18 import FIXED_POLICIES, QUERY_KINDS
from tasks.cbf_qwen_lowrank_bridge import unit_key, unit_value
from tasks.evaluate_controlled_interference_v18_formal import clustered_interval
from tasks.evaluate_controlled_interference_v18_pilot import state_arms
from tasks.evaluate_mquake_qwen_outcome_v16 import (
    encode_candidates, load_jsonl, normalize_generation,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "v18_qwen_lowrank_fastmemory_bridge_v1"
FORMAL_PROTOCOL = "controlled_memory_interference_v18_qwen_formal"
DATA_PROTOCOL = "controlled_memory_interference_v18_formal"
SCALES = (4.0, 8.0, 16.0)
PRIMARY_SCALE = 8.0
UPDATE_KINDS = frozenset(("update_old_surface", "update_current_surface"))
RETAIN_KINDS = frozenset(("retain_1", "retain_2"))
JOINT_KINDS = UPDATE_KINDS | RETAIN_KINDS
BOOTSTRAP_SEED = 1831
BOOTSTRAP_DRAWS = 20_000
HEADER = "Use the session fast memory and answer with only the requested code."


class FastMemoryReadSession:
    """Fresh-KV Qwen read path with native adapters loaded but no native writes."""

    def __init__(self, model, cache=None):
        from cbf_ttt.event_writer import cache_for
        self.model = model
        self.cache = cache if cache is not None else cache_for(model)

    def clone(self):
        return FastMemoryReadSession(self.model, copy.deepcopy(self.cache))

    def _forward(self, ids):
        from cbf_ttt.event_writer import forward
        return forward(self.model, ids, self.cache)


def read_json(path):
    return json.loads(Path(path).read_text())


def data_audit_path(data_root):
    data_root = Path(data_root)
    paths = [path for path in (data_root / "audit.json", data_root.parent / "audit.json")
             if path.exists()]
    if len(paths) != 1:
        raise ValueError("V18 formal data audit location is missing or ambiguous")
    return paths[0]


def verify_release(data_root, formal_root):
    data_root, formal_root = Path(data_root), Path(formal_root)
    manifest, data_audit = (read_json(data_root / "manifest.json"),
                            read_json(data_audit_path(data_root)))
    summary_path, audit_path = formal_root / "summary.json", formal_root / "audit.json"
    summary, audit = read_json(summary_path), read_json(audit_path)
    checks = (
        manifest.get("protocol") == DATA_PROTOCOL,
        manifest.get("formal_sha256") == sha(data_root / "formal.jsonl"),
        manifest.get("formal_states_sha256") == sha(data_root / "formal_states.jsonl"),
        data_audit.get("audit_passed") is True,
        data_audit.get("manifest_sha256") == sha(data_root / "manifest.json"),
        summary.get("protocol") == FORMAL_PROTOCOL,
        summary.get("causal_headroom_supported") is True,
        summary.get("headroom_gate", {}).get("passed") is True,
        summary.get("terminal_status") == "formal_headroom_supported",
        audit.get("audit_passed") is True,
        audit.get("summary_sha256") == sha(summary_path),
        audit.get("recomputed_headroom_gate", {}).get("passed") is True,
        (formal_root / "runner_exit_code.txt").read_text().strip() == "0",
        (formal_root / "terminal_status.txt").read_text().strip()
            == "completed_controlled_interference_v18_qwen_formal",
    )
    if not all(checks):
        raise ValueError("low-rank bridge requires an audited positive V18 formal result")
    return {
        "formal_data_manifest_sha256": sha(data_root / "manifest.json"),
        "formal_data_audit_sha256": sha(data_audit_path(data_root)),
        "formal_data_sha256": sha(data_root / "formal.jsonl"),
        "formal_states_sha256": sha(data_root / "formal_states.jsonl"),
        "formal_summary_sha256": sha(summary_path),
        "formal_outcome_audit_sha256": sha(audit_path),
    }


def render_query(statement):
    return f"{HEADER}\n\nQuery: {statement}\nAnswer:\n"


def query_key_id(episode, kind):
    index = {"update_old_surface": 0, "update_current_surface": 0,
             "retain_1": 1, "retain_2": 2}.get(kind)
    return None if index is None else episode["keys"][index]


def build_factor_cache(model, tokenizer, episode):
    import torch

    key_cache, value_cache, sessions = {}, {}, {}
    for session in episode["sessions"]:
        key_id, value = session["key_id"], session["value"]
        if key_id not in key_cache:
            key_cache[key_id] = unit_key(
                model, tokenizer, f"{episode['entity']} :: {key_id}")
        if value not in value_cache:
            ids = tokenizer.encode(value, add_special_tokens=False)
            if len(ids) != 5:
                raise ValueError("V18 low-rank value must have five tokens")
            value_cache[value] = torch.stack([unit_value(model, token_id) for token_id in ids])
        sessions[session["session_id"]] = {
            "key_id": key_id, "key": key_cache[key_id], "value": value,
            "values": value_cache[value],
        }
    return {"keys": key_cache, "values": value_cache, "sessions": sessions}


def memory_residuals(model, tokenizer, episode, factor_cache, identifiers, kind):
    import torch

    key_id = query_key_id(episode, kind)
    if key_id is None:
        dimension = model.config.hidden_size
        return torch.zeros((5, dimension), dtype=torch.float32, device=model.device), []
    query_key = factor_cache["keys"][key_id]
    residual = torch.zeros_like(next(iter(factor_cache["values"].values())))
    contributions = []
    for identifier in identifiers:
        component = factor_cache["sessions"][identifier]
        similarity = float((query_key * component["key"]).sum())
        residual = residual + similarity * component["values"]
        contributions.append({"session_id": identifier, "key_id": component["key_id"],
                              "value": component["value"], "key_similarity": similarity})
    return residual, contributions


def _candidate_result(model, hidden, labels, residuals, scale):
    import torch
    import torch.nn.functional as F

    adjusted = hidden.float() + scale * residuals[:hidden.shape[1]].unsqueeze(0)
    logits = model.lm_head(adjusted.to(model.lm_head.weight.dtype)).float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels, reduction="mean")
    return float(loss), logits


def generate_primary(model, tokenizer, prefix_session, prefix_hidden, residuals,
                     max_new_tokens=13):
    import torch

    session = prefix_session.clone(); hidden = prefix_hidden; generated = []
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
    stop_reason = "max_new_tokens"
    for index in range(max_new_tokens):
        residual = residuals[index] if index < residuals.shape[0] else residuals.new_zeros(
            residuals.shape[1])
        adjusted = hidden[:, -1].float() + PRIMARY_SCALE * residual.unsqueeze(0)
        logits = model.lm_head(adjusted.to(model.lm_head.weight.dtype)).float()
        next_id = int(logits.argmax(-1).item()); generated.append(next_id)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if next_id in eos_ids:
            stop_reason = "eos"; break
        if "\n" in text:
            stop_reason = "newline"; break
        if index + 1 < max_new_tokens:
            hidden = session._forward([next_id])
    return {"ids": generated, "text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "tokens": len(generated), "stop_reason": stop_reason,
            "max_new_tokens": max_new_tokens}


def score_query(model, tokenizer, episode, identifiers, spec, factor_cache):
    import torch

    prompt = render_query(spec["statement"])
    prompt_ids, encoded = encode_candidates(tokenizer, prompt, spec["candidates"])
    if any(len(ids) != 5 for ids in encoded.values()):
        raise ValueError("low-rank bridge answer token invariant failed")
    residuals, contributions = memory_residuals(
        model, tokenizer, episode, factor_cache, identifiers, spec["kind"])
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start_allocated, start_reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    started = time.perf_counter()
    prefix_session = FastMemoryReadSession(model); prefix_hidden = prefix_session._forward(prompt_ids)
    prefix_cache_length = prefix_session.cache.get_seq_length()
    scores = {str(scale): {} for scale in SCALES}
    for answer, answer_ids in encoded.items():
        branch = prefix_session.clone(); tail = branch._forward(answer_ids[:-1])
        hidden = torch.cat((prefix_hidden[:, -1:], tail), dim=1)
        labels = torch.tensor(answer_ids, dtype=torch.long, device=model.device)
        for scale in SCALES:
            value, _ = _candidate_result(model, hidden, labels, residuals, scale)
            if not math.isfinite(value):
                raise RuntimeError("non-finite V18 low-rank answer NLL")
            scores[str(scale)][answer] = value
        del branch, tail, hidden, labels
    generation = generate_primary(model, tokenizer, prefix_session, prefix_hidden, residuals)
    generation.update({"exact": int(generation["text"] == spec["answer"]),
                       "normalized_exact": int(normalize_generation(generation["text"])
                                               == normalize_generation(spec["answer"]))})
    if prefix_session.cache.get_seq_length() != prefix_cache_length:
        raise RuntimeError("low-rank branch mutated the shared prompt cache")
    torch.cuda.synchronize()
    by_scale = {}
    for scale in SCALES:
        table = scores[str(scale)]
        prediction = min(spec["candidates"], key=lambda value: (table[value], value))
        by_scale[str(scale)] = {"scores": table, "prediction": prediction,
                                "correct": int(prediction == spec["answer"]),
                                "answer_nll": table[spec["answer"]]}
    return {"kind": spec["kind"], "statement": spec["statement"],
            "answer": spec["answer"], "candidates": spec["candidates"],
            "answer_token_lengths": {answer: len(ids) for answer, ids in encoded.items()},
            "by_scale": by_scale, "generation_primary": generation,
            "memory_session_ids": list(identifiers), "contributions": contributions,
            "residual_norms": [float(value) for value in residuals.norm(dim=1)],
            "prompt_tokens": len(prompt_ids), "total_tokens_max": len(prompt_ids) + 5,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "resources": {"seconds": time.perf_counter() - started,
                          "candidate_scores": len(encoded) * len(SCALES),
                          "gpu_start_allocated_gib": start_allocated / 2**30,
                          "gpu_start_reserved_gib": start_reserved / 2**30,
                          "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                          "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}}


def content_controls(model, tokenizer, episode, factor_cache):
    """Score one correct, wrong-value and empty component for each joint query."""
    output = []
    sessions = episode["sessions"]
    current_by_key = {}
    for session in sessions:
        current_by_key[session["key_id"]] = session
    wrong_values = {0: episode["primary_old_value"], 1: episode["current_values"][3],
                    2: episode["current_values"][4]}
    for spec in episode["queries"]:
        key_id = query_key_id(episode, spec["kind"])
        if key_id is None:
            continue
        key_index = episode["keys"].index(key_id)
        correct_session = current_by_key[key_id]
        correct_ids = [correct_session["session_id"]]
        correct = score_query(model, tokenizer, episode, correct_ids, spec, factor_cache)
        # Use the observed key with a deliberately wrong observed value direction.
        wrong_value = wrong_values[key_index]
        synthetic = copy.deepcopy(factor_cache)
        synthetic_id = f"wrong:{spec['kind']}"
        synthetic["sessions"][synthetic_id] = {
            "key_id": key_id, "key": factor_cache["keys"][key_id], "value": wrong_value,
            "values": factor_cache["values"][wrong_value]}
        wrong = score_query(model, tokenizer, episode, [synthetic_id], spec, synthetic)
        zero = score_query(model, tokenizer, episode, [], spec, factor_cache)
        output.append({"kind": spec["kind"], "correct": correct, "wrong": wrong, "zero": zero})
    return output


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    data_root, output = Path(args.data_root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_release(data_root, args.formal_root)
    episodes, states = load_jsonl(data_root / "formal.jsonl"), load_jsonl(
        data_root / "formal_states.jsonl")
    selected = [(episode, state) for index, (episode, state) in enumerate(zip(episodes, states))
                if index % args.num_shards == args.shard]
    torch.manual_seed(1823); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer_files = {path.name: sha(path) for path in sorted(Path(args.tokenizer).iterdir())
                       if path.is_file() and path.suffix in {
                           ".json", ".model", ".txt", ".tiktoken"}}
    started = time.perf_counter(); records = []; unique_states = reused = 0
    with output.open("x") as sink, torch.no_grad():
        for index, (episode, state) in enumerate(selected):
            factor_cache = build_factor_cache(model, tokenizer, episode)
            scored, arms = {}, []
            for arm in state_arms(state):
                key = tuple(arm["memory_session_ids"])
                if key not in scored:
                    queries = [score_query(model, tokenizer, episode, key, spec, factor_cache)
                               for spec in episode["queries"]]
                    scored[key] = {"source_arm_id": arm["arm_id"], "queries": queries}
                    unique_states += 1
                else:
                    reused += 1
                result = copy.deepcopy(arm)
                result["score_source_arm_id"] = scored[key]["source_arm_id"]
                result["queries"] = copy.deepcopy(scored[key]["queries"])
                arms.append(result)
            row = {key: episode[key] for key in (
                "episode_id", "row_index", "subject", "subject_index", "template_group",
                "template_group_index", "fold", "length", "mechanism", "target_key_id",
                "current_session_id")}
            row["arms"], row["content_controls"] = arms, content_controls(
                model, tokenizer, episode, factor_cache)
            sink.write(json.dumps(row) + "\n"); sink.flush(); records.append(row)
            print(json.dumps({"completed": index + 1, "total": len(selected),
                              "episode_id": episode["episode_id"],
                              "unique_states": len(scored)}), flush=True)
    unique_queries = [query for row in records for arm in row["arms"]
                      if arm["score_source_arm_id"] == arm["arm_id"] for query in arm["queries"]]
    control_queries = [query for row in records for control in row["content_controls"]
                       for name in ("correct", "wrong", "zero") for query in [control[name]]]
    measurements = unique_queries + control_queries
    audit = {"protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
             "rows": len(records), "rows_sha256": sha(output), "dependencies": dependencies,
             "seconds": time.perf_counter() - started,
             "unique_memory_states_scored": unique_states,
             "reused_arm_states": reused, "query_measurements": len(measurements),
             "content_control_measurements": len(control_queries),
             "gpu_peak_allocated_gib": max(q["resources"]["gpu_peak_allocated_gib"]
                                           for q in measurements),
             "gpu_peak_reserved_gib": max(q["resources"]["gpu_peak_reserved_gib"]
                                          for q in measurements),
             "model": args.model, "tokenizer": args.tokenizer,
             "gpu": torch.cuda.get_device_name(), "loading_info": model_meta["loading_info"],
             "model_provenance": {key: model_meta[key] for key in (
                 "path", "class", "config_sha256", "attention", "parameter_count",
                 "source_files_sha256", "excluded_adapters")}
                 | {"ttt_mode": model_meta["config"]["ttt_mode"]},
             "tokenizer_files_sha256": tokenizer_files,
             "representation": "five_position_qwen_rank_one_key_value_components",
             "scales": list(SCALES), "primary_scale": PRIMARY_SCALE,
             "fresh_kv": True, "backbone_frozen": True,
             "torch": torch.__version__, "transformers": __import__("transformers").__version__,
             "external_sets_loaded": False}
    write_json(str(output) + ".audit.json", audit); return audit


def qmap(arm):
    result = {query["kind"]: query for query in arm["queries"]}
    if set(result) != set(QUERY_KINDS):
        raise ValueError("incomplete bridge query kinds")
    return result


def arm_metrics(arm, scale):
    queries = qmap(arm); key = str(float(scale))
    return {"joint_success": int(all(queries[kind]["by_scale"][key]["correct"]
                                     for kind in JOINT_KINDS)),
            "update_accuracy": statistics.fmean(
                queries[kind]["by_scale"][key]["correct"] for kind in UPDATE_KINDS),
            "retained_key_accuracy": statistics.fmean(
                queries[kind]["by_scale"][key]["correct"] for kind in RETAIN_KINDS),
            "joint_answer_nll": statistics.fmean(
                queries[kind]["by_scale"][key]["answer_nll"] for kind in JOINT_KINDS),
            "locality_accuracy": queries["locality"]["by_scale"][key]["correct"],
            "generation_joint_success": int(all(
                queries[kind]["generation_primary"]["normalized_exact"]
                for kind in JOINT_KINDS)) if scale == PRIMARY_SCALE else None}


def select_fixed(rows, scale):
    table = {}
    for policy in FIXED_POLICIES:
        values = [arm_metrics({arm["arm_id"]: arm for arm in row["arms"]}[
            f"fixed:{policy}"], scale) for row in rows]
        table[policy] = {key: statistics.fmean(value[key] for value in values)
                         for key in ("joint_success", "update_accuracy",
                                     "retained_key_accuracy", "joint_answer_nll",
                                     "locality_accuracy")}
    winner = max(FIXED_POLICIES, key=lambda policy: (
        table[policy]["joint_success"], table[policy]["update_accuracy"],
        table[policy]["retained_key_accuracy"], -table[policy]["joint_answer_nll"],
        -FIXED_POLICIES.index(policy)))
    return winner, table


def select_oracle(row, scale):
    candidates = [next(arm for arm in row["arms"] if arm["arm_id"] == "fixed:retain_all")] + [
        arm for arm in row["arms"] if arm["kind"] in {"delete_one", "delete_key"}]
    ranked = []
    for tie_index, arm in enumerate(candidates):
        value = arm_metrics(arm, scale)
        ranked.append((value["joint_success"], value["update_accuracy"],
                       value["retained_key_accuracy"], -value["joint_answer_nll"],
                       -tie_index, arm, value))
    winner = max(ranked, key=lambda value: value[:5])
    return winner[5], winner[6]


def aggregate_scale(rows, scale):
    selected, evidence = {}, {}
    for fold in range(5):
        selected[str(fold)], evidence[str(fold)] = select_fixed(
            [row for row in rows if row["fold"] != fold], scale)
    records, selections = [], []
    action_counts = {"retain_all": 0, "delete_one": 0, "delete_key": 0}
    for row in rows:
        oracle_arm, oracle = select_oracle(row, scale)
        policy = selected[str(row["fold"])]
        fixed = arm_metrics(next(arm for arm in row["arms"]
                                 if arm["arm_id"] == f"fixed:{policy}"), scale)
        action = "retain_all" if oracle_arm["arm_id"] == "fixed:retain_all" \
            else oracle_arm["kind"]
        action_counts[action] += 1
        records.append({"subject": row["subject"], "template_group": row["template_group"],
                        "fold": row["fold"], "length": row["length"],
                        "mechanism": row["mechanism"], "joint_gain": (oracle["joint_success"]
                                                             - fixed["joint_success"]),
                        "oracle": oracle, "fixed": fixed})
        selections.append({"episode_id": row["episode_id"], "oracle_arm_id": oracle_arm["arm_id"],
                           "fixed_policy": policy})
    subject = clustered_interval(records, "joint_gain", "subject",
                                 BOOTSTRAP_SEED + int(scale), BOOTSTRAP_DRAWS)
    template = clustered_interval(records, "joint_gain", "template_group",
                                  BOOTSTRAP_SEED + 100 + int(scale), BOOTSTRAP_DRAWS)
    mean = lambda key, which: statistics.fmean(row[which][key] for row in records)
    by_fold = {str(fold): statistics.fmean(row["joint_gain"] for row in records
                                           if row["fold"] == fold) for fold in range(5)}
    oracle_table = {key: mean(key, "oracle") for key in (
        "joint_success", "update_accuracy", "retained_key_accuracy",
        "joint_answer_nll", "locality_accuracy")}
    fixed_table = {key: mean(key, "fixed") for key in (
        "joint_success", "update_accuracy", "retained_key_accuracy",
        "joint_answer_nll", "locality_accuracy")}
    if scale == PRIMARY_SCALE:
        oracle_table["generation_joint_success"] = mean("generation_joint_success", "oracle")
        fixed_table["generation_joint_success"] = mean("generation_joint_success", "fixed")
    by_mechanism = {name: {"rows": sum(row["mechanism"] == name for row in records),
                           "joint_gain": statistics.fmean(
                               row["joint_gain"] for row in records
                               if row["mechanism"] == name)}
                    for name in sorted({row["mechanism"] for row in records})}
    by_length = {str(length): {"rows": sum(row["length"] == length for row in records),
                               "joint_gain": statistics.fmean(
                                   row["joint_gain"] for row in records
                                   if row["length"] == length)}
                 for length in sorted({row["length"] for row in records})}
    return {"scale": scale, "fixed_selected_by_outer_fold": selected,
            "fixed_selection_evidence": evidence, "oracle_selection": selections,
            "oracle_action_counts": action_counts,
            "oracle": oracle_table, "outer_fold_fixed": fixed_table,
            "joint_gain_micro": statistics.fmean(row["joint_gain"] for row in records),
            "joint_gain_subject_macro": subject, "joint_gain_template_macro": template,
            "by_fold_gain": by_fold, "by_mechanism": by_mechanism,
            "by_length": by_length}


def content_table(rows, scale=PRIMARY_SCALE):
    key = str(float(scale)); output = {}
    for policy in ("correct", "wrong", "zero"):
        queries = [control[policy] for row in rows for control in row["content_controls"]]
        output[policy] = {"accuracy": statistics.fmean(
            query["by_scale"][key]["correct"] for query in queries),
                          "answer_nll": statistics.fmean(
            query["by_scale"][key]["answer_nll"] for query in queries)}
    return output


def aggregate(rows):
    scales = {str(scale): aggregate_scale(rows, scale) for scale in SCALES}
    primary, content = scales[str(PRIMARY_SCALE)], content_table(rows)
    checks = {
        "correct_component_accuracy_at_least_90": content["correct"]["accuracy"] >= .90,
        "wrong_component_accuracy_at_most_20": content["wrong"]["accuracy"] <= .20,
        "oracle_update_accuracy_at_least_90": primary["oracle"]["update_accuracy"] >= .90,
        "oracle_retained_key_accuracy_at_least_90":
            primary["oracle"]["retained_key_accuracy"] >= .90,
        "oracle_locality_accuracy_at_least_97": primary["oracle"]["locality_accuracy"] >= .97,
        "oracle_joint_gain_at_least_10_points": primary["joint_gain_micro"] >= .10,
        "subject_cluster_ci_lower_above_zero":
            primary["joint_gain_subject_macro"]["lower_95"] > 0,
        "template_cluster_ci_lower_above_zero":
            primary["joint_gain_template_macro"]["lower_95"] > 0,
        "positive_joint_gain_at_scales_4_8_16":
            all(scales[str(scale)]["joint_gain_micro"] > 0 for scale in SCALES),
    }
    supported = all(checks.values())
    return {"scales": scales, "primary_scale": PRIMARY_SCALE,
            "content_binding": content, "bridge_gate": {"checks": checks, "passed": supported},
            "fastmemory_bridge_supported": supported,
            "next_stage": "evaluate_physical_conflict_rules" if supported
                else "stop_fastmemory_controller_claim"}


def summarize(args):
    dependencies = verify_release(args.data_root, args.formal_root)
    rows, resources, hashes = [], [], {}
    for value in args.inputs:
        path = Path(value); part = load_jsonl(path); resource = read_json(str(path) + ".audit.json")
        if resource.get("dependencies") != dependencies or resource.get("rows_sha256") != sha(path):
            raise ValueError("bridge shard dependency mismatch")
        rows.extend(part); resources.append(resource); hashes[path.name] = sha(path)
    if len(rows) != 320 or len({row["episode_id"] for row in rows}) != 320:
        raise ValueError("bridge formal coverage mismatch")
    table = aggregate(rows)
    measurements = [query for row in rows for arm in row["arms"]
                    if arm["score_source_arm_id"] == arm["arm_id"]
                    for query in arm["queries"]]
    measurements.extend(query for row in rows for control in row["content_controls"]
                        for name in ("correct", "wrong", "zero")
                        for query in [control[name]])
    latencies = sorted(query["resources"]["seconds"] for query in measurements)
    result = {"protocol": PROTOCOL, **table, "rows": len(rows),
              "raw_rows_sha256": hashes, "shard_resources": resources,
              "dependencies": dependencies,
              "resources": {
                  "unique_memory_states": sum(row["unique_memory_states_scored"]
                                              for row in resources),
                  "query_measurements": len(measurements),
                  "wall_seconds_slower_shard": max(row["seconds"] for row in resources),
                  "mean_seconds_per_query": statistics.fmean(latencies),
                  "median_seconds_per_query": statistics.median(latencies),
                  "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
                  "max_seconds_per_query": max(latencies),
                  "max_gpu_peak_allocated_gib": max(
                      query["resources"]["gpu_peak_allocated_gib"] for query in measurements),
                  "max_gpu_peak_reserved_gib": max(
                      query["resources"]["gpu_peak_reserved_gib"] for query in measurements),
              },
              "model_scored": True, "formal_scored": True,
              "external_sets_loaded": False,
              "terminal_status": "fastmemory_bridge_supported" if table["bridge_gate"]["passed"]
                  else "fastmemory_bridge_unsupported"}
    write_json(args.output, result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--formal-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--data-root", required=True)
    summary_parser.add_argument("--formal-root", required=True)
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args) if args.command == "collect" else summarize(args), indent=2))


if __name__ == "__main__":
    main()
