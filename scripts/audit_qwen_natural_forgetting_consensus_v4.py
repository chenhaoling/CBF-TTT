"""Independent trajectory audit of V4 cross-fit online calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.build_qwen_natural_forgetting_utility_pilot_v1 import scope_examples
from tasks.evaluate_mquake_qwen_outcome_v16 import encode_candidates
from tasks.evaluate_qwen_natural_causal_memory_v1 import (
    active_queries, chain_map, checkpoint_label, primary_count, record_for,
)
from tasks.evaluate_qwen_natural_forgetting_consensus_v4 import (
    ARMS, CANDIDATES, FoldController, PLAN, PROTOCOL, aggregate, candidate_action,
)
from tasks.evaluate_qwen_natural_forgetting_online_v3 import engineered_features
from tasks.evaluate_qwen_natural_scope_v5_development import load_v5_gate
from tasks.evaluate_qwen_natural_value_bridge_v1 import render_query
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import fold_for
from tasks.train_qwen_natural_scope_transfer_v2 import build_scorer


def close(a, b, tolerance=1e-6):
    return abs(float(a) - float(b)) <= tolerance


def audit(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root, repo = Path(args.root), Path(args.repo)
    crossfit_root, data_root = Path(args.crossfit_root), Path(args.data_root)
    v5_root, context, scope = map(Path, (args.v5_root, args.context_root, args.scope_root))
    summary = json.loads((root / "summary.json").read_text())
    rows=[]; traces=[]; resources=[]
    for shard in range(args.num_shards):
        rp=root/f"rows_shard{shard}.jsonl"; dp=root/f"decisions_shard{shard}.jsonl"
        resource=json.loads((root/f"resource_shard{shard}.json").read_text())
        rows.extend(json.loads(line) for line in rp.read_text().splitlines() if line)
        traces.extend(json.loads(line) for line in dp.read_text().splitlines() if line)
        resources.append(resource)
    rows.sort(key=lambda row:(row["session_id"],row["phase"],row["chain_id"],row["surface"]))
    traces.sort(key=lambda row:row["session_id"]); traces_by_id={row["session_id"]:row for row in traces}
    _, sessions=load(data_root,"calibration"); sessions_by_id={row["id"]:row for row in sessions}
    artifact=torch.load(crossfit_root/"result"/"crossfit_factor_gate.pt",
                        map_location="cpu",weights_only=False)
    controllers={outer:FoldController(artifact,outer) for outer in range(5)}
    model,_=load_model(args.model,"step81381_native")
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,use_fast=True)
    source_summary=json.loads((context/"summary.json").read_text())
    transforms=torch.load(context/"transforms.pt",map_location="cpu",weights_only=False)
    scope_artifact=torch.load(scope/"scope_gates.pt",map_location="cpu",weights_only=False)
    scorers={}; scorer_meta={}
    for shard in range(args.num_shards):
        selected=[row for index,row in enumerate(sessions) if index%args.num_shards==shard]
        scorers[shard],scorer_meta[shard]=build_scorer(scope_examples(selected),model,tokenizer,
            transforms["transforms"],source_summary["selected_config"]["regularizer"],
            scope_artifact,args.batch_size)
    v5_path=v5_root/"result"/"constrained_scope_gate.pt"
    v5_gate,_,_=load_v5_gate(v5_path,scope_artifact["heads"]["7301"])

    feature_checks=all(scorer_meta[i]==resources[i]["feature_meta"]
                       for i in range(args.num_shards))
    probability_checks=True; transition_checks=True; expected_identity=[]
    for session_index,session in enumerate(sessions):
        scorer=scorers[session_index%args.num_shards]; trace=traces_by_id[session["id"]]
        controller=controllers[fold_for(session["id"])]
        chains=chain_map(session); states={arm:[] for arm in ARMS}; active={}
        transition_checks &= trace["outer_fold"]==fold_for(session["id"]) \
            and set(trace["decisions"])==set(ARMS)
        for step,event in enumerate(session["events"]):
            chain=chains[event["chain_id"]]; version=chain["versions"][event["version"]]
            for arm in ARMS:
                observed=trace["decisions"][arm][step]; memory=states[arm]
                features,top,v5p=engineered_features(version["prompt"],memory,step,
                    len(session["events"]),scorer,v5_gate)
                feature_checks &= observed["top"]==top and close(observed["v5_probability"],v5p) \
                    and len(features)==len(observed["features"]) \
                    and all(close(a,b) for a,b in zip(features,observed["features"]))
                e=n={}; expected=False
                if arm=="always_scope_top1": expected=bool(memory)
                elif arm=="source_action_rule": expected=bool(memory) and event["action"]=="local"
                elif arm in CANDIDATES:
                    e=controller.probabilities(observed["features"],"engineered")
                    n=controller.probabilities(observed["features"],"no_schedule")
                    expected=bool(memory) and candidate_action(arm,e,n)
                elif arm!="retain_all": transition_checks=False
                stored_e={int(k):tuple(v) for k,v in observed["engineered_probabilities"].items()}
                stored_n={int(k):tuple(v) for k,v in observed["no_schedule_probabilities"].items()}
                probability_checks &= set(stored_e)==set(e) and set(stored_n)==set(n) \
                    and all(close(stored_e[k][j],e[k][j]) for k in e for j in (0,1)) \
                    and all(close(stored_n[k][j],n[k][j]) for k in n for j in (0,1))
                action="scope_top1_a0" if expected else "retain_all"
                transition_checks &= observed["action"]==action \
                    and observed["before_size"]==len(memory) \
                    and observed["after_delete_size"]==len(memory)-int(expected) \
                    and observed["after_write_size"]==len(memory)+1-int(expected)
                if expected:
                    transition_checks &= observed["deleted"]==[memory[top]]
                    memory=[row for index,row in enumerate(memory) if index!=top]
                else: transition_checks &= observed["deleted"]==[]
                states[arm]=[*memory,{**record_for(chain,version),"strength":1.0}]
            active[event["chain_id"]]=event["version"]
            phase=checkpoint_label(step,session)
            if phase is not None:
                final=phase==f"update_{primary_count(session)}"
                expected_identity.extend((session["id"],phase,query["chain_id"],
                    query["active_version"],query["surface"],query["query"],query["answer"],
                    final,event["chain_id"],fold_for(session["id"]))
                    for query in active_queries(session,active))
    observed_identity=[(row["session_id"],row["phase"],row["chain_id"],
        row["active_version"],row["surface"],row["query"],row["answer"],row["final"],
        row["recent_chain_id"],row["outer_fold"]) for row in rows]
    expected_identity.sort(); observed_identity.sort()
    row_checks=len(rows)==len(expected_identity)==5144 and observed_identity==expected_identity
    score_checks=True
    for row in rows:
        candidates=sessions_by_id[row["session_id"]]["candidates"]
        for arm in ARMS:
            value=row["arms"][arm]
            score_checks &= value["correct"]==int(value["prediction"]==row["answer"])
            if row["final"]:
                _,encoded=encode_candidates(tokenizer,render_query(row["query"]),candidates)
                score_checks &= value["generation"]["token_exact"]==int(
                    value["generation"]["ids"]==encoded[row["answer"]])
            else: score_checks &= "generation" not in value
    recomputed=aggregate(rows,traces)
    frozen={"crossfit_checkpoint":sha(crossfit_root/"result"/"crossfit_factor_gate.pt"),
        "crossfit_summary":sha(crossfit_root/"result"/"summary.json"),
        "crossfit_audit":sha(crossfit_root/"execution_audit.json"),
        "data_manifest":sha(data_root/"manifest.json"),"v5_checkpoint":sha(v5_path),
        "context_transforms":sha(context/"transforms.pt"),
        "scope_artifact":sha(scope/"scope_gates.pt")}
    hash_checks=all(resource["frozen_inputs"]==frozen and
        resource["rows_sha256"]==sha(root/f"rows_shard{index}.jsonl") and
        resource["decisions_sha256"]==sha(root/f"decisions_shard{index}.jsonl")
        for index,resource in enumerate(resources)) \
        and summary["plan_sha256"]==sha(repo/PLAN) \
        and all(summary["rows_sha256"][p.name]==sha(p) for p in root.glob("rows_shard*.jsonl")) \
        and all(summary["decisions_sha256"][p.name]==sha(p)
                for p in root.glob("decisions_shard*.jsonl"))
    terminal=("completed_qwen_natural_forgetting_consensus_v4_test_release"
              if recomputed["gate"]["test_authorized"] else
              "completed_qwen_natural_forgetting_consensus_v4_calibration_stop")
    checks={"terminal_status_and_exit":(root/"terminal_status.txt").read_text().strip()==terminal
            and (root/"runner_exit_code.txt").read_text().strip()=="0",
        "complete_crossfit_calibration_rows":row_checks,
        "state_transition_reconstruction":transition_checks,
        "feature_reconstruction":feature_checks,
        "checkpoint_probability_and_action_replay":probability_checks,
        "score_and_generation_consistency":score_checks,
        "artifact_and_frozen_input_hashes":hash_checks,
        "independent_metrics_search_bootstrap_and_gate":all(
            recomputed[key]==summary[key] for key in recomputed),
        "heldout_seals":summary["calibration_scored"] and not summary["development_loaded"]
            and not summary["test_loaded"] and not summary["external_benchmarks_loaded"]}
    return {"protocol":PROTOCOL+"_audit","checks":checks,"passed":all(checks.values()),
            "recomputed":recomputed}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("root","repo","crossfit_root","data_root","v5_root","model","tokenizer","context_root","scope_root","output"):
        parser.add_argument(f"--{name.replace('_','-')}",required=True)
    parser.add_argument("--num-shards",type=int,default=2); parser.add_argument("--batch-size",type=int,default=128)
    args=parser.parse_args(); result=audit(args); write_json(Path(args.output),result)
    print(json.dumps(result,indent=2)); raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__": main()
