"""Train subject-disjoint same-entity heads and combine with relation scope."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_counterfact_zero_shot_forgetting import SOURCE_SHA256
from tasks.build_scoped_delta_corpus import MAX_KEYS
from tasks.evaluate_automatic_factorized_scope_v6 import (
    build_fold_views, fold_scores, load_v5, relation_features, source_documents,
)
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossdomain_forgetting_reranker import FOLDS, deterministic_folds
from tasks.train_scoped_delta_ranker_v4 import load_corpus
from tasks.train_text_only_scope_calibrator_v8 import load_trace
from tasks.train_zsre_forgetting_controller import lexical_features
from tasks.train_zsre_scope_resolver import extract_pair_features
from tasks.train_relation_supervised_scope_v5 import load_v4


PROTOCOL = "learned_entity_scope_v9"
SEEDS = (17101, 17102, 17103)
EPOCHS = 10
ENTITY_TEMPLATE = "Text A: {left}\nText B: {right}\nSame entity:"


def load_raw(path):
    path = Path(path)
    if sha(path) != SOURCE_SHA256: raise ValueError("unexpected CounterFact source hash")
    return json.loads(path.read_text())


def entity_examples(rows):
    by_relation = {}
    for row in rows:
        rewrite = row["requested_rewrite"]
        by_relation.setdefault(rewrite["relation_id"], []).append(row)
    for values in by_relation.values():
        values.sort(key=lambda row: hashlib.sha256(
            str(row["case_id"]).encode()).hexdigest())
    output = []
    for row in rows:
        rewrite = row["requested_rewrite"]; subject = rewrite["subject"]
        clean = rewrite["prompt"].format(subject)
        candidates = by_relation[rewrite["relation_id"]]
        start = candidates.index(row); negative = None
        for offset in range(1, len(candidates)):
            candidate = candidates[(start + offset) % len(candidates)]
            if candidate["requested_rewrite"]["subject"] != subject:
                negative = candidate; break
        if negative is None: raise ValueError("relation lacks a negative subject")
        negative_rewrite = negative["requested_rewrite"]
        negative_clean = negative_rewrite["prompt"].format(negative_rewrite["subject"])
        for paraphrase in row["paraphrase_prompts"]:
            output.append({"left": paraphrase, "right": clean, "left_subject": subject,
                           "right_subject": subject, "label": 1})
            output.append({"left": paraphrase, "right": negative_clean,
                           "left_subject": subject,
                           "right_subject": negative_rewrite["subject"], "label": 0})
    return output


def pair_text(left, right): return ENTITY_TEMPLATE.format(left=left, right=right)


def pair_feature(left, right, table):
    import torch
    return torch.cat((table[pair_text(left, right)], torch.tensor(lexical_features(left, right))))


class EntityHead:
    def __init__(self, dim, seed):
        import torch
        torch.manual_seed(seed); self.linear = torch.nn.Linear(dim, 1)
        torch.nn.init.normal_(self.linear.weight, std=.001); torch.nn.init.zeros_(self.linear.bias)
    def parameters(self): return self.linear.parameters()
    def state_dict(self): return self.linear.state_dict()
    def load_state_dict(self, state): self.linear.load_state_dict(state)
    def __call__(self, values): return self.linear(values).squeeze(-1)


def fit_heads(examples, table, excluded, seed_offset):
    import torch
    rows = [row for row in examples if row["left_subject"] not in excluded
            and row["right_subject"] not in excluded]
    x = torch.stack([pair_feature(row["left"], row["right"], table) for row in rows])
    y = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
    mean=x.mean(0); std=x.std(0,unbiased=False).clamp_min(1e-4)
    device=torch.device("cuda"); train_x=((x-mean)/std).to(device); train_y=y.to(device)
    artifacts=[]
    for seed in SEEDS:
        head=EntityHead(x.shape[-1],seed+seed_offset); head.linear.to(device)
        optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.05)
        for _ in range(EPOCHS):
            optimizer.zero_grad(set_to_none=True); logits=head(train_x)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,train_y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.); optimizer.step()
        head.linear.cpu(); artifacts.append({"state":copy.deepcopy(head.state_dict()),
            "mean":mean,"std":std,"dim":x.shape[-1],"seed":seed+seed_offset})
    return artifacts, len(rows)


def entity_logits(values, artifacts):
    import torch
    output=[]
    for artifact in artifacts:
        head=EntityHead(artifact["dim"],artifact["seed"]); head.load_state_dict(artifact["state"])
        with torch.no_grad(): output.append(head((values-artifact["mean"])/artifact["std"]))
    return torch.stack(output).mean(0)


def auc(labels, scores):
    positives=[s for y,s in zip(labels,scores) if y]; negatives=[s for y,s in zip(labels,scores) if not y]
    wins=sum(p>n for p in positives for n in negatives)+.5*sum(p==n for p in positives for n in negatives)
    return wins/(len(positives)*len(negatives))


def evaluate_fold(fold_rows, relation_rows, pair_table, artifacts, fold):
    import torch
    output=[]; entity_labels=[]; entity_scores=[]
    for view_row, relation_row in zip(fold_rows, relation_rows):
        group=view_row["group"]
        values=torch.stack([pair_feature(group["query"],candidate["prompt"],pair_table)
                            for candidate in group["candidates"]])
        logits=entity_logits(values,artifacts); entity=logits.sigmoid()
        relation=relation_row["relation"]
        scores=entity.clamp_min(1e-12).log()+relation.clamp_min(1e-12).log()
        target=group["target"]; prediction=int(scores.argmax()); order=scores.argsort(descending=True).tolist()
        output.append({"session_id":group["session_id"],"group_id":group["group_id"],
            "target":target,"prediction":prediction,"correct":int(prediction==target),
            "reciprocal_rank":1/(order.index(target)+1),"fold":fold})
        for candidate,score in zip(group["candidates"],entity.tolist()):
            entity_labels.append(candidate["subject"].casefold()==group["query_subject"].casefold())
            entity_scores.append(score)
    return output,entity_labels,entity_scores


def aggregate(rows,labels,scores):
    return {"events":len(rows),"top1_accuracy":statistics.fmean(r["correct"] for r in rows),
            "mrr":statistics.fmean(r["reciprocal_rank"] for r in rows),
            "entity_pair_auc":auc(labels,scores)}


def compare(rows,baseline,seed):
    d=[r["correct"]-baseline[(r["session_id"],r["group_id"])]["correct"] for r in rows]
    return {"top1_gain":statistics.fmean(d),"bootstrap_95ci":_bootstrap(d,seed),"units":len(d)}


def development_gate(result,vs_v8,vs_v4):
    checks={"top1_at_least_96":result["top1_accuracy"]>=.96,"mrr_at_least_98":result["mrr"]>=.98,
      "entity_auc_at_least_97":result["entity_pair_auc"]>=.97,
      "gain_v8_at_least_1pp":vs_v8["top1_gain"]>=.01,"gain_v8_ci_positive":vs_v8["bootstrap_95ci"][0]>0,
      "gain_v4_at_least_12pp":vs_v4["top1_gain"]>=.12,"gain_v4_ci_positive":vs_v4["bootstrap_95ci"][0]>0}
    return {"checks":checks,"passed":all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    corpus,_=load_corpus(args.corpus_root); groups=[g for g in corpus if g["source"]=="counterfact" and g["target"]<MAX_KEYS]
    fold_map=deterministic_folds({g["session_id"] for g in groups}); raw_rows=load_raw(args.counterfact_source)
    examples=entity_examples(raw_rows); views,_=build_fold_views(groups,fold_map,source_documents(args.counterfact_source))
    relation_texts=sorted({c[f] for rows in views.values() for row in rows for c in row["candidates"]
                           for f in ("query_relation","key_relation")})
    pair_names={pair_text(e["left"],e["right"]) for e in examples}
    pair_names.update(pair_text(g["query"],c["prompt"]) for g in groups for c in g["candidates"])
    pair_names=sorted(pair_names); v5_artifact,_=load_v5(args.v5_root)
    source=Path(args.source_root); source_summary=json.loads((source/"summary.json").read_text())
    transforms=torch.load(source/"transforms.pt",map_location="cpu",weights_only=False)
    torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    model,model_meta=load_model(args.model,"step81381_native"); tokenizer=AutoTokenizer.from_pretrained(args.tokenizer)
    raw=extract(model,tokenizer,{"entity":relation_texts,"value":["unknown"]},args.batch_size)
    embeddings,embedding_hashes=apply(raw,{"entity":relation_texts,"value":["unknown"]},transforms["transforms"],
                                      source_summary["selected_config"]["regularizer"])
    rel_features=relation_features(relation_texts,embeddings)
    pair_table=extract_pair_features(model,tokenizer,pair_names,args.pair_batch_size)
    del raw,model; torch.cuda.empty_cache()
    final=[]; all_labels=[]; all_scores=[]; checkpoints={}; train_counts={}
    for fold in range(FOLDS):
        fold_groups=[g for g in groups if fold_map[g["session_id"]]==fold]
        held={g["query_subject"] for g in fold_groups}
        artifacts,count=fit_heads(examples,pair_table,held,fold*1000); checkpoints[str(fold)]=artifacts; train_counts[str(fold)]=count
        relation_rows=fold_scores(views[str(fold)],rel_features,v5_artifact["artifacts"][str(fold)])
        rows,labels,scores=evaluate_fold(views[str(fold)],relation_rows,pair_table,artifacts,fold)
        final.extend(rows); all_labels.extend(labels); all_scores.extend(scores)
    final.sort(key=lambda r:(r["session_id"],r["group_id"])); path=root/"counterfact_oof_learned_entity.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in final)); result=aggregate(final,all_labels,all_scores)
    v8=load_trace(args.v8_root,"counterfact_oof_text_calibrator.jsonl"); v4=load_v4(args.v4_root)
    vs_v8=compare(final,v8,17201); vs_v4=compare(final,v4,17211); gate=development_gate(result,vs_v8,vs_v4)
    checkpoint=root/"entity_heads.pt"; torch.save(checkpoints,checkpoint); seconds=time.perf_counter()-started
    summary={"protocol":PROTOCOL,"data_status":"observed_development_only","counterfact_oof":result,
      "v9_vs_v8":vs_v8,"v9_vs_v4":vs_v4,"development_gate":gate,
      "next_stage":"causal_automatic_scope_forgetting" if gate["passed"] else "revise_entity_encoder",
      "terminal_status":"completed_learned_entity_scope_v9"}
    training={"seconds":seconds,"gpu_peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
      "gpu_peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,"entity_examples":len(examples),
      "unique_entity_pairs":len(pair_names),"unique_relation_texts":len(relation_texts),
      "entity_training_rows":train_counts,"embedding_hashes":embedding_hashes,
      "raw_rows_sha256":{path.name:sha(path)},"checkpoint_sha256":sha(checkpoint)}
    manifest={"protocol":PROTOCOL,"counterfact_source_sha256":sha(args.counterfact_source),
      "corpus_manifest_sha256":sha(Path(args.corpus_root)/"manifest.json"),
      "v5_summary_sha256":sha(Path(args.v5_root)/"summary.json"),"model":args.model,"tokenizer":args.tokenizer,
      "loading_info":model_meta["loading_info"],"seeds":list(SEEDS),"epochs":EPOCHS,
      "inference_subject_metadata_used":False,"inference_relation_ids_used":False,
      "candidate_age_used":False,"counterfact_status":"observed_development_only",
      "longmemeval_scored":False,"locomo_scored":False}
    write_json(root/"summary.json",summary);write_json(root/"training.json",training);write_json(root/"training_manifest.json",manifest)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("corpus-root","counterfact-source","v4-root","v5-root","v8-root","source-root","output","model","tokenizer"):
        p.add_argument(f"--{name}",required=True)
    p.add_argument("--batch-size",type=int,default=256);p.add_argument("--pair-batch-size",type=int,default=128)
    print(json.dumps(run(p.parse_args()),indent=2))


if __name__=="__main__":main()

