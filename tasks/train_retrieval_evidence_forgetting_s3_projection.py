"""Train and select the S3 supervised-contrastive relation projection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from tasks.extract_retrieval_evidence_forgetting_s3_embeddings import PROTOCOL as EMBED_PROTOCOL
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_retrieval_evidence_forgetting_s3_projection_v1"
DATA_PROTOCOL = "qwen_retrieval_evidence_forgetting_s3"
AUDIT_PROTOCOL = "qwen_retrieval_evidence_forgetting_s3_data_audit_v1"
SEED = 35117
DIMENSIONS = (256, 512)
TEMPERATURES = (0.05, 0.10)
MARGINS = (0.10, 0.20)
CHECKPOINT_EPOCHS = (10, 25, 50)
SUPERVISED_STEPS_PER_EPOCH = 64
UNSUPERVISED_EPOCHS = 10
RELATIONS_PER_BATCH = 16
MENTIONS_PER_RELATION = 4
HARD_NEGATIVE_BATCH = 64
MARGIN_WEIGHT = 0.25


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def load_embedding_set(root: str | Path, dataset: str, expected_rows: int):
    import numpy as np

    root = Path(root)
    values, audits = [], []
    for shard in range(2):
        path = root / f"{dataset}_shard_{shard}.npz"
        audit_path = root / f"{dataset}_shard_{shard}.audit.json"
        audit = json.loads(audit_path.read_text())
        if (audit.get("protocol") != EMBED_PROTOCOL
                or audit.get("dataset") != dataset
                or audit.get("shard") != shard
                or audit.get("num_shards") != 2
                or audit.get("truncation") is not False
                or audit.get("dialogre_dev_loaded") is not False
                or audit.get("dialogre_test_loaded") is not False
                or sha(path) != audit.get("output_sha256")):
            raise ValueError(f"invalid S3 embedding shard: {dataset}/{shard}")
        with np.load(path) as arrays:
            values.extend(zip(
                arrays["indices"].tolist(),
                arrays["source_ids"].tolist(),
                arrays["relations"].tolist(),
                arrays["sources"].tolist(),
                arrays["dialogue_ids"].tolist(),
                arrays["type_pairs"].tolist(),
                arrays["embeddings"].astype("float32"),
            ))
        audits.append(audit)
    values.sort(key=lambda row: row[0])
    if len(values) != expected_rows or [row[0] for row in values] != list(range(expected_rows)):
        raise ValueError(f"incomplete S3 embedding dataset: {dataset}")
    return {
        "ids": [row[1] for row in values],
        "relations": [row[2] for row in values],
        "sources": [row[3] for row in values],
        "dialogues": [row[4] for row in values],
        "types": [row[5] for row in values],
        "matrix": np.stack([row[6] for row in values]),
    }, audits


def relation_batches(relations: list[str], generator, steps: int = SUPERVISED_STEPS_PER_EPOCH,
                     relations_per_batch: int = RELATIONS_PER_BATCH,
                     mentions_per_relation: int = MENTIONS_PER_RELATION):
    import torch

    groups = defaultdict(list)
    for index, relation in enumerate(relations):
        groups[relation].append(index)
    eligible = sorted(relation for relation, indices in groups.items() if len(indices) >= 2)
    if len(eligible) < relations_per_batch:
        raise ValueError("too few eligible S3 relations")
    output = []
    for _ in range(steps):
        relation_order = torch.randperm(len(eligible), generator=generator)[:relations_per_batch]
        batch = []
        for relation_index in relation_order.tolist():
            indices = groups[eligible[relation_index]]
            chosen = torch.randint(len(indices), (mentions_per_relation,), generator=generator)
            batch.extend(indices[index] for index in chosen.tolist())
        output.append(batch)
    return output


def supervised_contrastive_loss(projected, labels, temperature: float):
    import torch

    logits = projected @ projected.T / temperature
    eye = torch.eye(len(projected), dtype=torch.bool, device=projected.device)
    logits = logits.masked_fill(eye, -torch.inf)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    if not bool(positive.any(1).all()):
        raise ValueError("S3 supervised batch lacks a positive")
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_probability.masked_fill(~positive, 0).sum(1)
             / positive.sum(1)).mean()


def paired_info_nce(left, right, temperature: float = .10):
    import torch

    values = torch.cat((left, right), dim=0)
    batch = len(left)
    logits = values @ values.T / temperature
    logits.fill_diagonal_(-torch.inf)
    target = torch.cat((torch.arange(batch, 2 * batch, device=values.device),
                        torch.arange(batch, device=values.device)))
    return torch.nn.functional.cross_entropy(logits, target)


def roc_auc(gold: list[bool], scores: list[float]) -> float:
    positives = [score for score, label in zip(scores, gold) if label]
    negatives = [score for score, label in zip(scores, gold) if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        wins += sum(positive > negative for negative in negatives)
        wins += 0.5 * sum(positive == negative for negative in negatives)
    return wins / (len(positives) * len(negatives))


def retrieval_metrics(train_matrix, train_relations: list[str], query_matrix,
                      query_relations: list[str], ks=(4, 8, 16)) -> dict:
    import numpy as np
    import torch

    train_relations = np.asarray(train_relations)
    hits = {k: 0 for k in ks}
    reciprocal, queries = [], len(query_relations)
    train = torch.as_tensor(train_matrix, dtype=torch.float32, device="cuda")
    for start in range(0, queries, 256):
        query = torch.as_tensor(query_matrix[start:start + 256], dtype=torch.float32,
                                device="cuda")
        top = torch.topk(query @ train.T, k=max(ks), dim=1).indices.cpu().numpy()
        truth = train_relations[top] == np.asarray(query_relations[start:start + len(top)])[:, None]
        for k in ks:
            hits[k] += int(truth[:, :k].any(1).sum())
        reciprocal.extend((1.0 / (truth.argmax(1) + 1) * truth.any(1)).tolist())
    return {
        "queries": queries,
        **{f"recall_at_{k}": hits[k] / queries for k in ks},
        "mrr_at_16": sum(reciprocal) / queries,
    }


def project_matrix(head, matrix, batch_size: int = 1024):
    import numpy as np
    import torch

    output = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            value = torch.as_tensor(matrix[start:start + batch_size], dtype=torch.float32,
                                    device="cuda")
            output.append(head(value).cpu().numpy())
    return np.concatenate(output)


class ProjectionHead:
    """Small wrapper that keeps construction testable without importing torch at module load."""

    @staticmethod
    def create(input_dimension: int, output_dimension: int):
        import torch

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(input_dimension, output_dimension, bias=False)
                self.norm = torch.nn.LayerNorm(input_dimension)
                self.residual_in = torch.nn.Linear(input_dimension, output_dimension, bias=False)
                self.residual_out = torch.nn.Linear(output_dimension, output_dimension, bias=False)

            def forward(self, value):
                residual = self.residual_out(torch.nn.functional.gelu(
                    self.residual_in(self.norm(value))
                ))
                return torch.nn.functional.normalize(self.base(value) + residual, dim=-1)

        return Module()


def pair_indices(rows: list[dict], id_to_index: dict[str, int]):
    left, right, gold = [], [], []
    for row in rows:
        left.append(id_to_index[row["left_source_row_sha256"]])
        right.append(id_to_index[row["right_source_row_sha256"]])
        gold.append(bool(row["same_relation"]))
    return left, right, gold


def evaluate_head(head, train, calibration, calibration_pairs) -> dict:
    train_projected = project_matrix(head, train["matrix"])
    calibration_projected = project_matrix(head, calibration["matrix"])
    retrieval = retrieval_metrics(
        train_projected, train["relations"], calibration_projected,
        calibration["relations"],
    )
    index = {source_id: position for position, source_id in enumerate(calibration["ids"])}
    left, right, gold = pair_indices(calibration_pairs, index)
    scores = [float(calibration_projected[a] @ calibration_projected[b])
              for a, b in zip(left, right)]
    return {
        "retrieval": retrieval,
        "direct_pair_roc_auc": roc_auc(gold, scores),
        "direct_pair_score_mean_positive": sum(score for score, label in zip(scores, gold)
                                                if label) / sum(gold),
        "direct_pair_score_mean_negative": sum(score for score, label in zip(scores, gold)
                                                if not label) / (len(gold) - sum(gold)),
    }


def selection_key(summary: dict) -> tuple:
    config = summary["config"]
    return (
        summary["retrieval"]["recall_at_16"],
        summary["direct_pair_roc_auc"],
        -config["dimension"],
        -config["epoch"],
        -int(config["pubmed_pretraining"]),
        -config["temperature"],
        -config["margin"],
    )


def train(args) -> None:
    import numpy as np
    import torch

    data_root, embedding_root = Path(args.data_root), Path(args.embedding_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    data_audit = json.loads((data_root / "data_audit.json").read_text())
    if (manifest.get("protocol") != DATA_PROTOCOL
            or data_audit.get("protocol") != AUDIT_PROTOCOL
            or data_audit.get("passed") is not True
            or data_audit.get("dialogre_dev_rows_parsed") != 0
            or data_audit.get("dialogre_test_rows_parsed") != 0):
        raise ValueError("S3 projection requires sealed audited data")
    output = Path(args.output)
    weights = Path(args.weights)
    if output.exists() or weights.exists():
        raise FileExistsError(output)
    train_set, train_audits = load_embedding_set(
        embedding_root, "train", manifest["counts"]["train"]["mentions"]
    )
    calibration_set, calibration_audits = load_embedding_set(
        embedding_root, "calibration", manifest["counts"]["calibration"]["mentions"]
    )
    pubmed_full, full_audits = load_embedding_set(
        embedding_root, "pubmed_full", manifest["pubmed_unsupervised"]["unique_rows"]
    )
    pubmed_dropout, dropout_audits = load_embedding_set(
        embedding_root, "pubmed_dropout", manifest["pubmed_unsupervised"]["unique_rows"]
    )
    if pubmed_full["ids"] != pubmed_dropout["ids"]:
        raise ValueError("S3 PubMed view identities differ")
    train_pairs = load_jsonl(data_root / "train_relation_pairs.jsonl")
    calibration_pairs = load_jsonl(data_root / "calibration_relation_pairs.jsonl")
    train_index = {value: index for index, value in enumerate(train_set["ids"])}
    negative_pairs = [
        (train_index[row["left_source_row_sha256"]],
         train_index[row["right_source_row_sha256"]])
        for row in train_pairs if not row["same_relation"]
    ]
    relation_names = sorted(set(train_set["relations"]))
    relation_id = {name: index for index, name in enumerate(relation_names)}
    relation_labels = torch.tensor([relation_id[value] for value in train_set["relations"]],
                                   dtype=torch.long)
    train_matrix = torch.as_tensor(train_set["matrix"], dtype=torch.float32)
    pubmed_left = torch.as_tensor(pubmed_full["matrix"], dtype=torch.float32)
    pubmed_right = torch.as_tensor(pubmed_dropout["matrix"], dtype=torch.float32)
    input_dimension = train_matrix.shape[1]
    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    summaries, best_summary, best_state = [], None, None
    best_by_pretraining = {False: (None, None), True: (None, None)}
    for use_pubmed in (False, True):
        for dimension in DIMENSIONS:
            for temperature in TEMPERATURES:
                for margin in MARGINS:
                    torch.manual_seed(SEED)
                    head = ProjectionHead.create(input_dimension, dimension).to("cuda")
                    if use_pubmed:
                        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3,
                                                      weight_decay=1e-4)
                        generator = torch.Generator().manual_seed(SEED)
                        for _ in range(UNSUPERVISED_EPOCHS):
                            order = torch.randperm(len(pubmed_left), generator=generator)
                            for begin in range(0, len(order), 256):
                                index = order[begin:begin + 256]
                                left = head(pubmed_left[index].to("cuda"))
                                right = head(pubmed_right[index].to("cuda"))
                                loss = paired_info_nce(left, right)
                                optimizer.zero_grad(set_to_none=True)
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                                optimizer.step()
                    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3,
                                                  weight_decay=1e-4)
                    generator = torch.Generator().manual_seed(SEED)
                    last_loss = None
                    for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
                        batches = relation_batches(train_set["relations"], generator)
                        for batch_indices in batches:
                            batch = torch.tensor(batch_indices, dtype=torch.long)
                            projected = head(train_matrix[batch].to("cuda"))
                            contrastive = supervised_contrastive_loss(
                                projected, relation_labels[batch].to("cuda"), temperature
                            )
                            pair_choice = torch.randint(
                                len(negative_pairs), (HARD_NEGATIVE_BATCH,), generator=generator
                            )
                            pair_values = [negative_pairs[index] for index in pair_choice.tolist()]
                            left_indices = torch.tensor([value[0] for value in pair_values])
                            right_indices = torch.tensor([value[1] for value in pair_values])
                            left = head(train_matrix[left_indices].to("cuda"))
                            right = head(train_matrix[right_indices].to("cuda"))
                            negative_cosine = (left * right).sum(-1)
                            margin_loss = torch.relu(negative_cosine - margin).mean()
                            loss = contrastive + MARGIN_WEIGHT * margin_loss
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                            optimizer.step()
                            last_loss = float(loss.detach())
                        if epoch in CHECKPOINT_EPOCHS:
                            metrics = evaluate_head(
                                head, train_set, calibration_set, calibration_pairs
                            )
                            summary = {
                                "config": {
                                    "pubmed_pretraining": use_pubmed,
                                    "dimension": dimension,
                                    "temperature": temperature,
                                    "margin": margin,
                                    "epoch": epoch,
                                },
                                "last_training_loss": last_loss,
                                **metrics,
                            }
                            summaries.append(summary)
                            if best_summary is None or selection_key(summary) > selection_key(best_summary):
                                best_summary = copy.deepcopy(summary)
                                best_state = {name: value.detach().cpu().clone()
                                              for name, value in head.state_dict().items()}
                            arm_summary, _ = best_by_pretraining[use_pubmed]
                            if arm_summary is None or selection_key(summary) > selection_key(arm_summary):
                                best_by_pretraining[use_pubmed] = (
                                    copy.deepcopy(summary),
                                    {name: value.detach().cpu().clone()
                                     for name, value in head.state_dict().items()},
                                )
                            print(json.dumps({
                                "completed_configs": len(summaries),
                                "config": summary["config"],
                                "retrieval_at_16": summary["retrieval"]["recall_at_16"],
                                "auc": summary["direct_pair_roc_auc"],
                            }), flush=True)
                    del head, optimizer
                    torch.cuda.empty_cache()
    assert best_summary is not None and best_state is not None
    selected_head = ProjectionHead.create(
        input_dimension, best_summary["config"]["dimension"]
    )
    selected_head.load_state_dict(best_state)
    torch.save({
        "protocol": PROTOCOL,
        "config": best_summary["config"],
        "input_dimension": int(input_dimension),
        "state_dict": best_state,
        "pretraining_ablation": {
            str(key).lower(): {
                "config": summary["config"],
                "state_dict": state,
            }
            for key, (summary, state) in best_by_pretraining.items()
        },
    }, weights)

    # Resource and retrieval pilot: 872 independent calibration queries plus
    # the first 128 train mentions with self matches removed, exactly 1,000.
    selected_head = selected_head.to("cuda")
    selected_train = project_matrix(selected_head, train_set["matrix"])
    selected_calibration = project_matrix(selected_head, calibration_set["matrix"])
    pilot_started = time.perf_counter()
    train_tensor = torch.as_tensor(selected_train, dtype=torch.float32, device="cuda")
    cal_tensor = torch.as_tensor(selected_calibration, dtype=torch.float32, device="cuda")
    _ = torch.topk(cal_tensor @ train_tensor.T, k=16, dim=1).indices
    train_query = train_tensor[:128]
    train_scores = train_query @ train_tensor.T
    train_scores[torch.arange(128, device="cuda"), torch.arange(128, device="cuda")] = -torch.inf
    _ = torch.topk(train_scores, k=16, dim=1).indices
    torch.cuda.synchronize()
    pilot_seconds = time.perf_counter() - pilot_started
    seconds = time.perf_counter() - started
    result = {
        "protocol": PROTOCOL,
        "selected": best_summary,
        "best_by_pubmed_pretraining": {
            str(key).lower(): summary
            for key, (summary, _) in best_by_pretraining.items()
        },
        "summaries": summaries,
        "grid_size": len(summaries),
        "selection_order": [
            "retrieval_recall_at_16", "direct_pair_roc_auc", "smaller_dimension",
            "earlier_epoch", "no_pubmed_pretraining", "lower_temperature", "lower_margin",
        ],
        "training": {
            "seed": SEED,
            "supervised_steps_per_epoch": SUPERVISED_STEPS_PER_EPOCH,
            "checkpoint_epochs": CHECKPOINT_EPOCHS,
            "relations_per_batch": RELATIONS_PER_BATCH,
            "mentions_per_relation": MENTIONS_PER_RELATION,
            "hard_negative_batch": HARD_NEGATIVE_BATCH,
            "margin_weight": MARGIN_WEIGHT,
            "unsupervised_epochs": UNSUPERVISED_EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
        },
        "resource": {
            "seconds": seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "support_queries": 1000,
            "support_query_seconds": pilot_seconds,
            "support_queries_per_second": 1000 / pilot_seconds,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        },
        "data_manifest_sha256": sha(data_root / "manifest.json"),
        "data_audit_sha256": sha(data_root / "data_audit.json"),
        "embedding_audits_sha256": {
            dataset: [sha(embedding_root / f"{dataset}_shard_{shard}.audit.json")
                      for shard in range(2)]
            for dataset in ("train", "calibration", "pubmed_full", "pubmed_dropout")
        },
        "weights_sha256": sha(weights),
        "dialogre_dev_loaded": False,
        "dialogre_test_loaded": False,
        "pubmed_validation_loaded": False,
        "terminal_status": "projection_selected_before_fusion",
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["terminal_status"],
        "selected": best_summary,
        "seconds": seconds,
        "weights_sha256": result["weights_sha256"],
    }, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", required=True)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
