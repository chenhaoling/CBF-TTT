"""Run S4 deterministic controls and the gold-localized representation oracle."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from tasks.build_retrieval_evidence_forgetting_s3 import stable
from tasks.extract_pair_localized_relation_memory_s4_embeddings import PROTOCOL as EMBED_PROTOCOL
from tasks.pack_cbf_event_warmup import sha
from tasks.train_retrieval_evidence_forgetting_s3_projection import (
    ProjectionHead,
    pair_indices,
    project_matrix,
    relation_batches,
    retrieval_metrics,
    roc_auc,
    supervised_contrastive_loss,
)
from tasks.train_retrieval_evidence_forgetting_s3_projection import (
    load_embedding_set as load_s3_embedding_set,
)


PROTOCOL = "qwen_pair_localized_relation_memory_s4_oracle_v1"
SEED = 35117
CHECKPOINT_EPOCHS = (10, 25, 50)
STEPS_PER_EPOCH = 64
MARGIN = 0.20
MARGIN_WEIGHT = 0.25
TEMPERATURE = 0.05


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def load_s4_embedding_set(root: str | Path, dataset: str, expected_rows: int):
    import numpy as np

    root = Path(root)
    embedding_audit = json.loads((root / "embeddings_audit.json").read_text())
    if not embedding_audit.get("passed"):
        raise ValueError("S4 oracle requires passed embedding audit")
    values, audits = [], []
    for shard in range(2):
        path = root / f"{dataset}_shard_{shard}.npz"
        audit_path = root / f"{dataset}_shard_{shard}.audit.json"
        audit = json.loads(audit_path.read_text())
        if (audit.get("protocol") != EMBED_PROTOCOL
                or audit.get("dataset") != dataset
                or audit.get("shard") != shard
                or audit.get("truncation") is not False
                or audit.get("ttt_mode") is not False
                or audit.get("dialogre_dev_loaded") is not False
                or audit.get("dialogre_test_loaded") is not False
                or sha(path) != audit.get("output_sha256")):
            raise ValueError(f"invalid S4 embedding shard: {dataset}/{shard}")
        with np.load(path) as arrays:
            values.extend(zip(
                arrays["indices"].tolist(), arrays["view_ids"].tolist(),
                arrays["source_ids"].tolist(), arrays["relations"].tolist(),
                arrays["sources"].tolist(), arrays["dialogue_ids"].tolist(),
                arrays["views"].tolist(), arrays["center_turns"].tolist(),
                arrays["side_features"].astype("float32"),
                arrays["embeddings"].astype("float32"),
            ))
        audits.append(audit)
    values.sort(key=lambda row: row[0])
    if len(values) != expected_rows or [row[0] for row in values] != list(range(expected_rows)):
        raise ValueError(f"incomplete S4 embedding dataset: {dataset}")
    matrix = np.stack([row[9] for row in values])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return {
        "view_ids": [row[1] for row in values],
        "ids": [row[2] for row in values],
        "relations": [row[3] for row in values],
        "sources": [row[4] for row in values],
        "dialogues": [row[5] for row in values],
        "views": [row[6] for row in values],
        "centers": [row[7] for row in values],
        "side_features": np.stack([row[8] for row in values]),
        "matrix": matrix,
    }, audits


def select_view(embedding_set: dict, mention_ids: list[str], view: str):
    import numpy as np

    mapping = {(source_id, current_view): index for index, (source_id, current_view)
               in enumerate(zip(embedding_set["ids"], embedding_set["views"]))}
    indices = [mapping[(source_id, view)] for source_id in mention_ids]
    return embedding_set["matrix"][indices]


def random_view(embedding_set: dict, mention_ids: list[str]):
    import numpy as np

    groups = defaultdict(list)
    for index, source_id in enumerate(embedding_set["ids"]):
        groups[source_id].append(index)
    indices = []
    for source_id in mention_ids:
        choices = groups[source_id]
        selected = int(stable(SEED, "s4-random-window", source_id)[:16], 16) % len(choices)
        indices.append(choices[selected])
    return embedding_set["matrix"][indices]


def evaluate_matrices(train_ids, train_relations, train_matrix,
                      calibration_ids, calibration_relations, calibration_matrix,
                      pairs) -> dict:
    index = {source_id: position for position, source_id in enumerate(calibration_ids)}
    filtered = [row for row in pairs
                if row["left_source_row_sha256"] in index
                and row["right_source_row_sha256"] in index]
    left, right, gold = pair_indices(filtered, index)
    scores = [float(calibration_matrix[a] @ calibration_matrix[b])
              for a, b in zip(left, right)]
    return {
        "retrieval": retrieval_metrics(
            train_matrix, train_relations, calibration_matrix, calibration_relations
        ),
        "direct_pair_roc_auc": roc_auc(gold, scores),
        "direct_pair_pairs": len(filtered),
        "direct_pair_positive": sum(gold),
        "direct_pair_negative": len(gold) - sum(gold),
        "direct_pair_score_mean_positive": (
            sum(score for score, label in zip(scores, gold) if label) / sum(gold)
            if sum(gold) else 0.0
        ),
        "direct_pair_score_mean_negative": (
            sum(score for score, label in zip(scores, gold) if not label)
            / (len(gold) - sum(gold)) if len(gold) - sum(gold) else 0.0
        ),
    }


def oracle_subset(control_set: dict, view: str = "oracle") -> dict:
    import numpy as np

    indices = [index for index, value in enumerate(control_set["views"]) if value == view]
    return {
        "ids": [control_set["ids"][index] for index in indices],
        "relations": [control_set["relations"][index] for index in indices],
        "dialogues": [control_set["dialogues"][index] for index in indices],
        "matrix": control_set["matrix"][np.asarray(indices)],
    }


def selection_key(summary: dict) -> tuple:
    return (summary["retrieval"]["recall_at_16"], summary["direct_pair_roc_auc"],
            -summary["epoch"])


def run(args) -> None:
    import numpy as np
    import torch

    data_root, s3_data_root = Path(args.data_root), Path(args.s3_data_root)
    embedding_root, s3_embedding_root = Path(args.embedding_root), Path(args.s3_embedding_root)
    output, weights_output = Path(args.output), Path(args.weights_output)
    if output.exists() or weights_output.exists():
        raise FileExistsError(output)
    manifest = json.loads((data_root / "manifest.json").read_text())
    data_audit = json.loads((data_root / "data_audit.json").read_text())
    if (not data_audit.get("passed") or manifest.get("dialogre_dev_parsed") is not False
            or manifest.get("dialogre_test_parsed") is not False):
        raise ValueError("S4 oracle requires audited sealed data")
    s3_manifest = json.loads((s3_data_root / "manifest.json").read_text())
    train_mentions = load_jsonl(data_root / "train_localized_mentions.jsonl")
    calibration_mentions = load_jsonl(data_root / "calibration_localized_mentions.jsonl")
    train_ids = [row["source_row_sha256"] for row in train_mentions]
    calibration_ids = [row["source_row_sha256"] for row in calibration_mentions]
    train_relations = [row["relation"] for row in train_mentions]
    calibration_relations = [row["relation"] for row in calibration_mentions]
    train_candidates, candidate_train_audits = load_s4_embedding_set(
        embedding_root, "train_candidates", manifest["counts"]["train"]["candidates"]
    )
    calibration_candidates, candidate_calibration_audits = load_s4_embedding_set(
        embedding_root, "calibration_candidates",
        manifest["counts"]["calibration"]["candidates"],
    )
    train_controls, control_train_audits = load_s4_embedding_set(
        embedding_root, "train_controls",
        3 * len(train_mentions) + manifest["counts"]["train"][
            "dialogre_trigger_supervised_mentions"
        ],
    )
    calibration_controls, control_calibration_audits = load_s4_embedding_set(
        embedding_root, "calibration_controls",
        3 * len(calibration_mentions) + manifest["counts"]["calibration"][
            "dialogre_trigger_supervised_mentions"
        ],
    )
    train_full, _ = load_s3_embedding_set(
        s3_embedding_root, "train", s3_manifest["counts"]["train"]["mentions"]
    )
    calibration_full, _ = load_s3_embedding_set(
        s3_embedding_root, "calibration",
        s3_manifest["counts"]["calibration"]["mentions"],
    )
    if train_full["ids"] != train_ids or calibration_full["ids"] != calibration_ids:
        raise ValueError("S3 and S4 mention order differs")
    pairs = load_jsonl(s3_data_root / "calibration_relation_pairs.jsonl")
    train_pairs = load_jsonl(s3_data_root / "train_relation_pairs.jsonl")
    checkpoint = torch.load(args.s3_projection_weights, map_location="cpu", weights_only=False)
    head = ProjectionHead.create(train_full["matrix"].shape[1], 512).to("cuda")
    head.load_state_dict(checkpoint["state_dict"])
    controls = {}
    raw_controls = {
        "full_dialogue": (train_full["matrix"], calibration_full["matrix"]),
        "random_window": (random_view(train_candidates, train_ids),
                          random_view(calibration_candidates, calibration_ids)),
        **{f"deterministic_r{radius}": (
            select_view(train_controls, train_ids, f"deterministic_r{radius}"),
            select_view(calibration_controls, calibration_ids, f"deterministic_r{radius}"),
        ) for radius in (0, 1, 2)},
    }
    for name, (train_raw, calibration_raw) in raw_controls.items():
        controls[name] = evaluate_matrices(
            train_ids, train_relations, project_matrix(head, train_raw),
            calibration_ids, calibration_relations, project_matrix(head, calibration_raw), pairs,
        )

    train_oracle, calibration_oracle = map(
        oracle_subset, (train_controls, calibration_controls)
    )
    train_index = {value: index for index, value in enumerate(train_oracle["ids"])}
    negative_pairs = [
        (train_index[row["left_source_row_sha256"]],
         train_index[row["right_source_row_sha256"]])
        for row in train_pairs if not row["same_relation"]
        and row["left_source_row_sha256"] in train_index
        and row["right_source_row_sha256"] in train_index
    ]
    if not negative_pairs:
        raise ValueError("oracle subset has no registered hard negatives")
    relation_names = sorted(set(train_oracle["relations"]))
    relation_id = {name: index for index, name in enumerate(relation_names)}
    labels = torch.tensor([relation_id[value] for value in train_oracle["relations"]])
    matrix = torch.as_tensor(train_oracle["matrix"], dtype=torch.float32)
    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    oracle_head = ProjectionHead.create(matrix.shape[1], 512).to("cuda")
    oracle_head.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(oracle_head.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)
    started = time.perf_counter()
    summaries, states = [], {}
    for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
        last_loss = None
        for batch_indices in relation_batches(
                train_oracle["relations"], generator, steps=STEPS_PER_EPOCH):
            batch = torch.tensor(batch_indices)
            projected = oracle_head(matrix[batch].to("cuda"))
            contrastive = supervised_contrastive_loss(
                projected, labels[batch].to("cuda"), TEMPERATURE
            )
            choice = torch.randint(len(negative_pairs), (64,), generator=generator)
            selected = [negative_pairs[index] for index in choice.tolist()]
            left = oracle_head(matrix[torch.tensor([pair[0] for pair in selected])].to("cuda"))
            right = oracle_head(matrix[torch.tensor([pair[1] for pair in selected])].to("cuda"))
            margin_loss = torch.relu((left * right).sum(-1) - MARGIN).mean()
            loss = contrastive + MARGIN_WEIGHT * margin_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(oracle_head.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.detach())
        if epoch in CHECKPOINT_EPOCHS:
            summary = evaluate_matrices(
                train_oracle["ids"], train_oracle["relations"],
                project_matrix(oracle_head, train_oracle["matrix"]),
                calibration_oracle["ids"], calibration_oracle["relations"],
                project_matrix(oracle_head, calibration_oracle["matrix"]), pairs,
            )
            summary.update({"epoch": epoch, "last_training_loss": last_loss})
            summaries.append(summary)
            states[epoch] = copy.deepcopy(oracle_head.state_dict())
    selected = max(summaries, key=selection_key)
    torch.save({
        "protocol": PROTOCOL,
        "epoch": selected["epoch"],
        "state_dict": states[selected["epoch"]],
        "s3_initialization_sha256": sha(args.s3_projection_weights),
        "config": {"dimension": 512, "temperature": TEMPERATURE,
                   "margin": MARGIN, "margin_weight": MARGIN_WEIGHT,
                   "steps_per_epoch": STEPS_PER_EPOCH},
    }, weights_output)
    result = {
        "protocol": PROTOCOL,
        "controls": controls,
        "oracle_summaries": summaries,
        "selected_oracle": selected,
        "oracle_train_mentions": len(train_oracle["ids"]),
        "oracle_calibration_mentions": len(calibration_oracle["ids"]),
        "oracle_registered_negative_pairs": len(negative_pairs),
        "selection_rule": "recall@16, direct pair ROC-AUC, earlier epoch",
        "weights_sha256": sha(weights_output),
        "resource": {
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        },
        "input_audit_sha256": {
            "data": sha(data_root / "data_audit.json"),
            "embeddings": sha(embedding_root / "embeddings_audit.json"),
        },
        "embedding_shard_outputs_sha256": [
            value["output_sha256"] for value in candidate_train_audits
            + candidate_calibration_audits + control_train_audits
            + control_calibration_audits
        ],
        "dialogre_dev_loaded": False,
        "dialogre_test_loaded": False,
        "pubmed_validation_loaded": False,
        "terminal_status": "oracle_feasibility_scored_before_learned_localizer",
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_oracle": selected, "controls": controls,
                      "resource": result["resource"]}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--s3-data-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--s3-embedding-root", required=True)
    parser.add_argument("--s3-projection-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights-output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
