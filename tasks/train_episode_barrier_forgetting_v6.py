"""Run V5 cross-fitting with an episode-level max-retain barrier objective."""

from __future__ import annotations

import argparse
import copy
import json

from tasks.train_crossfit_calibrated_forgetting_v5 import EPOCH, run
from tasks.train_long_session_sparse_gate_v1 import SparseGate


PROTOCOL = "episode_barrier_forgetting_v6"
MARGIN = .5
BCE_WEIGHT = .25


def fit_barrier(rows, seed, epochs=EPOCH):
    import torch

    values = torch.stack([row["feature"] for row in rows])
    labels = torch.tensor([row["label"] for row in rows])
    mean = values.mean(0); std = values.std(0, unbiased=False).clamp_min(1e-4)
    values = (values - mean) / std
    grouped = {}
    for index, row in enumerate(rows): grouped.setdefault(row["episode_id"], []).append(index)
    positive, negative = [], []
    for indices in grouped.values():
        positives = [index for index in indices if rows[index]["label"]]
        negatives = [index for index in indices if not rows[index]["label"]]
        if len(positives) != 1 or not negatives:
            raise ValueError("barrier training requires one update and retain events per episode")
        positive.append(positives[0]); negative.append(torch.tensor(negatives, dtype=torch.long))
    gate = SparseGate(values.shape[-1], seed)
    optimizer = torch.optim.AdamW(gate.model.parameters(), lr=.01, weight_decay=.1)
    output = {}
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True); logits = gate(values)
        barriers = [torch.nn.functional.softplus(
            MARGIN - (logits[pos] - logits[neg].max()))
            for pos, neg in zip(positive, negative)]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss = torch.stack(barriers).mean() + BCE_WEIGHT * bce
        loss.backward(); torch.nn.utils.clip_grad_norm_(gate.model.parameters(), 1.); optimizer.step()
        if epoch == epochs:
            output[epoch] = {"state": copy.deepcopy(gate.model.state_dict()),
                             "mean": mean, "std": std, "dim": values.shape[-1],
                             "epoch": epoch, "seed": seed, "train_loss": float(loss.detach()),
                             "barrier_margin": MARGIN, "bce_weight": BCE_WEIGHT}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--v1-root", required=True); parser.add_argument("--v3-root", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True); parser.add_argument("--batch-size", type=int, default=128)
    result = run(parser.parse_args(), protocol=PROTOCOL, fit_gate_fn=fit_barrier,
                 objective="episode_max_retain_barrier_margin_0.5_plus_0.25_bce")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
