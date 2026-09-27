"""Summarize saved controller fits against constant predictors on the fixed dev labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cbf_ttt.controller import ForgettingController


def _rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def summarize(study_dir: Path, dev_paths: list[Path], sizes: list[int], seeds: list[int]) -> dict:
    dev = _rows(dev_paths)
    if not dev or any(row["split"] != "dev" for row in dev):
        raise ValueError("nonempty dev-only label files are required")
    semantic = torch.tensor([row["semantic"] for row in dev], dtype=torch.float32)
    scalars = torch.tensor([row["scalars"] for row in dev], dtype=torch.float32)
    labels = torch.tensor([row["alpha_star"] for row in dev], dtype=torch.float32)
    train_full = _rows([study_dir / "train_groups_1000.jsonl"])
    train_mean = sum(row["alpha_star"] for row in train_full) / len(train_full)
    constants = {str(value): F.mse_loss(torch.full_like(labels, value), labels).item()
                 for value in (0.0, 0.5, 1.0, train_mean)}

    models = []
    with torch.no_grad():
        for size in sizes:
            for seed in seeds:
                path = study_dir / f"controller_g{size}_s{seed}.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                controller = ForgettingController(
                    checkpoint["hidden_size"], checkpoint["scalar_size"],
                    checkpoint["width"], checkpoint["semantic_size"],
                )
                controller.load_state_dict(checkpoint["state_dict"])
                controller.eval()
                raw = controller(semantic, scalars)
                predicted = raw.clamp(0.0, 1.0)
                models.append({
                    "groups": size, "seed": seed, "checkpoint": str(path),
                    "best_epoch": checkpoint["best_epoch"],
                    "dev_raw_mse": F.mse_loss(raw, labels).item(),
                    "dev_clipped_mse": F.mse_loss(predicted, labels).item(),
                    "mean_alpha": predicted.mean().item(),
                    "alpha_std": predicted.std(unbiased=False).item(),
                    "fraction_below_0_1": (predicted < 0.1).float().mean().item(),
                    "fraction_above_0_9": (predicted > 0.9).float().mean().item(),
                })
    selected = {str(size): min((item for item in models if item["groups"] == size),
                               key=lambda item: item["dev_raw_mse"])["checkpoint"]
                for size in sizes}
    result = {
        "dev_labels": len(dev), "dev_groups": len({row["group_id"] for row in dev}),
        "train_mean_alpha": train_mean, "constant_dev_mse": constants,
        "models": models, "selected_by_dev_raw_mse": selected,
    }
    (study_dir / "learning_curve_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--dev-labels", nargs="+", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=(250, 500, 1000))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    args = parser.parse_args()
    print(json.dumps(summarize(args.study_dir, args.dev_labels, args.sizes, args.seeds), indent=2))


if __name__ == "__main__":
    main()
