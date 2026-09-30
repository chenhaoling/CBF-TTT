"""Measure raw twin-candidate differences on dev; never score held-out answers."""

import argparse
import json
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    episodes = [json.loads(line) for line in (args.root/"episodes.jsonl").read_text().splitlines()]
    groups = {}
    for row in episodes:
        if row["split"] == "dev":
            groups.setdefault(row["group_id"], []).append(row)
    values = []
    for group, pair in groups.items():
        if len(pair) != 2 or pair[0]["twin_id"] != pair[1]["id"]:
            raise ValueError("invalid dev pair")
        a, b = [torch.load(args.root/"features"/(r["id"]+".pt"), map_location="cpu", weights_only=True)["candidates"] for r in pair]
        for layer in a:
            x, y = a[layer].float(), b[layer].float()
            difference = x-y
            values.append({"group_id": group, "layer": layer,
                           "relative_norm": float(difference.norm()/((x.norm()+y.norm())/2).clamp_min(1e-12)),
                           "changed_fraction": float((difference != 0).float().mean())})
        del a, b, x, y, difference
    summary = {"split": "dev", "groups": len(groups), "layer_pair_records": len(values),
               "relative_difference_definition": "||D_a-D_b||_F / ((||D_a||_F+||D_b||_F)/2)",
               "all_layer_pairs_identical": all(r["relative_norm"] == 0 for r in values),
               "relative_norm_min": min(r["relative_norm"] for r in values),
               "relative_norm_median": statistics.median(r["relative_norm"] for r in values),
               "relative_norm_max": max(r["relative_norm"] for r in values),
               "changed_fraction_median": statistics.median(r["changed_fraction"] for r in values),
               "by_layer": {str(layer): {"relative_norm_mean": statistics.mean(r["relative_norm"] for r in values if r["layer"] == layer),
                                          "changed_fraction_mean": statistics.mean(r["changed_fraction"] for r in values if r["layer"] == layer)}
                            for layer in sorted({r["layer"] for r in values})}}
    (args.root/"candidate_signal.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
