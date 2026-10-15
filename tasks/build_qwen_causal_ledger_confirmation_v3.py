"""Build the one-time 64-group-per-family causal-ledger confirmation corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_qwen_causal_ledger_residual import (
    CONFIRMATION_SEED,
    digest,
    FAMILIES,
    generate_group,
    null_value,
    TEMPLATES,
    validate,
)
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_causal_ledger_confirmation_v3_corpus_v1"
PHASE = "confirmation"
GROUPS_PER_FAMILY = 64


def materialize(output, groups_per_family=GROUPS_PER_FAMILY, seed=CONFIRMATION_SEED):
    if groups_per_family != GROUPS_PER_FAMILY or seed != CONFIRMATION_SEED:
        raise ValueError("registered confirmation size or seed changed")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pairs = [
        pair for family in FAMILIES for group in range(groups_per_family)
        for pair in generate_group(seed, PHASE, family, group)
    ]
    audit = validate(pairs, expected_groups=groups_per_family, expected_phase=PHASE)
    output.mkdir(parents=True)
    inputs = output / "confirmation.inputs.jsonl"
    targets = output / "confirmation.targets.jsonl"
    inputs.write_text("".join(json.dumps(inp) + "\n" for inp, _ in pairs))
    targets.write_text("".join(json.dumps(target) + "\n" for _, target in pairs))
    manifest = {
        "protocol": PROTOCOL,
        "phase": PHASE,
        "seed": seed,
        "groups_per_family": groups_per_family,
        "families": list(FAMILIES),
        "template_sha256": digest(TEMPLATES[PHASE]),
        "null": null_value(PHASE),
        "files_sha256": {path.name: sha(path) for path in (inputs, targets)},
        "audit": audit,
        "model_scored": False,
        "confirmation_generated": True,
        "ood_generated": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("phase") != PHASE:
        raise ValueError("unexpected confirmation corpus")
    paths = [root / f"confirmation.{kind}.jsonl" for kind in ("inputs", "targets")]
    for path in paths:
        if sha(path) != manifest["files_sha256"].get(path.name):
            raise ValueError("confirmation corpus hash mismatch")
    inputs = [json.loads(line) for line in paths[0].read_text().splitlines() if line]
    targets = [json.loads(line) for line in paths[1].read_text().splitlines() if line]
    if len(inputs) != len(targets) or any(
        inp["id"] != target["id"] for inp, target in zip(inputs, targets)
    ):
        raise ValueError("confirmation corpus alignment mismatch")
    return manifest, list(zip(inputs, targets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups-per-family", type=int, default=GROUPS_PER_FAMILY)
    parser.add_argument("--seed", type=int, default=CONFIRMATION_SEED)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output, args.groups_per_family, args.seed), indent=2))


if __name__ == "__main__":
    main()
