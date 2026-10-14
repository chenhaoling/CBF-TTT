"""Build the fresh scale-32 causal-ledger content-validation corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_qwen_causal_ledger_residual import (
    digest,
    FAMILIES,
    generate_group,
    null_value,
    RESOURCE_V2_SEED,
    TEMPLATES,
    validate,
    VALUE_POOL_SIZE,
)
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_causal_ledger_residual_v2_corpus_v1"
PHASE = "resource_v2"


def materialize(output, groups_per_family=8, seed=RESOURCE_V2_SEED):
    if groups_per_family != 8 or seed != RESOURCE_V2_SEED:
        raise ValueError("registered V2 resource corpus requires eight groups and frozen seed")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pairs = [
        pair for family in FAMILIES for group in range(groups_per_family)
        for pair in generate_group(seed, PHASE, family, group)
    ]
    audit = validate(pairs, expected_groups=8, expected_phase=PHASE)
    output.mkdir(parents=True)
    inputs, targets = output / "resource_v2.inputs.jsonl", output / "resource_v2.targets.jsonl"
    inputs.write_text("".join(json.dumps(inp) + "\n" for inp, _ in pairs))
    targets.write_text("".join(json.dumps(target) + "\n" for _, target in pairs))
    manifest = {
        "protocol": PROTOCOL,
        "phase": PHASE,
        "seed": seed,
        "groups_per_family": groups_per_family,
        "families": list(FAMILIES),
        "template_sha256": digest(TEMPLATES[PHASE]),
        "value_namespace": "Y plus four space-prefixed A-P tokens",
        "value_pool_size": VALUE_POOL_SIZE,
        "null": null_value(PHASE),
        "files_sha256": {path.name: sha(path) for path in (inputs, targets)},
        "audit": audit,
        "model_scored": False,
        "development_generated": False,
        "confirmation_generated": False,
        "ood_generated": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(root):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("phase") != PHASE:
        raise ValueError("unexpected V2 resource corpus")
    paths = [root / f"resource_v2.{kind}.jsonl" for kind in ("inputs", "targets")]
    for path in paths:
        if sha(path) != manifest["files_sha256"].get(path.name):
            raise ValueError("V2 resource corpus hash mismatch")
    inputs = [json.loads(line) for line in paths[0].read_text().splitlines() if line]
    targets = [json.loads(line) for line in paths[1].read_text().splitlines() if line]
    if len(inputs) != len(targets) or any(
        inp["id"] != target["id"] for inp, target in zip(inputs, targets)
    ):
        raise ValueError("V2 resource corpus alignment mismatch")
    return manifest, list(zip(inputs, targets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups-per-family", type=int, default=8)
    parser.add_argument("--seed", type=int, default=RESOURCE_V2_SEED)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output, args.groups_per_family, args.seed), indent=2))


if __name__ == "__main__":
    main()
