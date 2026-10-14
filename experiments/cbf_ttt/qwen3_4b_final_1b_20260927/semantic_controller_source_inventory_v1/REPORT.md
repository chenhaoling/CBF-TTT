# Post-V20 semantic-controller source inventory

## Purpose and release boundary

This data-only stage prepares the development sources for the semantic
forgetting controller registered in `POST_V20_SEMANTIC_CONTROLLER_NATURAL_TRANSFER_PLAN.md`.
The allocation code and rules were frozen while V19 had written 148/320 rows,
before any V19 aggregate, arm outcome or causal-rule result was read.  Building
this inventory does not release controller training: that still requires the
registered V19/V20 predecessor gate.

The inventory reads no model outcomes, LoCoMo, LongMemEval-S, ZsRE dev/test or
MQuAKE-CF.  CounterFact is already-observed development data.  MQuAKE-T uses
only its previously fixed train+tune subjects.  ZsRE is loaded only from the
previously materialized calibration JSONL and is reserved for readout
calibration.

## Implementation

- `tasks/build_semantic_controller_source_inventory.py` validates registered
  source hashes, extracts relation surfaces, masks exact subject/value spans,
  constructs five deterministic relation folds, removes rows that would place
  one normalized subject in multiple folds, and records every exclusion.
- `scripts/audit_semantic_controller_source_inventory.py` reconstructs all
  rows and exclusions from the three sources, verifies every file hash and
  fold map, and checks that sealed sources and model outcomes were unused.
- `scripts/run_semantic_controller_source_inventory.sh` is the reproducible
  build/audit entry point and accepts `PYTHON_BIN` for non-interactive servers.
- `tests/test_semantic_controller_source_inventory.py` covers masking,
  deterministic allocation, relation/subject isolation, explicit conflict
  exclusion, MQuAKE dev exclusion and rejection of non-calibration ZsRE input.

## Audited result

Source hashes match the registered files:

- CounterFact: `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`;
- MQuAKE-T: `58500a1d4aaf9e036c23a1d9dd6be2d73d693e35a695614842f211ea5aff59f2`;
- ZsRE calibration: `d19bb92ca6f487ce046e75a0a65e384b831a7bbc988e81436df68b3cd7beec6f`.

There are 23,413 eligible semantic rows before the subject-conflict filter and
22,496 afterward.  The builder removes 917 cross-fold subject conflicts and all
374 reserved MQuAKE dev rows.  ZsRE retains 260/267 calibration subjects after
removing seven subjects that overlap the semantic corpus.  All five folds are
non-empty and exactly relation-disjoint and subject-disjoint.  The independent
audit passes every check; the manifest SHA256 is
`df3002f4913663ef829b865b3d1502305b712f5a96c401568c413a9fa2862836`.

Fold row counts are 4,381/4,461/4,523/4,523/4,608.  MQuAKE rows per fold are
790/10/7/445/1.  This source imbalance is structural: allowed MQuAKE train+tune
contains P6=995 rows, P35=481 and only 18 rows across the other five relations.
Balancing MQuAKE across five relation-disjoint folds is impossible.

A count-only resource pilot first tried exact equal relation and equal
within-relation source mass.  It reduced effective sample size from roughly
13,500 rows to 239--1,534 and gave a maximum row weight of 560, so it was
rejected before any model outcome was available.  The frozen replacement uses
each train row once and weights relation-source group frequency `n` by
`n^-0.5`, normalized to mean one.  Across outer folds its effective sample size
is 10,739--12,458 (80.35%--92.20% of rows) and maximum weight is
25.29--25.81.  All five resource gates and the exact reconstruction audit pass;
the balancing manifest SHA256 is
`9eed136cb71ba92df18252dc80232bf691a5e0337ae88208babef33f2c4d795c`.
The formal evaluation must report micro, subject-macro, relation-macro and
source-stratified results.  No row or relation is duplicated across folds.

The complete data remain on hku-gpu2 at
`/home/ctj/cbf_ttt_semantic_controller_inventory_20261014`; only `manifest.json`,
`audit.json` and this report are versioned.  The semantic-row output hash is
`222f7c51d384a9e26da93ecfd0dda91dd0add148c9901e5765f2fe4458614a99`.
The full balancing rows remain at
`/home/ctj/cbf_ttt_semantic_controller_balancing_sqrt_20261014`; compact
`balancing_manifest.json` and `balancing_audit.json` are versioned here.

## Reproduction

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
PYTHON_BIN=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
  scripts/run_semantic_controller_source_inventory.sh \
  /home/ctj/cbf_ttt_semantic_controller_inventory_20261014
```

The output directory must not already exist.  Reproduction should use a fresh
path, or run the auditor alone against the archived inventory.

## Compatibility, limitations and next step

All changes are opt-in data tooling and do not alter baseline model, training,
TTT writer or inference behavior.  This inventory is design and leakage
evidence, not a model-performance result.  CounterFact has already been
observed extensively and can support controller development only.  MQuAKE has
too few balanced relation families for a standalone five-fold source claim.
If the predecessor gate releases controller training, the next implementation
must consume only the retained rows, use relation-balanced sampling, keep Qwen
frozen, and preserve the registered nested OOF and natural-test release gates.
