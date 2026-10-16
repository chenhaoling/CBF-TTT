# V10 FewRel model-free corpus audit

## Decision

The V10 corpus passed every independent model-free construction check and is
authorized for the registered 28-group Qwen3-4B resource pilot.

## Capacity

- 48/48 registered FewRel training relations are eligible.
- 33,600 official rows were accounted for; 31,883 remained after exclusions.
- 1,314 rows with multiple or invalid entity mentions and 403 duplicate
  relation statements were excluded.
- 672 source groups, 1,344 twin episodes, 4,032 boundaries and 20,160 queries
  were constructed.
- Every relation has 14 groups; every lifecycle mechanism has 96 groups.
- The formal corpus uses 24,192 unique source rows and 24,192 unique entity
  pairs, with no reuse across groups.
- The pilot contains the complete 14 groups for two relations: 28 groups,
  56 twin episodes and 168 labeled boundaries.

## Integrity

The independent auditor reconstructed every material and episode from the
official FewRel source and the two development exclusion files.  All 15 checks
passed, including source hashes and train partition, development overlap
exclusion, source-row/entity-pair disjointness, natural exact/rephrased query
structure, counts, file hashes and sealed benchmark controls.  No Qwen model,
development label/model score or held-out benchmark was loaded in this stage.

The first attempt stopped on repeated entity mentions before producing data.
The retry records these ambiguous rows as explicit exclusions; it did not
relax any eligibility, disjointness or leakage condition.

## Labels and features

The 28-group pilot passed all gates with 11.49 points of safe-oracle headroom,
15.67% positive safe-benefit candidates, 100% answer coverage and 11.30 GiB
peak reserved memory.  The two-GPU projection of 1,253 seconds matched the
formal 1,255.92-second run.

All 4,032 formal boundary labels passed the scale-consistency audit.  They
contained 12,096 record candidates, 14.14% positive safe-benefit candidates,
11.42 points of oracle headroom, 100% answer coverage and zero oracle
collateral.  The locked V7 feature stage then produced 12,096 finite 66-feature
rows in 157.49 seconds; exact teachers, context features, relation coverage and
all hashes were independently reconstructed.  Compact summaries and audits
for each stage are included in this archive.
