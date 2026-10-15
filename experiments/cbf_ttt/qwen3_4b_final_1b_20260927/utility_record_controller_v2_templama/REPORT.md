# Utility record controller V2: TempLAMA replication

## Current stage

The official TempLAMA source, deterministic decontamination and train-only
resource pilot are complete.  Validation and test model outcomes remain
unobserved.  The V1 LAMA/ParaRel confirmation and all external benchmarks also
remain sealed.

## Source audit

- Official repository commit:
  `865fae65f63ef7e6b2989d4ff8b47f61750415a8`
- Official rows: 10,693 train, 4,654 validation and 34,963 test.
- The first audit stopped on official cross-split overlap: 26 train/test, one
  train/validation and 14 validation/test subjects; two rendered surfaces also
  overlapped.
- The preregistered subject-owner hash and removal of both colliding surfaces
  left 884/371/2,945 eligible temporal trajectories with all nine relations.
- Retry1 passed all nine source checks.

## Train-only resource pilot

The model-free corpus contains 72 real temporal-transition groups, 144 twins,
432 sparse boundaries and 2,160 immediate queries.  Every relation contributes
eight target groups; episode lengths are exactly balanced and all ten
independent data checks passed.

The two-GPU ternary action grid evaluated 432 labels and 4,320 future queries:

| Metric | Result |
|---|---:|
| Retain-all accuracy | 83.866% |
| Safe oracle accuracy | 91.181% |
| Safe oracle gain | +7.315pp |
| Group-bootstrap 95% CI | [+4.977, +9.931]pp |
| Added unrelated errors | 0 |
| Additional unrelated correct | 93 |
| Half-strength unique fraction | 2.083% |
| Half strength changed prediction | No |
| Two-GPU critical time | 135.07 s |
| Peak reserved memory | 7.69 GiB/card |

All eight scientific resource checks and the independent execution audit
passed.  The registered primary action space is binary.  A 256-group formal
run is projected at 480 seconds and 0.012 GiB of label output.

## Next decision

Construct the 256-group validation corpus using the frozen Hamilton quotas in
`QWEN_UTILITY_RECORD_CONTROLLER_V2_CAPACITY_CLARIFICATION.md`, label all binary
endpoint interventions, extract the unchanged 66-dimensional features and
evaluate the already frozen three V1 controllers.  TempLAMA test is opened only
if every prospective development gate passes.
