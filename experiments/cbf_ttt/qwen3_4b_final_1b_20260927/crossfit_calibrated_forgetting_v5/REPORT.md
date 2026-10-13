# Cross-fit calibrated forgetting V5

## Status

Completed and independently audited. The pooled out-of-fold prerequisite missed
two efficacy floors, so subject-disjoint development was not loaded or scored.
LongMemEval-S remained locked and LoCoMo remained sealed.

## OOF result

The five validation folds cover 1,068 episodes / 42,720 actions. Every episode
was evaluated by models that excluded its subject from both training and
threshold calibration.

| Metric | OOF result |
|---|---:|
| Action accuracy | 99.27% |
| Strict pair state | 83.90% |
| Stale deletion | 83.99% |
| Current retention | 99.81% |
| Non-target retention | 99.70% |
| Hard-retain action accuracy | 99.86% |
| Primary-update action accuracy | 83.90% |

Pair state by length was 85.02/84.64/82.77/83.15% for 16/32/48/64.
Calibration therefore removed almost all cumulative false deletion but remained
about one point below the fixed 85% stale-deletion and pair-state prerequisites.

## Conclusion

V5 establishes a strong safety/recall frontier rather than a release result.
It resolves V4's seed instability and refit calibration shift: current,
unrelated, and hard-retain safety all exceed 99.7%. The remaining errors are
missed updates. More threshold search is prohibited by the stop rule. The next
admissible change is an episode-level barrier objective that directly ranks the
single update above the maximum retain score while keeping the identical
nested cross-fit calibration protocol.

## Reproducibility

- independent audit: all ten checks passed;
- development artifacts are absent, as required;
- summary SHA-256: `66904bc5b63be5057c0f54c20e2010af5e6b04a6a1a7b79a1b7cab3c732f9bb4`;
- 7.0 MiB OOF raw trace remains hash-covered on `hku-gpu2` at
  `/home/ctj/cbf_ttt_crossfit_calibrated_forgetting_v5`.
