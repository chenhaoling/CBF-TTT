# MQuAKE-T Frozen Transfer V7 Report

## Result

The frozen ZsRE V3 resolver and V5 forgetting ensemble do not transfer zero-shot to
MQuAKE-T natural temporal sessions. The independent audit passed all seven integrity
checks. MQuAKE train/tune, MQuAKE-CF, LongMemEval-S, and LoCoMo were not used.

| Metric | Frozen controller | Matched random | Oracle |
|---|---:|---:|---:|
| candidate shortlist recall | 31.28% | — | — |
| candidate top-1 | 26.34% | — | — |
| stale deletion | 78.01% | 92.58% | 100.00% |
| current retention | 31.82% | 55.68% | 100.00% |
| non-target retention | 29.16% | 31.30% | 100.00% |
| strict old-deleted/current-retained pair state | 9.83% | 51.14% | 100.00% |

The controller made 41,418 deletions across 59,840 actions. With exactly the same
deletion count, random deletion exceeded controller pair state by 41.31 points; the
controller-minus-random subject-clustered 95% interval was `[-45.52,-39.18]` points.
Only the stale-deletion floor passed, so the preregistered transfer gate failed.

## Interpretation

The failure localizes first to scope retrieval: the true stale session appears in the
four-item shortlist only 31.28% of the time. The ZsRE-calibrated deletion margins also
shift strongly and cause broad over-deletion. This result rejects a claim of universal
zero-shot forgetting. It does not reject selective forgetting as a mechanism, because
the exact local-deletion oracle obtains 100% on every metric and session length.

## Resources and provenance

- 1,496 development episodes, 18 held-out edited subjects, 59,840 actions.
- Runtime: 156.89 seconds.
- Peak GPU allocated/reserved: 7.867/8.115 GiB on one RTX 5090.
- Raw MQuAKE-T SHA256:
  `58500a1d4aaf9e036c23a1d9dd6be2d73d693e35a695614842f211ea5aff59f2`.
- Compact metrics, training metadata, data manifest, and independent audit are stored
  beside this report. Hash-covered raw trajectories remain on hku-gpu2 under
  `/home/ctj/cbf_ttt_mquake_temporal_transfer_v7`.

## Registered next step

Fit the factorized scope resolver and deletion head on MQuAKE train only, calibrate
thresholds on tune only, and do not refit after calibration. A tune prerequisite must
pass before evaluating the entity-disjoint dev split. LoCoMo remains sealed.
