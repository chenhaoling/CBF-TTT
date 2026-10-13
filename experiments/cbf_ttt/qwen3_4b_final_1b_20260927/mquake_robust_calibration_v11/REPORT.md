# MQuAKE-T Subject-Level Robust Calibration V11

## Question

V10 achieved 96.12% strict pair state on its one-time MQuAKE-T development split,
but non-target and hard-retain safety were only about 91.8%. V11 asked whether a
nested subject-level calibration protocol could preserve deletion while raising unseen
subject safety. It did not inspect V10 development, MQuAKE-CF, LongMemEval, or LoCoMo.

## Protocol

- Data: MQuAKE-T train+tune, 5,976 episodes from 68 subjects.
- Model features: frozen Qwen3-4B checkpoint after the 1B-token pretraining run.
- Candidate: frozen `anchor_4` plus V8 ranker.
- Outer validation: five deterministic subject-disjoint folds.
- Inner calibration: the next fold; fitting used the remaining three folds.
- Gate: three rank-16 BCE seeds, 50 epochs, no post-calibration refit.
- Calibration constraint: current, non-target, and hard-retain each at least 99%.
- Control: matched random deletion with the exact same number of deletions.

## Audited result

| Metric | OOF result | Registered floor | Pass |
|---|---:|---:|:---:|
| Candidate top-1 | 96.82% | 99% | No |
| Current retention | 100.00% | 99% | Yes |
| Non-target retention | 99.05% | 97% | Yes |
| Hard-retain | 99.51% | 97% | Yes |
| Stale deletion | 83.08% | 90% | No |
| Strict pair state | 83.08% | 90% | No |
| Minimum-fold pair state | 69.88% | 85% | No |
| Gain over matched random | +79.53 points, 95% CI [72.69, 86.49] | CI lower bound > 0 | Yes |

Pair state by fold was 83.28%, 85.62%, 79.68%, 96.91%, and 69.88%.
The independent auditor passed all seven checks, including exact metric recomputation,
threshold selection, subject identities, matched deletion counts, hashes, and access
locks. The registered gate failed, so MQuAKE-CF and LoCoMo remain unscored.

## Read-only failure attribution

The post-terminal diagnostic made no model calls and selected no threshold. At each
primary update it compared the on-policy top candidate and deletion with the true stale
session:

| Outcome | Episodes | Rate |
|---|---:|---:|
| Correct deletion | 4,912 | 82.20% |
| Correct candidate, threshold rejected | 832 | 13.92% |
| Wrong candidate, deleted | 61 | 1.02% |
| Wrong candidate, threshold rejected | 171 | 2.86% |

The dominant controller error is therefore conservative rejection under subject shift.
Fold 4 rejected 30.20% of updates despite selecting the right stale memory; fold 2
rejected 20.32%.

Candidate failure was also sharply clustered. Only three of 68 subjects had imperfect
oracle-memory candidate top-1. One subject contributed 700 episodes and had only 73.71%
`anchor_4` shortlist recall, causing almost the entire fold-0 candidate failure. This
reveals a subject-frequency imbalance that episode-weighted training and averages do
not handle.

## Interpretation

V11 establishes a real safety/deletion tradeoff under unseen-subject calibration. The
gate retains current facts perfectly and unrelated facts at about 99%, while still
deleting 83% of stale facts and strongly beating matched random deletion. It does not
support the registered robust-release claim because deletion is too heterogeneous.

The result also narrows the next mechanism change. A wider candidate shortlist is
needed for the single high-frequency ambiguous subject, and the gate should optimize
subjects rather than episodes. Calibration should enforce a subject-clustered risk
bound and maximize deletion inside that safety set.

## Runtime and resources

- Wall time: 2,784.49 s (46.41 min).
- Peak CUDA allocated/reserved: 7.87/8.12 GiB.
- OOF decisions: 239,040.
- Matched controller/random deletions: 6,954 each.

## Artifacts

- `summary.json`: registered OOF result and gate.
- `training.json`: fold grids, thresholds, hashes, timing, and resources.
- `training_manifest.json`: immutable protocol and access locks.
- `mquake_robust_crossfit_gates.pt`: five-fold, three-seed controller artifact.
- `audit.json`: independent audit result.
- `diagnostic.json`: read-only failure attribution.
- `result.log`, `audit.log`, timestamps, and terminal status.

Large raw OOF JSONL files remain on `hku-gpu2` under
`/home/ctj/cbf_ttt_mquake_robust_calibration_v11/result` and are hash-locked by
`training.json`.

## Next experiment

V12 will use nested subject-disjoint candidate training, a fixed wider shortlist,
subject-equal gate loss, and subject-clustered risk-controlled calibration. It will use
only MQuAKE-T train+tune. MQuAKE-CF remains a one-time external confirmation dataset
and is opened only after the V12 OOF gate passes.
