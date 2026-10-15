# Qwen natural scope-transfer repair V2

## Protocol

V2 repairs the V1 retrieve/defer transfer failure without changing the writer,
value residual, memory lifecycle, candidates or Qwen checkpoint.  It trains a
linear gate on the frozen pair-plus-geometry feature with five-fold session
OOF predictions.  Natural wrong-top examples and deterministic in-fold
cross-session no-match queries supervise safe deferral.  Each of the three
strata receives equal total loss weight.

The search, loss, hyperparameters, selection rule and release gates were
registered before producing V2 scores.  Natural development and every test or
external benchmark remained sealed.

## Audited calibration result

The selected OOF configuration used anchor `0.01`, learning rate `0.005`, 100
epochs and threshold `0.35`.

| Reader metric | Result | Gate |
|---|---:|---:|
| Correct-top recall | 89.03% | ≥90%, fail |
| Natural wrong-top specificity | 94.30% | ≥85%, pass |
| No-match specificity | 96.59% | ≥92%, pass |
| Three-stratum balanced accuracy | 93.31% | ≥91%, pass |

| End-to-end metric | V1 reader | V2 reader | Gate |
|---|---:|---:|---:|
| Candidate accuracy | 87.05% | **91.54%** | ≥90%, pass |
| Final candidate accuracy | 85.95% | **90.60%** | ≥88%, pass |
| Final generation exact | 64.24% | **72.48%** | ≥70%, pass |
| Gap to oracle reader | 11.26pp | **6.77pp** | ≤6pp, fail |

V2 improved candidate accuracy over the frozen V1 reader by 4.52 points with a
subject-group bootstrap 95% CI of `[3.31, 5.76]`.  Exact accuracy remained
98.60%; rephrase accuracy increased from 75.51% to 84.49%.  Only 11 OOF rows
retrieved an incorrect top and required new Qwen scoring; 4,408 reused an exact
oracle-equivalent residual and 725 deferred to the already scored zero arm.

The independent audit passed all eight checks: terminal state and seals,
artifact hashes, feature shapes and strata, exact preregistered search
selection, row coverage, independent metric/bootstrap recomputation and gate
recomputation.

## Decision

The experiment stopped at calibration because correct-top recall missed its
gate by 0.97 points and the oracle gap exceeded its gate by 0.77 points.  The
large, significant improvement demonstrates that the V1 reader failure was
repairable calibration transfer rather than a failure of natural fast memory.
The remaining error requires pairwise rank-one repair; lowering the threshold
would violate the registered no-match safety constraint.  Development remains
sealed for a joint key/defer ranker V3.

The full 60.9 MB feature tensor remains in the immutable remote experiment root
with SHA256 recorded in `summary.json`.  The repository archive contains the
search, OOF rows, checkpoint, summary, execution audit, logs and a compact
remote-root archive.
