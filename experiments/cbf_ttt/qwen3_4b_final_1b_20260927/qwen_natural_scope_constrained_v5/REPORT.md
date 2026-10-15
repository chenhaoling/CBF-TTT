# Qwen natural constrained scope gate V5

## Calibration protocol

V5 applies the original V4 preference order only after all four reader release
constraints are satisfied.  It reuses the immutable V4 search and V2 feature
artifact, reproduces the selected OOF logits, and permits one exact Qwen outcome
score.  The model, hyperparameter grid and scientific gates are unchanged.

The selected width-32 residual uses penalty 1.0, learning rate 0.005, 50 epochs
and threshold 0.30.

| Reader metric | Result | Gate |
|---|---:|---:|
| Correct-top recall | 95.25% | ≥92% |
| Wrong-top specificity | 88.60% | ≥85% |
| No-match specificity | 94.58% | ≥92% |
| Balanced accuracy | 92.81% | ≥92% |

| Outcome metric | Result | Gate |
|---|---:|---:|
| Candidate accuracy | **94.34%** | ≥93% |
| Rephrase candidate accuracy | **90.09%** | descriptive |
| Gap to oracle reader | **3.97pp** | ≤5.5pp |
| Final candidate accuracy | **93.22%** | ≥91% |
| Final generation exact | **77.03%** | ≥74% |

V5 improved over audited V2 by 2.79 points with subject-group bootstrap 95% CI
`[1.90, 3.76]`.  All nine calibration gates passed.  The independent audit
passed all eight checks, including input/output hashes, exact constrained
selection, OOF coverage, aggregate/bootstrap recomputation and gate logic.

This result releases one frozen natural-development reader evaluation.  It
does not by itself establish a learned-forgetting advantage or any held-out
test/external benchmark result.
