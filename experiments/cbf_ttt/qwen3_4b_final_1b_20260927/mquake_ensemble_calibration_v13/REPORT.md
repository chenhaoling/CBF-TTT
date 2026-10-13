# MQuAKE-T Final-Ensemble Calibration V13 Report

## Question

V13 tested the specific V12 failure hypothesis that separately calibrating three Gate
members and then applying a fixed ensemble threshold caused the low deletion rate.
Every V12 candidate and Gate weight was frozen. V13 instead took the median raw Gate
logit and causally calibrated one final threshold per outer fold.

## Integrity and resources

- The four consumed V12 artifacts matched their pre-registered SHA-256 values.
- 5,976 OOF episodes and 239,040 actions used the unchanged subject-disjoint folds.
- All five calibration selections were feasible under the registered cluster-safety
  constraints.
- MQuAKE-CF, V10 development, LongMemEval-S, and LoCoMo remained inaccessible.
- The independent audit passed all seven metric, threshold, identity, deletion-count,
  raw-hash, source-hash, and access-lock checks.
- Training/evaluation took 4,015.68 seconds; the full runner including audit completed
  between 17:43:34 and 18:50:47 +08:00. Peak CUDA allocated/reserved memory was
  7.867/8.115 GiB.

## Registered result

| Metric | V12 | V13 | Change |
|---|---:|---:|---:|
| Strict pair state | 37.015% | 37.048% | +0.033 pp |
| Stale deletion | 37.182% | 37.216% | +0.033 pp |
| Current retention | 99.816% | 99.799% | -0.017 pp |
| Non-target retention | 99.506% | 99.493% | -0.013 pp |
| Hard-retain accuracy | 99.724% | 99.699% | -0.025 pp |
| Subject-macro pair | 44.661% | 44.724% | +0.062 pp |

V13-minus-V12 pair gain was 0.033 percentage points with subject-clustered 95% CI
`[-0.050, 0.145]`. Only six of 5,976 episode pair states changed: four improved and
two regressed. Primary updates changed from 2,191 to 2,192 correct deletions. The
registered calibration hypothesis was therefore not supported.

Fold thresholds were 0.975, 0.95, 0.95, 0.60, and 0.95. Corresponding validation
pair accuracies were 13.21%, 20.32%, 46.15%, 93.39%, and 12.08%. Candidate top-1
remained 98.929%, as expected from frozen weights. The complete release gate failed
candidate top-1, stale/pair, subject-macro pair, and minimum-fold pair; all aggregate
safety and matched-random checks passed.

## Mechanistic conclusion

Raw-logit ranking is strong: pooled AUROC is 0.9933 and fold AUROCs are
0.9871--0.9969. V13 update probabilities have median 0.896, while hard-retain scores
have median 0.030. Yet a single calibration fold often selects 0.95 or 0.975, rejecting
most correct updates in another subject fold. The result refutes the threshold-
composition explanation and identifies cross-subject operating-point drift as the
dominant failure.

A read-only architecture screen found that subtracting each episode's previous-score
median improves AUROC to 0.9875--0.9978 and raises update recall at 0.5% empirical
retain false-positive rate in every fold, but the existing BCE Gate still reaches only
53.0--92.4% recall. This supports a new episode-listwise Gate trained directly for
sparse within-session ranking; it does not establish causal performance.

## Decision

V13 stops global absolute-threshold calibration. The next registered experiment is
[V14 online listwise margin](../../../../MQUAKE_ONLINE_MARGIN_V14_PLAN.md): train a
subject-balanced episode-listwise Gate on the same fit folds, make the decision from
its current score relative to prior scores in the same session, and calibrate that
shift-invariant margin by exact causal replay. External confirmation remains sealed.

Large raw JSONL traces remain on hku-gpu2 and are hash-locked by `training.json`.
