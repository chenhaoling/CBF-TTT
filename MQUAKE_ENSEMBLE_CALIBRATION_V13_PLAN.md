# MQuAKE Final-Ensemble Calibration V13

## Motivation

V12 retained current/non-target/hard memories at 99.82/99.51/99.72%, but reached
only 37.01% strict pair state. Its read-only diagnostic found that 61.96% of all
updates selected the correct stale memory and then rejected deletion. This is not a
weak ranking signal: the deployed V12 ensemble score has pooled AUROC 0.9898 and
per-fold AUROC 0.9871--0.9968. V12 calibrated three members separately, subtracted
three independently selected logit thresholds, took their median, and then applied a
fixed 0.5 ensemble threshold. V13 isolates whether that composition caused the
failure.

## Immutable source

V13 reuses the audited V12 models without fitting or modifying any weight. The locked
V12 SHA-256 values are:

- `summary.json`: `76b61afa5e10affea808e06a9ea82cd489f943f65db379e962c2f3b618aa4cc4`
- `training.json`: `8444427160e6fe8d9721efbf527fa9569782d057df4f58cdfbb1948e5f53392e`
- `training_manifest.json`: `0450f440f0e39ef6a486979d508ebf8b94db90514e47b124e118d55a790d08d5`
- `mquake_subject_risk_crossfit.pt`: `d61386baebac1bd02352db5eeb1a0fe7ef00a894c130f1b408536bc5c995d41c`

The deterministic V11/V12 subject folds, fold-specific anchor-16 candidate rankers,
three fold-specific Gate artifacts, Qwen3-4B 1B-token checkpoint, contextual
transforms, and feature definitions remain fixed. V10 development, MQuAKE-CF,
LongMemEval-S, and LoCoMo remain inaccessible.

## Registered intervention

For a candidate feature, each frozen Gate emits its raw probability. V13 clips it only
for numerical stability, converts it to a logit, takes the median logit, and maps that
median back to a probability. It does not subtract the old per-member V12 thresholds.
One final threshold is then applied to this deployed ensemble score.

For outer validation fold `k`, `(k+1) mod 5` is the calibration fold. Every threshold
is evaluated by complete causal session replay with the fold-specific frozen
candidate resolver and frozen Gate ensemble. No model is refit after calibration.

The single pre-registered threshold grid is:

`0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3,
0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99`.

A threshold is feasible only when aggregate current, non-target, and hard-retain are
each at least 97%, subject-macro safety is at least 98.5%, and its one-sided 95%
normal lower bound is at least 97%. Among feasible thresholds, select maximum
subject-macro pair state, then aggregate pair state, stale deletion, safety lower
bound, and proximity to 0.5. If no threshold is feasible, select maximum safety lower
bound and record infeasibility.

## Registered OOF gate

V13 uses the unchanged V12 release gate:

- shortlist recall >= 99%;
- candidate top-1 >= 99%;
- subject-macro candidate top-1 >= 97%;
- current retention >= 99%;
- non-target and hard-retain >= 97%;
- subject-macro safety lower bound >= 95%;
- stale deletion and strict pair state >= 90%;
- subject-macro pair state >= 85%;
- minimum-fold pair state >= 85%;
- subject-clustered matched-random gain lower bound > 0.

Candidate top-1 was 98.929% in V12 and is expected to remain near that value. It is
kept as a strict failure rather than silently relaxing a nearly met criterion.

## Decision rule

1. If pair/stale, safety, subject-macro, and every-fold checks pass, the calibration
   failure hypothesis is supported. If candidate top-1 is the only failure, the next
   experiment targets relation-explicit candidate ranking while freezing V13.
2. If ranking remains high but pair/stale still fail, inspect causal replay and then
   stop scalar global calibration; move to a pre-registered relation-conditional or
   abstaining calibration model.
3. If safety fails, do not open an external benchmark. Introduce an explicit abstain
   action and subject-frequency shrinkage on development data.
4. MQuAKE-CF is opened exactly once only after the complete registered gate passes.

An independent auditor must recompute fold assignments, calibration selection, OOF
metrics, subject clusters, matched-random confidence interval, all hashes, and all
data-access locks.
