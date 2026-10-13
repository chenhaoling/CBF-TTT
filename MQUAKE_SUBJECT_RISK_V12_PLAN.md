# MQuAKE Subject-Balanced Risk-Controlled Forgetting V12

## Motivation

V11 passed all safety floors but reached only 83.08% OOF pair state. Read-only
attribution found that 13.92% of all updates selected the correct stale memory but were
rejected by the calibrated gate. Candidate errors were concentrated in one subject
with 700 episodes and 73.71% `anchor_4` recall. V12 addresses these two measured
failure modes without using V10 development or any external confirmation labels.

## Locked data and access

- Use MQuAKE-T train+tune only.
- Reuse the deterministic five subject folds from V11.
- For outer fold `k`, validate on `k`, calibrate on `(k+1) mod 5`, and fit on the
  remaining three folds.
- V10 development, MQuAKE-CF-3k-v2, LongMemEval, and LoCoMo stay unscored.
- Qwen3-4B checkpoint, contextual projections, feature definitions, rank 16, 50
  epochs, and three seeds remain fixed.

## Candidate change

Train the three candidate rankers inside each outer fold using only fit subjects and a
subject-equal listwise loss. Use a fixed `anchor_16` shortlist. Sixteen is declared
before the run and is not selected from outer-fold performance. Report aggregate and
subject-macro shortlist recall/top-1, per-fold results, and the previously problematic
high-frequency subject as an ordinary outer-fold sample.

## Gate change

Train each gate with subject-equal BCE: every fit subject has the same total loss
weight, followed by the existing square-root class reweighting. Candidate and gate
models are never refit after calibration.

For every fixed threshold in
`0.05, 0.10, ..., 0.95, 0.975, 0.99, 0.995`, replay complete calibration sessions.
For each calibration subject compute the minimum of current retention, non-target
retention, and hard-retain accuracy. A threshold is feasible when:

1. aggregate current, non-target, and hard-retain are each at least 97%;
2. subject-macro safety is at least 98.5%;
3. its one-sided 95% normal lower confidence bound is at least 97%.

Among feasible thresholds choose maximum subject-macro pair state, then aggregate pair
state, stale deletion, safety lower bound, and proximity to 0.5. If none is feasible,
choose the maximum safety lower bound and record infeasibility. Three calibrated seeds
form the same median logit-margin ensemble used in V11.

## Registered OOF gate

- shortlist recall >= 99%;
- candidate top-1 >= 99%;
- subject-macro candidate top-1 >= 97%;
- current retention >= 99%;
- non-target retention and hard-retain >= 97%;
- subject-macro safety lower bound >= 95%;
- stale deletion and pair state >= 90%;
- subject-macro pair state >= 85%;
- minimum-fold pair state >= 85%;
- subject-clustered matched-random gain lower bound > 0.

All metrics, threshold selection, fold membership, source hashes, raw row hashes, and
access locks require an independent audit.

## Decision rule

- If the complete OOF gate passes, run matched ablations on the same development
  folds, freeze the final ensemble, and then open MQuAKE-CF once.
- If candidate gates pass but deletion still fails, stop this feature family and move
  to a relation-explicit scope representation.
- If safety fails, stop external release and introduce an abstaining two-stage risk
  head; do not search more thresholds on these folds.
