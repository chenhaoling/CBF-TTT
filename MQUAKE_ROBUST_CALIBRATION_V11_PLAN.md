# MQuAKE Robust Calibration V11 Plan

## Motivation

V10 reached 96.12% strict pair state and 99.87% current retention on unseen MQuAKE-T
subjects, but non-target and hard-retain safety fell from about 98% on tune to about
92% on dev. Candidate accuracy remained 99.67%. The next experiment changes only
calibration robustness; it does not alter the `anchor_4` resolver or inspect V10 dev
examples to select a threshold.

## Stage A: subject-level OOF calibration

Use only the existing MQuAKE-T train+tune subjects. Construct five deterministic outer
subject folds. For each outer fold and three seeds:

1. train the same rank-16 BCE gate for 50 epochs on three folds;
2. calibrate its threshold on the next fold with stricter 99% current, non-target, and
   hard-retain constraints;
3. do not refit after calibration;
4. evaluate the untouched outer fold by complete causal replay.

The V9 `anchor_4` resolver, Qwen3-4B projections, feature set, optimizer, epochs, and
threshold grid remain fixed. Report pooled OOF metrics, per-fold metrics, threshold
dispersion, and subject-clustered intervals.

The OOF gate requires candidate top-1 >=99%, current retention >=99%, non-target and
hard-retain >=97%, stale deletion and pair state >=90%, and minimum-fold pair state
>=85%. V10 dev is not rescored and is not used to choose any V11 option.

## Stage B: independent safety confirmation

Only if Stage A passes, freeze the ensemble and evaluate official MQuAKE-CF-3k-v2 once
as an independent counterfactual multi-hop stress test. Require candidate top-1 >=95%,
current/non-target/hard-retain >=95%, pair state >=85%, and a positive clustered
matched-random confidence lower bound. This dataset cannot establish natural temporal
generalization; it checks that conservative calibration is not specific to the 86
MQuAKE-T scopes.

LoCoMo remains sealed until both stages pass. If Stage A fails, stop threshold tuning
and redesign the gate as an explicit constrained or conformal safety predictor.
