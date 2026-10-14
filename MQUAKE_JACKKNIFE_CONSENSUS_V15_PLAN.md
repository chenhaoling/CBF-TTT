# MQuAKE Jackknife Consensus V15 Plan

## Motivation

V14 established that a causal within-session median margin improves OOF
pair-state accuracy from 37.05% to 49.87% while retaining 99.87% of current
memory and 99.85% of non-target memory.  The result is not releasable because
fold pair accuracy ranges from 1.34% to 94.73%.

The registered post-hoc diagnostics localize the failure:

- relation support is not the cause: roughly 99% of episodes belong to a
  relation with at least 500 fit episodes;
- other-fold performance for the same relation predicts current-fold success
  at only 46.12%;
- MAD, IQR, and mean/standard-deviation normalization reduce the safe pair
  proxy substantially;
- the original causal median margin remains the strongest feature, but a
  threshold calibrated on a single subject fold drifts across unseen subjects.

V15 therefore changes calibration and aggregation only.  It does not add a
relation ID, subject ID, position, length, future score, or new gate feature.

## Registered method

For each outer subject fold:

1. Hold the outer fold out for final OOF validation.
2. For each of the four remaining folds, train a three-seed listwise gate
   ensemble on the other three folds.
3. On that ensemble's held-out inner fold, collect causal median margins under
   a retain-all trace.  Treat retain actions and update actions whose selected
   candidate is wrong as non-executable actions.
4. Form threshold candidates from empirical upper non-executable quantiles at
   target FPRs `0.0005`, `0.001`, `0.0025`, and `0.005`.
5. Evaluate those four candidates with exact causal replay on the inner fold.
   Select the feasible candidate with the best subject-macro pair accuracy;
   feasibility uses the unchanged V14 safety constraints.
6. On the outer validation fold, use all four independently calibrated inner
   ensembles.  Every ensemble keeps its own causal score history.  Forget only
   when at least three of four ensembles vote to forget the shared candidate.

There is no refit after inner calibration.  The four deployed voters are
exactly the four trained and calibrated inner ensembles.

## Frozen protocol

- Qwen3-4B step-81381 checkpoint and tokenizer are unchanged.
- V12 fold-specific anchor-16 candidate rankers are unchanged.
- The MQuAKE-T train+tune subject folds are unchanged.
- Gate loss, feature vector, rank, epochs, and per-member fit IQR
  normalization are unchanged from V14.
- External MQuAKE-CF, LongMemEval-S, and LoCoMo labels remain sealed.
- Final evaluation uses exact causal replay; the V14 fixed-trace screen is not
  reported as a policy result.

## Release gate

V15 uses the unchanged V12-V14 release gate:

- stale deletion and pair-state accuracy at least 90%;
- subject-macro pair accuracy at least 85%;
- every outer fold pair accuracy at least 85%;
- current retention at least 99%;
- non-target and hard-retain accuracy at least 97%;
- subject-cluster safety lower confidence bound at least 95%;
- matched-random gain lower confidence bound above zero.

If V15 fails, the listwise gate and scalar-threshold feature family stops.  A
subsequent experiment must supervise forgetting from the real fast-weight
readout trajectory rather than add another calibration heuristic.
