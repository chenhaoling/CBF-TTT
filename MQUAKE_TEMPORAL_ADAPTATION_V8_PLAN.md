# MQuAKE Temporal Adaptation V8 Preregistered Plan

## Question

Can the factorized resolver and selective forgetting rule learn natural temporal
scope from MQuAKE train and generalize to unseen edited subjects?

## Fixed protocol

- Use the already hashed MQuAKE-T subject-disjoint train/tune/dev corpus.
- Train three rank-16 factorized scope resolvers on train for exactly 50 epochs.
- Train three rank-16 BCE deletion heads on causal train rows for exactly 50 epochs.
- Use tune only to calibrate one threshold for each deletion head. Do not refit after
  calibration. Candidate values, age, unit position, subject IDs, and relation IDs are
  excluded from candidate ranking.
- The Qwen3-4B checkpoint and three contextual projections stay frozen.
- MQuAKE-CF, LongMemEval-S, and LoCoMo remain unused.

MQuAKE dev was previously scored only by the frozen zero-shot V7 method. V8 code,
features, epochs, thresholds, and gates are frozen before its adapted result is read.
No V7 dev label or metric is used in V8 fitting or threshold calibration.

## Conditional tune prerequisite

Before loading dev episodes, the three-member calibrated ensemble must reach on tune:

- shortlist recall >= 90% and candidate top-1 >= 80%;
- current retention >= 97%, non-target retention >= 95%, hard-retain >= 95%;
- pair-state accuracy >= 70%.

If any check fails, V8 stops and dev is absent.

## Development gate

- shortlist recall >= 90% and candidate top-1 >= 85%;
- current retention >= 97%, non-target retention >= 95%, hard-retain >= 95%;
- stale deletion and pair-state accuracy >= 80%;
- every length has pair state >= 70%;
- the subject-clustered 95% bootstrap lower bound over exact deletion-count matched
  random is greater than zero.

A full pass freezes the adapted method for a separately preregistered LoCoMo
confirmation. Failure triggers error analysis by candidate versus action failure and
does not permit tuning on dev.
