# MQuAKE Anchor Forgetting V10 Preregistered Plan

## Method

Freeze V9's `anchor_4` resolver: shortlist four active sessions by mean cosine between
their declared scope-anchor units and the incoming scope anchor, then use the three
V8 train-fitted listwise heads to rerank only that shortlist. Candidate values, age,
unit position beyond the declared anchor field, subject IDs, and relation IDs are not
used for retrieval.

Train three fresh rank-16 BCE deletion heads on MQuAKE train causal states for exactly
50 epochs. Calibrate one threshold per head on tune by complete causal replay under
the existing safety constraints. Do not refit after calibration. Qwen3-4B and its
three contextual projections remain frozen.

## Conditional tune prerequisite

- shortlist recall and candidate top-1 >= 99%;
- current retention >= 97%, non-target retention >= 95%, hard-retain >= 95%;
- pair-state accuracy >= 75%.

Dev episodes are loaded only if all checks pass.

## Development gate

- shortlist recall and candidate top-1 >= 99%;
- current retention >= 97%, non-target retention >= 95%, hard-retain >= 95%;
- stale deletion and strict pair state >= 85%;
- every session length has pair state >= 80%;
- the subject-clustered 95% confidence lower bound over exact deletion-count matched
  random is above zero.

A full pass is required before preregistering LoCoMo confirmation. MQuAKE-CF and
LongMemEval-S are not used.
