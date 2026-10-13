# MQuAKE Temporal Transfer V7 Preregistered Plan

## Question

Does the factorized forgetting policy learned on constructed ZsRE sessions transfer,
without retraining or threshold selection, to real temporal knowledge changes?

## Data and split lock

- Source: official `MQuAKE-T.json` from `princeton-nlp/MQuAKE`.
- The raw file hash is recorded in the generated manifest.
- All cases for one edited subject remain in one split. A deterministic greedy
  assignment targets 60% train, 20% calibration, and 20% development cases.
- V7 reads only the entity-disjoint development split. Train and calibration are
  reserved for a later adaptation experiment if zero-shot transfer fails.
- MQuAKE-CF, LongMemEval-S, and LoCoMo are not read or scored.

## Frozen method

- Qwen3-4B checkpoint after the 1B-token pretraining run.
- V3 factorized scope resolver, with values excluded from candidate ranking.
- V5 nested cross-fit calibrated forgetting ensemble and its saved thresholds.
- No MQuAKE parameter fitting, refitting, threshold tuning, or model selection.

## Natural session construction

Each MQuAKE-T case becomes sessions of 16, 32, 48, and 64 events. The old
temporal fact is inserted at one quarter of the session and its real update at
three quarters. Other hops in the original MQuAKE chain are retained as natural
context. Two same-relation, different-subject events form hard-retain negatives;
the remaining events are entity-distinct distractors. Every memory event contains
four units, matching the previously frozen sparse-session protocol.

## Primary preregistered gate

- shortlist recall >= 90%; candidate top-1 >= 80%;
- current memory retention >= 97%; non-target retention >= 95%;
- hard-retain accuracy >= 95%;
- stale deletion and final pair-state accuracy >= 70%;
- the subject-clustered 95% bootstrap lower bound over an exact deletion-count
  matched random policy is greater than zero.

If every check passes, preregister one-time LoCoMo confirmation. If any check
fails, fit on MQuAKE train, calibrate on tune, keep dev locked, and report both
zero-shot and adapted results.
