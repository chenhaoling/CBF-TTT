# MQuAKE Online Listwise Margin V14

## Motivation

V13 independently falsified the final-threshold-composition hypothesis. Replacing
member-wise calibration with one raw-logit ensemble threshold changed only six of
5,976 OOF pair states and yielded a nonsignificant +0.033-point gain. Nevertheless,
raw Gate ranking remains strong (pooled AUROC 0.9933). Absolute operating points drift
across subject folds: feasible thresholds range from 0.60 to 0.975 and validation pair
accuracy ranges from 12.08% to 93.39%.

V14 treats a local memory replacement as a sparse, session-relative conflict event.
It trains the Gate to rank the one update above retain events from the same episode
and makes the online decision from the current conflict energy relative to the
episode's preceding score baseline. An additive subject/session score shift cancels
from this margin.

## Locked data and sources

- Use MQuAKE-T train+tune only and the unchanged deterministic V11 subject folds.
- Outer fold `k` validates on `k`, calibrates on `(k+1) mod 5`, and fits on the other
  three folds.
- Reuse the audited V12 fold-specific subject-balanced anchor-16 candidate rankers.
- Keep Qwen3-4B, contextual projections, feature definitions, three seeds, rank 16,
  and 50 epochs fixed.
- V10 development, MQuAKE-CF, LongMemEval-S, and LoCoMo remain inaccessible.
- V13 is diagnostic evidence only; no V13 validation prediction is used as a V14
  training label or calibration sample.

## Gate training

For every fit episode, construct the existing oracle-memory candidate feature for
each action. Exactly one action is the temporal update. A rank-16 MLP scores every
action and is optimized with episode-listwise cross entropy. Episodes receive weights
so every fit subject contributes equal total loss. Feature normalization is also
subject-equal. No action position, episode length, subject identity, fold identity, or
future event is an input feature.

After fitting, store the fit-score median and interquartile range for deterministic
member-scale normalization. Three independently seeded members form a median score
ensemble. Models are never refit after calibration.

## Online rule

At action `t`, let `z_t` be the median normalized member score and let
`b_t = median(z_1, ..., z_{t-1})`. The causal conflict margin is

`m_t = z_t - b_t`.

The first scored action has no baseline and cannot delete memory. Every later action
deletes only the candidate selected by the frozen anchor-16 resolver when
`m_t >= tau`. The history contains scores observed before the current action and is
reset between episodes. It does not use labels, future scores, or whether earlier
actions were correct.

## Registered calibration

For every fixed margin threshold in

`-1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20`,

replay complete calibration sessions. A threshold is feasible only when aggregate
current/non-target/hard-retain are each at least 97%, subject-macro safety is at least
98.5%, and its one-sided 95% lower confidence bound is at least 97%. Select maximum
subject-macro pair, then aggregate pair, stale deletion, safety lower bound, and
proximity to margin 2. If none is feasible, maximize safety and record infeasibility.

## Registered OOF gate

Use the unchanged V12/V13 release checks:

- shortlist recall >= 99%;
- candidate top-1 >= 99% and subject-macro top-1 >= 97%;
- current retention >= 99%;
- non-target and hard-retain >= 97%;
- subject-macro safety lower bound >= 95%;
- stale deletion and strict pair state >= 90%;
- subject-macro pair >= 85% and minimum-fold pair >= 85%;
- subject-clustered matched-random gain lower bound > 0.

The mechanism hypothesis is supported when all pair and safety checks pass. If only
candidate top-1 remains below 99%, freeze V14 and proceed to a candidate-only relation
scope experiment. If pair checks still fail, stop this feature/model family and move
to an explicit abstaining relation-conditional controller. MQuAKE-CF opens once only
after the complete release gate passes.

An independent auditor must reproduce threshold selection, folds, metrics, clustered
comparisons, output/source/checkpoint hashes, online-history locks, and all external-
data access locks.
