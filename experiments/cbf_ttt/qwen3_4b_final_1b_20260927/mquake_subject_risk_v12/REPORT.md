# MQuAKE-T Subject-Risk V12 Report

## Question

V12 tested whether subject-balanced candidate/Gate training, a fixed anchor-16
shortlist, and cluster-risk calibration could preserve V10's deletion efficacy while
meeting subject-level safety requirements. It used only MQuAKE-T train+tune in five
subject-disjoint outer folds. V10 development, MQuAKE-CF, LongMemEval-S, and LoCoMo
were not accessed.

## Integrity and resources

- 5,976 OOF episodes and 239,040 actions were evaluated exactly once in their outer
  fold.
- Runtime was 10,366.71 seconds (172.78 minutes).
- Peak CUDA allocated/reserved memory was 7.867/8.115 GiB.
- The pair cache held 1,000,000 entries and recorded 195,590,908 hits, 16,464,955
  misses, and 15,464,955 evictions.
- The independent audit passed all seven metric, threshold, identity, deletion-count,
  output-hash, source-hash, and access-lock checks.

## Registered results

| Metric | V12 OOF |
|---|---:|
| Anchor-16 shortlist recall | 100.00% |
| Candidate top-1 | 98.93% |
| Subject-macro candidate top-1 | 97.34% |
| Stale deletion | 37.18% |
| Strict pair state | 37.01% |
| Current retention | 99.82% |
| Non-target retention | 99.51% |
| Hard-retain accuracy | 99.72% |
| Subject-macro pair | 44.66% |
| Subject-macro safety | 98.32% |
| Safety one-sided 95% LCB | 96.79% |

Fold pair-state accuracies were 13.38%, 20.32%, 46.15%, 93.06%, and 12.08%.
Against an exact deletion-count matched random policy, V12 gained 34.84 percentage
points with subject-clustered 95% CI `[23.19, 49.46]`.

The registered gate failed candidate top-1 (98.93% versus 99%), stale deletion and
pair state (37% versus 90%), subject-macro pair (44.66% versus 85%), and minimum-fold
pair (12.08% versus 85%). It passed all aggregate safety checks, anchor recall,
subject-macro candidate top-1, and the matched-random confidence check.

## Failure attribution

The read-only diagnostic did not call the model or select a threshold. Among 5,976
primary updates:

- 2,191 (36.66%) correctly deleted the target;
- 3,703 (61.96%) selected the correct target but rejected deletion;
- 21 (0.35%) deleted a wrong candidate;
- 61 (1.02%) selected a wrong candidate and rejected deletion.

The deployed score separates updates from retain events well: pooled AUROC is 0.9898
and every fold lies between 0.9871 and 0.9968. Primary-update probability has median
0.346, whereas hard-retain probability has median 0.0021 and 90th percentile 0.0186.
The dominant failure is therefore the composed decision threshold. V12 separately
calibrated each member, centered each logit at its own high threshold, took the median,
and then used a fixed 0.5 threshold on that composed score.

## Conclusion and next experiment

V12 provides strong evidence that subject balancing and wider candidate retrieval
solve most candidate and safety errors, but its final calibration discards most useful
deletions. It does not establish an effective deployed forgetting policy. The locked
next experiment is [V13 final-ensemble calibration](../../../../MQUAKE_ENSEMBLE_CALIBRATION_V13_PLAN.md):
freeze every V12 weight, ensemble raw Gate logits first, and causally calibrate one
final threshold on an independent fold.

Large OOF JSONL traces remain on hku-gpu2 and are cryptographically locked by
`training.json`; this directory contains the compact summary, training metadata,
checkpoint, audit, diagnostic, logs, timestamps, and terminal state.
