# Scoped Delta Ranker V4

## Objective

Test whether explicit subject and relation factorization breaks the approximately
80% candidate-memory ceiling observed in V2/V3 before training another causal
forgetting gate. CounterFact remains observed development data and all reported
predictions are five-fold session-disjoint OOF.

## Data and method

The new corpus contains 4,990 sequential decisions: 1,030 ZsRE and 785
CounterFact local-update groups plus retain decisions reserved for later gate
work. Candidate pairs are audited as same-chain positives,
same-subject/different-chain hard negatives, or different-subject negatives.

The frozen final Qwen3-4B checkpoint encodes the existing full query/key pair
and a second pair in which supplied subject spans are replaced by `<ENTITY>`.
The primary linear residual ranker receives both representations, relation
cosines/lexical features, and a subject-match bit. Five outer folds, three
initialization seeds, and inner-fold epoch selection were fixed in advance.

## CounterFact OOF result

| Arm | Top-1 | MRR | Same-subject pair | All different subjects |
|---|---:|---:|---:|---:|
| V2 | 79.62% | — | — | — |
| No relation features | 80.89% | 89.72% | 83.82% | 93.75% |
| No subject bit | 82.55% | 90.68% | 85.61% | 93.62% |
| **Structured full** | **82.93%** | **90.91%** | **85.86%** | **93.88%** |

The primary arm improved top-1 over V2 by **3.31 points**, with event-paired
bootstrap 95% CI **[0.64, 5.86]**. It corrected 68 V2 errors but introduced 42
new errors. Relation-focused features contributed about 2.04 points relative
to the no-relation ablation; the subject bit contributed about 0.38 points.

The primary still made 95 same-subject/different-relation errors and 39
different-subject errors. A post-run label audit found that all 95 same-subject
errors selected a different CounterFact relation, directly locating the
remaining bottleneck.

## Gate and decision

Only the positive-CI and same-subject ≥85% checks passed. Top-1 ≥88%, MRR ≥93%,
gain ≥5 points, and different-subject ≥95% failed. The complete development
gate therefore failed and no causal deletion gate was trained.

The statistically positive V4 gain supports relation factorization, while the
remaining error structure rejects this frozen pair representation as
sufficient. V5 tests subject-disjoint relation-supervised text representations;
relation IDs are training targets only and never inference inputs.

## Resources and claim boundary

- Runtime: 158.52 seconds on one RTX 5090.
- Peak CUDA allocated/reserved: 8.602/10.547 GiB.
- Frozen Qwen3-4B checkpoint; 7,763 unique texts and 28,577 pair texts.
- Formal code commit: `172f646`.
- LongMemEval-S and LoCoMo remained unscored.

This result supports a development-set representation gain and an error
diagnosis. Supplied subject spans are an oracle segmentation diagnostic; it does
not establish automatic entity extraction, native fast-weight deletion, causal
query improvement, or sealed benchmark performance.
