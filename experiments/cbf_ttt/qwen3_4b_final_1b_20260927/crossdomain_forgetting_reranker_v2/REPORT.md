# Cross-domain forgetting reranker V2

## Purpose

V2 used all four non-held-out CounterFact folds to fit each outer ranker, trained
a class-balanced semantic-conflict gate, and selected thresholds from four inner
OOF folds. It kept the Qwen backbone frozen and the current-input write gate at
one.

## Results

| Metric | V2 causal OOF | V2 oracle state | V1 causal OOF |
|---|---:|---:|---:|
| Query accuracy | 46.79% | — | 49.66% |
| Overall action | 78.26% | 81.84% | 84.93% |
| Candidate top-1 | — | **79.62%** | 78.47% |
| Initial retain | 93.38% | 90.31% | 98.73% |
| Hard retain | 80.25% | 79.11% | 92.36% |
| Update | 61.15% | **78.22%** | 63.69% |

The stronger outer-training ranker improved candidate top-1 by 1.15 points and
placed the oracle-state update ceiling above the 75% target. However, the
balanced threshold objective selected 0.20–0.25 in every fold. Its false
deletions changed later candidate sets: causal action fell 3.58 points below the
oracle-state estimate, and query accuracy fell 2.69 points relative to V1 (95%
CI `[-3.24,-2.11]`). Action fell 6.67 points, CI `[-8.49,-4.84]`. Every complete
development-gate check except candidate top-1 failed.

The independent audit passed all checks, including raw trajectory hashes,
oracle-event recomputation, fold identity, checkpoint hashes, paired bootstrap
comparisons, source manifests, and sealed LongMemEval-S/LoCoMo flags. Runtime was
315.24 seconds on one RTX 5090, with 9.026/12.551 GiB peak allocated/reserved
memory.

## Interpretation

V2 isolates exposure bias as the next bottleneck. Static oracle-state metrics
make aggressive forgetting appear viable, but an early false deletion is
irreversible and changes all later states. The candidate ranker is above its
registered floor; more scalar tuning against the same oracle objective is not a
principled fix.

The saved inner grids nevertheless contain safer thresholds that meet the
hard-retain/update constraints in four of five folds. The next diagnostic keeps
all V2 weights fixed and selects thresholds by constraint feasibility, then
performs full causal replay. Failure of that diagnostic triggers on-policy state
collection and causal threshold selection.

