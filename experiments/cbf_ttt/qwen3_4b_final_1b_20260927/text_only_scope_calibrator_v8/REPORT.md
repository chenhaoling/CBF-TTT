# Text-only scope calibrator V8

## Objective

Replace V7's scalar entity/relation weight with a nested-OOF nonlinear
calibrator over fixed text-only evidence. No subject, relation, chain, value,
age, position, or future-query metadata is available at inference.

## Result

| Method | Top-1 | MRR |
|---|---:|---:|
| V4 | 82.93% | 90.91% |
| V7 | 93.63% | 96.76% |
| **V8** | **94.01%** | **96.97%** |

V8 improved over V4 by 11.08 points, CI `[8.28,13.89]`, but improved over V7
by only 0.38 points, CI `[-0.76,1.66]`. It failed top-1 ≥95%, MRR ≥97%, and
both preregistered V7-gain checks. The complete gate failed.

The fixed heuristic feature family is now stopped. The next architecture must
learn entity identity directly from query/key text using same-relation,
different-entity hard negatives.

## Claim boundary

This is observed CounterFact development evidence. LongMemEval-S and LoCoMo
remain unscored. The result does not support causal forgetting or native
fast-weight claims.

