# Token-span relation scope V11 report

## Result

V11 passed every preregistered development check on 785 subject-disjoint
CounterFact local-update groups.

| Metric | Result |
| --- | ---: |
| Candidate top-1 | 97.20% |
| MRR | 98.60% |
| Gain over V10 | +2.55 points |
| Paired-bootstrap 95% CI versus V10 | [1.02, 4.20] points |
| Gap to oracle-segmented V5 | -0.64 points |
| Span exact match | 93.27% |
| Span token F1 | 98.14% |
| Runtime | 1,276.58 s |
| Peak allocated / reserved GPU memory | 8.39 / 9.36 GiB |

All five folds selected epoch 15, threshold 0.3, and the registered `overlap8`
arm. Fold span exact match ranged from 92.12% to 95.09%. Two of 65,433 raw
training examples were truncated before their supervised span; no evaluation
example was missing.

## Mechanistic evidence

Relation-only ranking after predicted masking reached only 79.24%, while
`overlap8` reached 97.20%. The exact-match hard filter reached 93.76%. The result
therefore supports a factorized mechanism: a soft subject-overlap term preserves
entity identity while subject masking exposes the relation pattern. Neither term
is sufficient by itself.

The gain over V10 is statistically positive and closes 80% of V10's gap to the
oracle-segmented V5 result. This confirms the V10 error audit: the useful role of
token-level entity extraction is relation de-identification, rather than adding
another whole-text entity score.

## Scope of the claim

CounterFact remains observed development-only. The result establishes automatic
candidate localization, not causal forgetting or native dense fast-weight
forgetting. V12 will freeze the V11 resolver, train only a retain/local gate on
outer-training folds, and replay the full sequential memory trajectories.
LongMemEval-S and LoCoMo remain sealed.
