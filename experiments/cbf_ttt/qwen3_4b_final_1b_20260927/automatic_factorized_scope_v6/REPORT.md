# Automatic factorized scope V6

## Objective and method

Remove V5's supplied subject span at inference. V6 locates the shared contiguous
query/key span with maximum fold-specific IDF mass, masks that span, and combines
normalized entity overlap with the frozen V5 posterior relation compatibility.
No subject string, relation ID, chain ID, target, age, or future query enters the
resolver.

## CounterFact OOF result

| Method | Top-1 | MRR |
|---|---:|---:|
| V4 structured pair ranker | 82.93% | 90.91% |
| V5 oracle subject segmentation | **97.83%** | **98.92%** |
| **V6 automatic text-only** | **93.25%** | **96.37%** |

V6 improved over V4 by **10.32 points**, paired-bootstrap 95% CI
**[7.52,13.25]**. It was 4.59 points below V5, CI `[-6.24,-3.06]`.

The top-1 ≥95%, MRR ≥97%, and within-three-points checks failed. The gain over
V4 and positive-CI checks passed, so the complete gate failed and causal
forgetting remained stopped.

Every inner fold selected the maximum registered combination weight 8, and each
inner accuracy curve was increasing at that boundary. V7 performs one
preregistered grid-boundary extension without changing the representation. A
V7 failure permanently stops scalar combination tuning.

## Resources and claim boundary

- Runtime: 29.99 seconds on one RTX 5090.
- Peak CUDA allocated/reserved: 8.655/10.531 GiB.
- 6,490 unique automatic relation-window texts.
- Formal code commit: `d71176a`.
- LongMemEval-S and LoCoMo remained unscored.

This supports a large text-only scope gain on observed CounterFact. It does not
yet pass the frozen automatic-resolution gate or establish causal/native-memory
benefit.

