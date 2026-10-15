# Native-writer causal-ledger bridge V1

## Outcome

The preregistered single-record upper-bound pilot failed its scientific gate,
while all six execution and integrity audit checks passed. The unchanged
seven-layer `ttt_conv`/`ttt_proj` writer did not bind an explicitly repeated
key/value fact into query-readable fast weights, even when the mapping filled
one complete 4,096-token native TTT chunk.

The correct native writer reached 14.29% six-way candidate accuracy and 0%
five-token exact generation. Supplying a deliberately wrong value produced the
same truth accuracy, only 28.57% injected-wrong accuracy and no reliable NLL
separation. The registered lower confidence bounds for Correct over Wrong and
Zero were both negative. In contrast, the structured external residual reached
100% candidate accuracy and 100% exact generation on the identical queries.

## Main comparison

| Arm | Candidate accuracy | Five-token exact | Answer NLL |
|---|---:|---:|---:|
| Zero memory | 7.14% | 0.00% | 6.4356 |
| Correct native writer | 14.29% | 0.00% | 6.4082 |
| Wrong native writer | 14.29% | 0.00% | 6.4133 |
| Structured external residual | **100.00%** | **100.00%** | **0.000040** |

The 95% source-group bootstrap interval for the correct-writer NLL gain was
`[-0.0097, 0.0193]` against Wrong and `[-0.0475, 0.1051]` against Zero. The
wrong writer selected its injected value in only 28.57% of rows.

## Integrity and resources

- 14 source groups from development-only `resource_v2` data.
- Seven trained TTT layers: 0, 6, 12, 18, 24, 30 and 35.
- Native chunk length 4,096 and checkpoint TTT learning rate 0.3.
- Exactly five greedy output tokens were generated for every arm and row.
- Model loading had no missing, unexpected or mismatched parameters.
- All six audit checks and all 13 focused unit tests passed.
- Two RTX 5090 critical path: 9.06 seconds after model setup.
- Peak allocated/reserved memory: 10.136/10.684 GiB.

## Decision and claim boundary

The frozen protocol requires this failure to stop the unmodified native-writer
branch. Sparse routing or a forgetting controller cannot recover information
that the writer did not encode. Further scale, threshold or controller tuning
on `ttt_conv`/`ttt_proj` is therefore not justified by these data.

This result does not invalidate the independently confirmed causal ledger. It
localizes the remaining end-to-end gap to content binding: structured,
addressable value residuals are readable, while the checkpoint's original
dense writer is not a factual key/value writer under this favorable test. The
next registered experiment is the still-sealed lifecycle OOD evaluation of the
structured causal ledger, followed by natural-text scope and lifecycle event
extraction.
