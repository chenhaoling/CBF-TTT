# Causal-ledger development V2

## Outcome

The run completed and passed all seven independent execution-audit checks. It
passed ten of eleven preregistered method checks, so the development release
gate failed and confirmation/OOD remained ungenerated.

The only failed check was the 95% five-token exact floor for both current and
anchor queries. Candidate selection was 100%, exact agreement with the
semantic oracle was 100%, and every lifecycle family and length-bucket gain
over the strongest non-oracle comparator was non-negative. This localizes the
remaining development failure to the scale-32 contextual generation margin.

## Main results

| Policy | Candidate accuracy | Five-token exact |
|---|---:|---:|
| No memory | 8.01% | 8.01% |
| Retain latest | 92.66% | 81.42% |
| Global boundary clear | 47.69% | 44.22% |
| Last 1 | 47.69% | 44.22% |
| Last 2 | 50.55% | 46.08% |
| Dense full | 63.93% | 7.03% |
| Full causal ledger | 100.00% | 91.34% |
| Semantic oracle | 100.00% | 91.34% |

`retain_latest` was the strongest non-oracle comparator. The primary exact
gain was 9.9206 percentage points with a 20,000-resample source-group bootstrap
95% interval of `[8.6806, 11.1855]` points. The answer-NLL improvement was
1.5689 with interval `[1.3696, 1.7714]`.

Primary exact by role was 91.37% current, 93.45% historical and 89.21% anchor;
absence accuracy was 100%. Family exact ranged from 88.89% to 93.66%, and
length-bucket exact ranged from 90.48% to 91.77%.

## Protocol and resources

- Final Qwen3-4B checkpoint after the 1B-token pretraining run.
- Frozen scale 32 and exactly five generated tokens.
- 448 source groups, 896 episodes and 8,064 query-policy rows.
- Seven lifecycle families and sequence lengths 8/12/16.
- Two RTX 5090 GPUs; critical path 1,392.49 seconds (23.21 minutes).
- Peak allocated/reserved CUDA memory 7.666/7.693 GiB.
- Model loading reported no missing, unexpected or mismatched keys.

The independent audit recomputed source coverage, row alignment, policy state,
generation traces, aggregate metrics, bootstrap intervals, gate decisions,
hashes and split sealing. `confirmation_generated=false` and
`ood_generated=false` are present in both shard resources and the summary.

## Decision

The positive policy effect is retained as development evidence, but this run
does not release confirmation because the registered role floor failed. The
next and only scalar repair is the preregistered scale-48 matched replay in
`QWEN_CAUSAL_LEDGER_SCALE_RESCUE_V3_PLAN.md`. No scale sweep or confirmation
inspection is allowed before that replay and its independent audit complete.

This experiment uses a controlled externally constructed residual. It does not
yet show that the baseline `ttt_proj`/`ttt_conv` writer can store and retrieve
the same causal records.
