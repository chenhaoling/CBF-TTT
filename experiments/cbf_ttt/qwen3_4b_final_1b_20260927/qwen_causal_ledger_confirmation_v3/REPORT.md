# Causal-ledger confirmation V3

## Outcome

The one-time confirmation passed all eleven frozen scientific gates and all
eight independent audit checks. It used the reserved seed `202610161`, unseen
surface templates, a disjoint value/record namespace and no confirmation-set
calibration.

The full causal ledger reached 100% candidate accuracy and 100% exact
five-token generation on all 8,064 confirmation queries. Current, historical
and anchor roles, all seven lifecycle families, sequence lengths 8/12/16 and
absent/null queries were each 100%. It exactly agreed with the independently
supplied semantic oracle.

## Main comparison

| Policy | Candidate accuracy | Five-token exact | Answer NLL |
|---|---:|---:|---:|
| No memory | 8.01% | 8.01% | 22.3120 |
| Retain latest | 92.45% | 89.91% | 2.4252 |
| Global boundary clear | 47.69% | 47.69% | 12.7278 |
| Last 1 | 47.69% | 47.69% | 12.7278 |
| Last 2 | 50.38% | 49.78% | 12.2194 |
| Dense full | 61.93% | 6.40% | 22.0813 |
| **Full causal ledger** | **100.00%** | **100.00%** | **0.000807** |
| Semantic oracle | 100.00% | 100.00% | 0.000807 |

`retain_latest` was again the strongest non-oracle policy. The source-group
macro exact gain was 10.0942 percentage points, with bootstrap 95% interval
`[8.8542, 11.3839]` points. Answer-NLL improved by 2.4244 with interval
`[2.1269, 2.7389]`. Every family and length-bucket gain was non-negative.

## Integrity and resources

- 448 source groups, 896 episodes, 8,064 query-policy rows.
- Development/confirmation episode-ID overlap: zero.
- Two RTX 5090 critical path: 1,411.38 seconds (23.52 minutes).
- Peak allocated/reserved memory: 7.666/7.693 GiB.
- Model loading had no missing, unexpected or mismatched parameters.
- Runner and formal-chain exit codes: zero.
- Confirmation data was unavailable until the passing scale-rescue summary and
  audit hashes were verified. OOD remained ungenerated.

The independent audit rebuilt the confirmation state trace, prompts, policy
returns, answer IDs, aggregates, comparator selection, bootstrap intervals and
gate. It also verified the passing V3 dependency and disjoint episode IDs.

## Supported claim and remaining boundary

Together with development V3, this supports a replicated controlled claim:
dynamic record-level reset/revoke/shadow/restoration semantics plus sparse
addressable reading remove multi-record interference and outperform the
strongest fixed policy by about ten exact-match points. Dense aggregation
failed again, at 6.40% exact.

The memory value residual and exact structured key are still externally
constructed. This result does not establish native `ttt_proj`/`ttt_conv`
content writing, automatic lifecycle extraction or natural benchmark gains.
