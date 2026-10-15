# Scale-48 causal-ledger development replay V3

## Outcome

The single preregistered scale-48 replay passed all eleven scientific gates and
all eight independent integrity checks. It releases exactly one matched
confirmation run with scale, policies, decoder and thresholds frozen.

The full causal ledger reached 100% six-candidate accuracy and 100% exact
five-token generation on all 8,064 queries. Current, historical and anchor
roles, every lifecycle family, all three sequence lengths, and absent/null
queries were each 100%. It exactly agreed with the semantic oracle.

## Comparators and effect

| Policy | Candidate accuracy | Five-token exact | Answer NLL |
|---|---:|---:|---:|
| No memory | 8.01% | 8.01% | 21.8642 |
| Retain latest | 92.52% | 89.91% | 2.3767 |
| Global boundary clear | 47.69% | 47.69% | 12.3445 |
| Last 1 | 47.69% | 47.69% | 12.3445 |
| Last 2 | 50.56% | 49.78% | 11.8189 |
| Dense full | 64.07% | 7.14% | 21.0200 |
| **Full causal ledger** | **100.00%** | **100.00%** | **0.000883** |
| Semantic oracle | 100.00% | 100.00% | 0.000883 |

`retain_latest` remained the strongest non-oracle comparator. The primary
exact gain was 10.0942 percentage points with source-group bootstrap 95%
interval `[8.8294, 11.3591]` points. Answer-NLL improved by 2.3758 with interval
`[2.0736, 2.6821]`. Every family and sequence-length gain was non-negative.

## Resources and audit

- Two RTX 5090 critical path: 1,388.89 seconds (23.15 minutes).
- Peak allocated/reserved memory: 7.666/7.693 GiB.
- 448 groups, 896 episodes and 8,064 query-policy rows.
- Final Qwen3-4B 1B-token checkpoint loaded without missing, unexpected or
  mismatched parameters.
- Newly rebuilt data was byte-identical to the audited development V2 corpus.
- Confirmation and OOD remained ungenerated during this replay.

The first audit invocation used `python scripts/...py`, so Python placed the
`scripts` directory rather than the repository root on `sys.path` and raised
`ModuleNotFoundError` before reading any result. No model scoring or summary was
rerun. The same independent audit was immediately invoked via `python -m`; it
passed all eight checks. Both logs and exit codes are preserved.

## Claim boundary

This development result shows that the scale-32 role-floor failure was a
contextual generation-margin problem. At the frozen scale 48, the dynamic
addressable ledger gives an audited 10.09-point advantage over the strongest
non-oracle policy and avoids the severe dense-read cross-talk. It remains a
controlled externally constructed residual result. Confirmation replication,
native `ttt_proj`/`ttt_conv` transfer and natural-data evaluation are separate
requirements.
