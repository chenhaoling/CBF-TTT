# Causal-ledger lifecycle OOD V1

## Outcome

The one-time reserved-seed lifecycle OOD evaluation passed all ten frozen
scientific gates and all seven independent audit checks. The scale-48 full
causal ledger reached 100% six-way candidate accuracy and 100% five-token exact
generation on all 5,760 queries. It was perfect for every lifecycle suite,
current/historical/anchor role, length 16/24/32 and absent/null query, and
agreed exactly with the independently supplied semantic oracle.

This is the first controlled result in the project that combines a fresh
matched confirmation with a disjoint lifecycle OOD test. OOD identifiers and
symbols have zero overlap with both development and confirmation.

## Main comparison

| Policy | Candidate accuracy | Five-token exact | Answer NLL |
|---|---:|---:|---:|
| No memory | 15.56% | 15.56% | 20.1376 |
| Retain latest | 83.77% | 77.78% | 5.1700 |
| Global boundary clear | 55.56% | 55.56% | 10.6050 |
| Last 1 | 55.56% | 55.56% | 10.6050 |
| Last 2 | 56.67% | 56.67% | 10.3459 |
| Dense full | 60.09% | 4.44% | 28.9766 |
| **Full causal ledger** | **100.00%** | **100.00%** | **0.000500** |
| Semantic oracle | 100.00% | 100.00% | 0.000500 |

`retain_latest` was the strongest non-oracle comparator. The source-group
macro exact gain was 22.22 percentage points, with bootstrap 95% interval
`[20.07, 24.41]` points. The answer-NLL gain was 5.1695 with interval
`[4.6647, 5.6838]`.

## OOD suites

| Suite | Candidate accuracy | Five-token exact | Answer NLL |
|---|---:|---:|---:|
| Length 24/32 | 100.00% | 100.00% | 0.000464 |
| Two distant revocations | 100.00% | 100.00% | 0.000510 |
| Restore older value | 100.00% | 100.00% | 0.000569 |
| Reset then duplicate/no-op | 100.00% | 100.00% | 0.000481 |
| Unseen mixed lifecycle | 100.00% | 100.00% | 0.000476 |

## Integrity and resources

- Reserved seed `202610162`, 320 groups, 640 episodes and 5,760 rows.
- Fresh OOD surface template, identifiers and fourth-quarter value namespace.
- Earlier/OOD episode-ID overlap: zero; earlier/OOD symbol overlap: zero.
- Model loading had no missing, unexpected or mismatched parameters.
- Runner and formal-chain exit codes: zero.
- All row prompts, state transitions, policy returns, generations, hashes,
  bootstrap intervals and gates were independently reconstructed.
- Two RTX 5090 critical path: 1,019.32 seconds (16.99 minutes).
- Peak allocated/reserved GPU memory: 7.669/7.695 GiB.

## Supported claim and next boundary

Together with the matched confirmation, this supports a strong controlled
claim: addressable record state plus causal reset/revoke/shadow/restoration
semantics generalizes to longer sessions and unseen event compositions, avoids
dense cross-talk and outperforms every fixed global/window/single-rule policy.

The experiment uses externally constructed exact keys, value residuals and
visible lifecycle operations. It does not establish automatic lifecycle
extraction from natural text or factual binding by the checkpoint's original
`ttt_proj`/`ttt_conv` writer. The registered next stage is natural-text scope
and lifecycle transfer on the structured memory substrate; the failed native
writer branch remains stopped.
