# MQuAKE-T V17 Formal Exact Causal Headroom

## Status

- Protocol: `mquake_qwen_causal_headroom_v17_formal`
- Remote root: `/home/ctj/cbf_ttt_mquake_qwen_headroom_v17_formal_20261014`
- Code commit: `fd36c3f`
- Rows: 272
- Runner exit code: `0`
- Terminal marker: `completed_mquake_qwen_headroom_v17_formal`
- Independent audit: passed all 11 checks
- External sets accessed: no
- Registered headroom gate: failed

## Registered primary result

| Metric | Exact oracle | Outer-fold fixed |
|---|---:|---:|
| Primary candidate accuracy | 100.0% | 100.0% |
| Primary pair accuracy | 100.0% | 100.0% |
| Primary old-phrase NLL | 0.1633 | 0.0908 |
| Primary current-phrase NLL | 0.1680 | 0.0914 |

The subject-macro oracle-minus-fixed primary-accuracy gain was **0.0 points**
with subject-cluster 95% interval `[0.0, 0.0]` over 68 subjects and 20,000
bootstrap draws. Every outer fold selected `global_reset`; every fold's
accuracy gain was exactly zero. The fixed-minus-oracle primary NLL difference
was -0.0746 with 95% interval `[-0.1005,-0.0539]`, so the exact single-deletion
oracle also had worse primary NLL than reset-to-current memory.

Both preregistered checks failed:

- subject-macro primary accuracy gain >=5 points: false;
- subject-cluster 95% interval lower bound >0: false.

Therefore `causal_headroom_supported=false` and the registered next stage is
`stop_controller_training_no_causal_headroom`. No readout-utility controller
may be trained on this MQuAKE formulation and external confirmation remains
locked.

## Safety trade-off discovered by the complete matrix

| Query kind | Exact oracle | Selected fixed `global_reset` |
|---|---:|---:|
| Hard-retain accuracy | 100.0% | 70.59% |
| Unrelated accuracy | 98.53% | 55.15% |
| Hard-retain NLL | 0.1821 | 1.0563 |
| Unrelated NLL | 0.2731 | 1.5713 |

This does not overturn the registered primary failure. It shows why the
primary-only endpoint is unsuitable for the next experiment: it rewards
discarding session history whenever the current update alone answers the two
primary queries. The selective oracle preserved useful history, but
`retain_all` already solved the primary candidate task, leaving no accuracy
headroom for selective deletion.

The oracle retained all memory in 45 episodes and selected a single deletion
in 227. Only 50 deletions targeted the registered primary-old session; 177
targeted another history component through the accuracy/NLL tie-break. This is
additional evidence that symbolic stale-deletion accuracy and downstream
candidate accuracy are not interchangeable targets.

## Resources

| Metric | Result |
|---|---:|
| Unique memory states | 9,248 |
| Query measurements | 36,992 |
| Shard 0 states / wall time | 3,808 / 3,121.15 s |
| Shard 1 states / wall time | 5,440 / 5,722.22 s |
| Mean query latency | 0.2194 s |
| Median query latency | 0.2292 s |
| P95 query latency | 0.3359 s |
| Maximum query latency | 0.5486 s |
| Peak allocated memory | 8.96 GiB |
| Peak reserved memory | 9.94 GiB |

The unequal wall times follow the frozen round-robin row assignment combined
with unequal history lengths. The auditor independently derives exact state
counts for each shard.

## Integrity

- Summary SHA-256:
  `e96ab43c5e8e4ab4bd5bab5959784bd8bb1ab7d769747ffc0881ccadab346a20`
- Audit SHA-256:
  `80cb31a1dde5aa19fb62a048fe3e8149401a41106b03f60dc74c04f76abee3ce`
- Shard 0 SHA-256:
  `457dfbf6b0e620b6b7517e4aac372098489bd1494d2370b5e752fb7ce535502e`
- Shard 1 SHA-256:
  `79d20964cc789e9ebcd8766a94793ede28b1752aae62f38d9d611b97cd130703`

## Next experiment

The follow-up was preregistered before this formal result was read in
`POST_V17_INTERFERENCE_CORPUS_PLAN.md`. It constructs opaque unseen facts and
uses joint update-plus-retention episode success, free generation and a
source/template-disjoint split. This removes both the candidate ceiling and
the primary-only incentive for global reset while preserving V17 as a complete
negative result.
