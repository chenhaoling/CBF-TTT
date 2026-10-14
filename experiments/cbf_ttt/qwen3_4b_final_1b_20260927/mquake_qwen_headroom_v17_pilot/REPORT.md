# MQuAKE-T V17 Causal-Headroom Resource Pilot

## Status

- Protocol: `mquake_qwen_causal_headroom_v17_pilot`
- Remote root: `/home/ctj/cbf_ttt_mquake_qwen_headroom_v17_pilot_20261014`
- Terminal marker: `completed_mquake_qwen_headroom_v17_pilot`
- Runner exit code: `0`
- Independent audit: passed all 10 checks
- Formal and external sets scored by this run: no

## Frozen design

The final Qwen3-4B slow checkpoint was evaluated with native `ttt_proj` and
`ttt_conv` adapters disabled. For each of 40 frozen pilot episodes, the run
scored the five fixed states and every causal single deletion of a visible
historical session. The current session was never deleted. Identical fixed
states were scored once. Each unique state received both primary queries plus
the hard-retain and unrelated safety queries.

This pilot was preregistered as a resource and integrity gate. Its headroom
estimate is descriptive and was not allowed to decide whether formal scoring
would run.

## Resource result

| Metric | Result |
|---|---:|
| Rows | 40 |
| Unique memory states | 1,360 |
| Query measurements | 5,440 |
| Shard 0 wall time | 646.86 s |
| Shard 1 wall time | 631.41 s |
| Mean query latency | 0.2160 s |
| Median query latency | 0.2155 s |
| P95 query latency | 0.3401 s |
| Maximum query latency | 0.5000 s |
| Peak allocated memory | 8.91 GiB |
| Peak reserved memory | 9.46 GiB |
| Projected 272-row two-GPU time | 1.22 h |

All six registered resource checks passed: coverage, finite scores, context
length, per-query measurements, the 31.5 GiB memory cap, and the 48-hour
formal-time cap.

## Descriptive headroom

| Metric | Oracle | Outer-fold fixed |
|---|---:|---:|
| Primary candidate accuracy | 100.0% | 100.0% |
| Primary answer NLL | 0.1286 | 0.0720 |
| Hard-retain accuracy | 100.0% | not a pilot gate |
| Unrelated accuracy | 100.0% | not a pilot gate |

The subject-macro oracle-minus-fixed primary accuracy gain was **0.0 points**
with subject-cluster 95% interval `[0.0, 0.0]`. Every outer fold selected
`global_reset` as its fixed comparator. This is a ceiling-effect warning: on
the pilot, deleting all history except the current session already solved the
two primary candidate queries. The result does not support a controller and
does not refute causal headroom on the formal sample; by registration it only
unlocks formal scoring.

## Integrity notes

The summary SHA-256 is
`41479b8c43ab77d9396a80e96ce34967db2a95b0a0137290d6131c44c5a0889f`.
The independent audit SHA-256 is
`1fb8ccd60a544a3f6a309f83f81a00e943a3697115cd94419d62aba2d60e914c`.
Raw shard hashes are stored in `audit.json` and match the archived files.

Episode history lengths differ, so the auditor derives expected per-shard
state counts from the frozen state file rather than assuming equal work. A
regression test covers this invariant. Both realized shards happened to score
680 unique states and 2,720 queries.

## Decision

The resource pilot passed and released the preregistered 272-row formal
causal-headroom experiment. It did not release controller training or external
benchmark evaluation.
