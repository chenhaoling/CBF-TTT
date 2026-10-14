# V18 controlled memory-interference Qwen pilot

## Status

- Terminal status: `completed_controlled_interference_v18_qwen_pilot`
- Runner exit code: `0`
- Independent audit: passed, 12/12 checks
- Corpus admission: passed, 6/6 preregistered checks
- Formal corpus was not generated while this pilot was scored.

## Primary result

| Policy | Joint success | Free-generation joint | Update accuracy | Retained-key accuracy | Locality |
|---|---:|---:|---:|---:|---:|
| No memory | 0.0% | 0.0% | 11.25% | 18.75% | 100% |
| Retain all | 75.0% | 75.0% | 80.0% | 100% | 100% |
| Global reset | 0.0% | 0.0% | 100% | 0.0% | 100% |
| Last 1 | 0.0% | 0.0% | 100% | 0.0% | 100% |
| Last 2 | 0.0% | 0.0% | 98.75% | 12.5% | 100% |
| Exact selective oracle | **100%** | **100%** | **100%** | **100%** | **100%** |

The exact selective-deletion oracle improves joint update-and-retention success
over the strongest fixed arm by **25 percentage points**.  Global reset solves
the new update but removes all queried valid history, which is the failure mode
the joint endpoint was designed to expose.

## Mechanism and length checks

Oracle joint success is 100% for every mechanism and length.  Retain-all joint
success is 90/70/60/80% for clean update, irrelevant near duplicate, stale
repetition and useful repetition.  It is 70/70/70/90% at lengths 8/16/32/64.
The formal study must establish that the positive gap persists with unseen
subjects, values and surface templates.

## Resources and provenance

- 40 episodes
- 1,435 unique memory states
- 7,175 query measurements
- Slower shard wall time: 1,029.61 seconds
- Mean/median/p95 query time: 0.2694/0.2740/0.3287 seconds
- Peak allocated/reserved GPU memory: 8.30/8.75 GiB
- Model: final 1B-token Qwen3-4B slow checkpoint
- Native `ttt_proj`/`ttt_conv` adapters excluded during causal-headroom scoring
- Summary SHA-256: `3487fdf2fef70a9aa65276ffd81e9495992f8385bb0127ec955b99679f6fa0d2`
- Audit SHA-256: `397b20382e4328c6f4bab0061cd2e5c495e302e4f6981419e24436c7162e2bf6`

This pilot validates corpus difficulty and causal headroom.  It is not a formal
efficacy claim and does not authorize natural-benchmark testing by itself.
