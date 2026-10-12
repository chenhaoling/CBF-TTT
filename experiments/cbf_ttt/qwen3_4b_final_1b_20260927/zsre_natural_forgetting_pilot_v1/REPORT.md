# Natural ZsRE sequential forgetting pilot v1

## Objective

Test, on calibration only, whether the frozen natural scope gate can drive local deletion in deterministic multi-subject update streams before any development or test release.

## Construction and protocol

- 26 calibration subject/fact chains from duplicate ZsRE rows.
- 56 distinct-target versions: 26 initial retain events and 30 local supersessions.
- Four sessions with up to eight independent chains.
- After every update, query current exact/rephrase, all historical exact/rephrases for the newest value, and current locality/defer.
- Six policies and three frozen gate seeds replay identical features.
- Development and test files were never loaded by the evaluator.

## Calibration result

| Policy | Accuracy | Action accuracy | Current exact | Current rephrase | Historical | Locality | Mean memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed retain | 79.18% | 46.43% | 97.00% | 75.68% | 14.54% | 99.40% | 8.46 |
| Fixed global | 41.74% | 7.14% | 16.82% | 16.82% | 24.11% | 74.12% | 1.00 |
| Fixed top delete | 41.74% | 10.71% | 16.82% | 16.82% | 24.11% | 74.12% | 1.00 |
| **Cosine local** | **97.21%** | **98.21%** | **100.00%** | 94.29% | 92.20% | 99.40% | 6.11 |
| Learned scope local | 84.43% | 53.57% | 72.67% | 67.27% | 95.39% | 94.24% | 4.32 |
| Oracle local | 97.77% | 100.00% | 100.00% | 94.29% | 95.39% | 99.40% | 5.95 |

The best fixed policy was cosine local. Learned local was 11.41 points worse, with 26-chain bootstrap 95% CI `[-16.04,-6.82]`. Cosine made only one action error in 56 events; the learned gate correctly handled only 13/26 retain and 17/30 local decisions.

## Interpretation

This pilot invalidates direct transfer from query-time scope gating to update-time deletion. When a new fact's own key is absent, the learned gate sees an out-of-distribution candidate set and often deletes an unrelated top key. The simple cosine threshold nearly saturates the task because every retain event introduces a different subject, while every local event paraphrases an existing fact.

The benchmark therefore lacks the hard retain boundary needed to test semantic scope. Stage B must not run on v1. The registered repair adds each main fact's ZsRE locality ground truth as an independently stored same-subject fact: adding it requires retain, whereas later main-fact versions require local deletion. This uses calibration diagnosis and keeps development/test unopened.

## Runtime and audit

- Run: 2026-10-12 17:49:58 to 17:50:17 Asia/Shanghai.
- Task time: 16.16 seconds.
- Unique texts/pairs: 209/3,180.
- Peak CUDA allocated/reserved: 8.231/9.576 GiB.
- Formal commit: `ce1e9f8d435f1fee6fd8e852edaac29a200c2592`.
- Server tests: 10/10 passed.
- Independent data, source, scope-checkpoint, raw-record, aggregation, comparison and held-out-lock audit: passed.

## Claim boundary

Supported: physical cosine local deletion is sufficient for the easy cross-subject v1 streams; direct learned-gate transfer fails. Not supported: development/test performance, a learned forgetting advantage, native dense TTT memory, or official ZsRE performance.
