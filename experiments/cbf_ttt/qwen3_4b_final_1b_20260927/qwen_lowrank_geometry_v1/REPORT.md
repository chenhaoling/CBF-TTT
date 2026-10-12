# Qwen low-rank memory geometry v1

## Objective

Test whether covariance whitening makes the final Qwen3-4B checkpoint's mean-token key/value geometry readable under multi-component interference before retraining a forgetting controller.

The diagnostic constructs the exact active semantic state at every development decision boundary. It therefore contains no forgetting-policy error. The existing test split was not loaded or scored.

## Registered variants

- `rp128`: exact bitwise reproduction of the failed multi-component experiment.
- `rp128_whiten`: 128-dimensional projected covariance whitening.
- `rp256_whiten`: 256-dimensional projected covariance whitening.
- `random128`: deterministic random-code upper-bound control, ineligible for selection.

Whitening regularization was selected from `{1e-4, 1e-3, 1e-2}` using train readability only. Both dimensions selected `1e-2` under the frozen largest-within-one-point rule.

## Results

| Representation | Dev accuracy | Minimum seed | Minimum family | NLL | 16-component sweep |
|---|---:|---:|---:|---:|---:|
| rp128 | 32.65% | 31.86% | 29.34% | 2.4827 | 69.16% |
| rp128 + whitening | 31.62% | 31.55% | 27.60% | 2.5134 | **87.03%** |
| rp256 + whitening | 32.20% | 32.12% | 28.39% | 2.5031 | 87.01% |
| random128 | **100.00%** | **100.00%** | **100.00%** | 0.3822 | 100.00% |

The load curves were:

| Representation | 1 | 2 | 4 | 8 | 16 | 32 components |
|---|---:|---:|---:|---:|---:|---:|
| rp128 | 100.00% | 100.00% | 97.92% | 92.32% | 69.16% | 36.36% |
| rp128 + whitening | 100.00% | 100.00% | 97.72% | 95.31% | 87.03% | 66.85% |
| rp256 + whitening | 100.00% | 100.00% | 97.72% | 95.28% | 87.01% | 67.29% |

Whitening substantially improved generic cross-group capacity at 16 and 32 components, but did not improve the event-world diagnostic. None of the Qwen variants met the fixed 90% mean, 85% minimum-seed, or 85% minimum-family requirements. The random control passed all checks, showing that the evaluator and memory construction can reach the intended ceiling.

## Post-result localization

The following decomposition uses only the already saved development rows and is marked exploratory:

- rp256-whiten known-value accuracy: 37.93%; unknown accuracy: 3.56%.
- current/historical/anchor accuracy: 41.67% / 17.38% / 37.57%.
- random128 is 100% in every slice.

Cross-group load-sweep performance is much higher than within-world event performance at similar component counts. The event worlds deliberately share long key prefixes within a group. The evidence therefore points to local key collisions, plus poor unknown calibration, rather than dimension alone. A registered key/value factorial and contextual-hidden-state study is the next step.

## Gate and audit

- Readability gate: failed.
- Selected representation: none.
- Controller training: not released.
- Test loaded/scored: no/no.
- Independent audit: passed every protocol, hash, source bridge, aggregate, selection, gate and test-lock check.
- Exact bridge: the `rp128` embeddings reproduce the preceding experiment bit for bit.

## Runtime

- Formal run: 2026-10-12 16:12:33 to 16:13:27 Asia/Shanghai.
- Diagnostic time: 49.27 seconds.
- Peak CUDA allocated/reserved: 7.633 / 7.643 GiB.
- Hardware: one RTX 5090 used; the second GPU remained free.

An earlier preflight at commit `cd127ee` stopped before evaluation because batched projection was mathematically equivalent but not bit-identical to the v1 scalar projection. Commit `34a2e70` restored the exact old projection path for `rp128`; no data, gate or method setting changed.

## Artifacts

- Compact summary, training metadata, manifest and independent audit are archived here.
- Raw development JSONL and the 25.6 MB transform checkpoint remain at `/home/ctj/cbf_ttt_qwen_lowrank_geometry_v1_34a2e70` on `hku-gpu2`.
- Next frozen protocol: `QWEN_CONTEXTUAL_GEOMETRY_PLAN.md`.

## Claim boundary

This result shows that global covariance whitening increases generic associative capacity but does not solve same-world identifier interference. It does not test a forgetting controller, native TTT writer, autoregressive answer generation or a natural benchmark.
