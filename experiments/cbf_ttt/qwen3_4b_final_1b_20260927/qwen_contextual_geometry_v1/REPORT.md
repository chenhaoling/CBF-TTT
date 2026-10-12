# Qwen contextual key/value geometry v1

## Objective

Localize the failure of mean-token Qwen memory and test whether frozen final-layer contextual representations provide a readable multi-component substrate. All representation choices were frozen before this run, and the test split remained sealed.

## C0: key/value factorial

The best prior mean-token representation was combined with deterministic random key or value codes at the same 256-dimensional width.

| Key | Value | Dev accuracy | Minimum family | Mean NLL |
|---|---|---:|---:|---:|
| mean-Qwen | mean-Qwen | 32.20% | 28.39% | 2.5031 |
| mean-Qwen | random | 54.88% | 35.59% | 1.2881 |
| random | mean-Qwen | **98.18%** | **97.57%** | 0.5954 |
| random | random | 100.00% | 100.00% | 0.3743 |

Replacing only the key geometry recovers almost all readability, while replacing only values leaves a large failure. This identifies entity-key collision as the dominant problem in the mean-token construction; value geometry contributes a much smaller residual error.

## C1: contextual representations

Both variants use final-layer hidden states from the frozen final Qwen3-4B checkpoint, 256-dimensional Gaussian projection, covariance whitening, and train-only selection of whitening regularization and unknown score.

| Representation | Selected regularizer | Unknown score | Train accuracy | Dev accuracy | Minimum seed | Minimum family | 16 components |
|---|---:|---:|---:|---:|---:|---:|---:|
| `context_last` | 0.01 | 0.75 | 99.89% | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| `context_mean` | 0.001 | 0.50 | 78.07% | 76.51% | 76.13% | 66.67% | 99.01% |

`context_last` also reached 100% at every registered load from 1 through 32 components. `context_mean` reached 97.06% at 32 generic components but failed the event-world gate, especially on reset episodes. The final identifier token carries the within-world discriminative information that is diluted by averaging the target span.

## Gate and release

- Random-control gate: passed.
- `context_last`: passed all five readability checks.
- `context_mean`: failed mean, minimum-seed and minimum-family checks.
- Frozen selected representation: `context_last_r0.01`, `unknown_raw=0.75`.
- Test loaded/scored: no/no.
- Forgetting-controller v2: released under the unchanged policy gate.
- Independent audit: passed every source, hash, selection, aggregate, gate and test-lock check.

## Runtime

- Formal run: 2026-10-12 16:27:07 to 16:29:37 Asia/Shanghai.
- Total task time: 145.99 seconds.
- Peak CUDA allocated/reserved: 7.867 / 8.277 GiB.
- Hardware: one RTX 5090 used.

## Artifacts

- Compact summary, training metadata, manifest and audit are archived here.
- Raw JSONL and the 34.7 MB contextual transform checkpoint remain at `/home/ctj/cbf_ttt_qwen_contextual_geometry_v1` on `hku-gpu2`.
- The formal result was generated at commit `d445a8e`.

## Claim boundary

This experiment establishes a readable frozen-Qwen contextual low-rank memory and causally localizes the prior failure to key geometry. It does not yet show that the learned forgetting controller beats fixed policies, and it does not evaluate native dense TTT writing or natural-language benchmarks.
