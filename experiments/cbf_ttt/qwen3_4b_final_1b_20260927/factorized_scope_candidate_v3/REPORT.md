# Factorized scope-key candidate V3

## Status

Completed and independently audited. All six preregistered development checks
passed. This releases a separately preregistered causal deletion-gate
experiment; it does not itself establish forgetting efficacy. LongMemEval-S
remained locked and LoCoMo was not loaded.

## Main result

| Resolver | Top-1 | MRR | Mean rank |
|---|---:|---:|---:|
| Full-text maximum cosine (V1) | 33.99% | 47.18% | 7.07 |
| Full-text learned V2 | 67.38% | 73.16% | 4.86 |
| Scope-key maximum cosine | 93.63% | 96.21% | 1.11 |
| Scope-key learned V3 | **97.07%** | **98.38%** | **1.04** |
| Scope anchor-only diagnostic | 99.81% | 99.91% | 1.00 |

The primary seed improved top-1 over learned V2 by 29.40 points with
subject-cluster bootstrap 95% CI `[24.72,34.18]`. Three-seed top-1 was
96.72/97.85/96.63%. Mean top-1 by session length was 99.00%, 97.50%, 95.63%,
and 96.13% for 16/32/48/64; every length exceeded the fixed 93% floor.

The unlearned mechanism control is decisive: excluding value payload from the
key raises maximum-cosine top-1 by 59.64 points over the identical full-text
encoder. The learned listwise layer adds another 3.43 points. This supports a
factorized memory design in which prompt/relation scope locates an existing
memory and value payload supplies its content.

## Release gate

All checks passed: mean top-1, minimum seed, every length, MRR, paired gain over
V2, and fixed factorization gain. The next-stage marker is
`preregister_factorized_sparse_gate`.

## Reproducibility

- 2,136 mechanically parsed unique scope keys;
- selected epochs: 50/200/50;
- runtime: 32.22 seconds;
- peak GPU allocated/reserved: 8.026/8.988 GiB;
- independent audit: all nine checks passed;
- summary SHA-256: `998cf4def12c884caccc101cbd2574c7f67945f7b48ce716656b0112bd4e51c0`.

Hash-covered raw records remain at
`/home/ctj/cbf_ttt_factorized_scope_candidate_v3` on `hku-gpu2`.
