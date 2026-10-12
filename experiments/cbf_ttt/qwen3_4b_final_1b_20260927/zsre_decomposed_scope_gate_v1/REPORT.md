# ZsRE decomposed scope gate v1

## Objective

Automatically decide whether a natural query should retrieve one session update or defer to the frozen language model, while preserving the strong frozen cosine candidate ranking. This removes the symbolic reference privilege used by the controlled forgetting experiment.

## Method

Three frozen contextual projections select the candidate key by mean cosine. The learned gate cannot alter this candidate. It predicts only `retrieve` or `defer` from:

- the frozen normalized Qwen joint hidden for the query and top key;
- top cosine;
- top-score variation across projections;
- top1–top2 margin;
- projection argmax agreement.

The four scalar features are standardized on 27 calibration-fit sessions. Seven disjoint calibration sessions select the checkpoint and probability threshold. Three linear-gate seeds use the same frozen Qwen3-4B checkpoint, data split, pair template, and candidate transforms. A cosine-threshold ensemble is the control.

## Development gate

| Resolver | Overall | Exact | Rephrase | Locality/defer | NLL |
|---|---:|---:|---:|---:|---:|
| Ensemble threshold | 91.20% | 100.00% | 99.63% | 82.58% | 0.2062 |
| **Decomposed gate** | **98.50%** | **100.00%** | **98.13%** | **97.94%** | **0.0890** |

All gate seeds selected epoch 200 and threshold 0.4. Their development accuracies were identical at 98.50%. The primary improved overall accuracy by 7.48 points, source-session bootstrap 95% CI `[5.51,9.53]`. Locality improved by 15.69 points, CI `[11.58,19.85]`; rephrase changed by -1.47 points, CI `[-3.31,0.37]`.

All six preregistered checks passed: exact ≥95%, rephrase ≥90%, locality ≥92%, minimum seed ≥93%, overall gain ≥1 point, and positive bootstrap lower bound. This released the 534-subject test once.

## One-time test result

| Resolver | Overall | Minimum seed | Exact | Rephrase | Locality/defer | NLL |
|---|---:|---:|---:|---:|---:|---:|
| Ensemble threshold | 88.58% | 88.58% | 100.00% | 98.88% | 77.72% | 0.2781 |
| **Decomposed gate** | **98.47%** | **98.46%** | **100.00%** | **97.75%** | **98.06%** | **0.0901** |

The learned gate improved test accuracy by **9.90 points**, with 67-session paired bootstrap 95% CI **[8.57,11.24]**. The gain came from a 20.37-point locality improvement, CI `[17.75,23.04]`; rephrase changed by -1.12 points, CI `[-2.61,0.19]`. Candidate-selection upper bound was 99.95%, confirming that residual errors are almost entirely gating rather than key ranking.

## Interpretation

The result validates the proposed decomposition on natural, subject-disjoint text: candidate identity and memory applicability are separate decisions. A single similarity threshold transfers poorly to held-out test locality, while a lightweight gate over frozen Qwen pair evidence rejects out-of-scope queries without a statistically detectable rephrase loss.

This supplies automatic reference evidence that the earlier forgetting controller lacked. It does not yet show that deleting session memory improves a natural sequential task; the next causal experiment must feed this evidence into retain/local/global forgetting decisions over natural update sequences.

## Runtime and audit

- Formal run: 2026-10-12 17:35:38 to 17:37:09 Asia/Shanghai.
- Task time: 86.79 seconds.
- Peak CUDA allocated/reserved: 8.504/11.004 GiB.
- Formal code commit: `043462cae8a978887d9d758450d1fa3c59bb7dba`.
- Server tests: 13/13 passed.
- Independent source, feature, checkpoint, calibration split, raw-record, aggregation, comparison, gate, conditional-test, and test reconstruction audit: passed.

## Artifacts and claim boundary

Compact artifacts are archived here. The gate checkpoint and raw dev/test JSONL files remain at `/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1` on `hku-gpu2`.

- Supported: automatic natural query-to-update scope gating in an explicit structured eight-key memory.
- Supported: a 9.90-point one-time test gain over a frozen cosine-threshold control, with a source-session confidence interval excluding zero.
- Not supported: correctness of the frozen model after `defer`, official ZsRE generation/editing accuracy, native dense `ttt_proj`/`ttt_conv` writing, or learned forgetting gains on natural sequential tasks.
