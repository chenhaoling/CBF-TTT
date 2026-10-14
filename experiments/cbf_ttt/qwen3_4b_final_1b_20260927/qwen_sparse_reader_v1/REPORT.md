# Qwen sparse-reader ceiling diagnostic

## Question

The registered dense rank-one memory reached only 67.90% historical accuracy
even under a 27-grid oracle.  This development-only diagnostic asks whether
unrelated active keys accumulate enough cross-talk to limit retrieval.

The diagnostic freezes the Qwen3-4B checkpoint, contextual rank-256 transform,
unknown calibration, development sessions and forgetting actions.  It changes
only causal read routing.  `component_topk` ranks active fact keys by cosine to
the query key and aggregates the best `k`; `exact_key_upper` is an ineligible
string-equality upper bound.

## Audited result

Three projection seeds produce the following oracle results:

| Reader | Accuracy | NLL | Historical | Minimum family | Components |
|---|---:|---:|---:|---:|---:|
| Dense all | 89.06% | 0.4356 | 67.90% | 71.82% | 14.77 |
| Component top-1 | **99.33%** | **0.1610** | **98.00%** | **97.34%** | 1.00 |
| Component top-2 | 97.12% | 0.2431 | 91.45% | 91.38% | 2.00 |
| Component top-4 | 89.84% | 0.3952 | 69.70% | 72.63% | 3.70 |
| Component top-8 | 89.00% | 0.4378 | 67.77% | 71.82% | 6.63 |
| Component top-16 | 89.06% | 0.4356 | 67.90% | 71.82% | 9.09 |
| Exact-key upper | 100.00% | 0.1900 | 100.00% | 100.00% | 1.47 |

Top-1 accuracy is 99.436/99.436/99.132% across seeds.  It passes every
registered accuracy, role, family and every-seed check.  Top-2 also passes,
but the registered smallest-eligible-reader rule selects top-1.  Its 0.67-point
gap from the exact-key upper bound shows that frozen Qwen contextual keys are
already sufficient for causal routing; dense accumulation is the main
historical-readability bottleneck.

For top-1, retain-all remains at 66.67%, while the explicit physics rule and
exact referenced-batch deletion reach 92.90%.  The 27-grid oracle reaches
99.33%, leaving 6.44 accuracy points beyond the symbolic physics rule for the
fresh formal controller experiment.

## Integrity and excluded attempts

Two implementation attempts are excluded and retained on the server:

1. the first summed per-component matrix-vector products rather than summing
   matrices before one multiplication, changing 173 near-tied oracle actions;
2. `retry1` extracted development identifiers alone, so padding/batch-shape
   roundoff differed from the registered train+development extraction.

`retry2` fixes both.  Its dense oracle exactly matches the registered cohort
experiment on all 4,608 seed-2301 queries, all actions and every NLL value;
maximum NLL difference is zero.  The independent audit reconstructs all 84
raw files, source rows, aggregates, selection logic and sealing state.  All 11
checks pass.  Test identifiers were never loaded or scored.

## Provenance

- server root: `/home/ctj/cbf_ttt_qwen_sparse_reader_20261015_retry2`;
- code commit: `dd26d438a74acebc44056bdc67433774d3e844d8`;
- runtime: 998.36 seconds;
- extraction: 11.84 seconds;
- peak CUDA allocated/reserved: 7.867/8.115 GiB;
- summary SHA256: `c281b79253dd7ab13bf4a30467a751af3f3985e4bd2729a9176b5a0552153836`;
- training SHA256: `f106805b049db2237e4623209d051479d9e1abea8e601c8dd110702d781653c1`;
- manifest SHA256: `ace92f1ef2293ec39ca38b9f17fb705eebed1fe91c00c143bd75b8289a8545db`;
- audit SHA256: `684f4e2b9e99074f3daf08b08b605a05cbe6d78818eca09585f7bb6f7066256e`.

The selected reader is released only into the separately frozen fresh-corpus
confirmation in `QWEN_SPARSE_COHORT_FORMAL_PLAN.md`.
