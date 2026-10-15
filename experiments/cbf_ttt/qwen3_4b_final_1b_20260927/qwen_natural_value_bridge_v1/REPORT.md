# Qwen natural-value residual bridge V1

## Question and protocol

This preregistered experiment asks whether the structured Qwen fast-memory
substrate that passed controlled lifecycle OOD can bind real, variable-length
counterfactual answers. It uses only the previously observed, subject-disjoint
ZsRE contextual-readout calibration and development splits. ZsRE test,
LongMemEval-S, LoCoMo, MMLU, RULER and RULER/QA remained unopened.

Each fact was scored under its exact and registered rephrased query with six
model-free candidates. Correct injected the updated answer's position-specific
normalized LM-head directions; Wrong injected the original answer; Zero added
no residual; Random applied deterministic norm-preserving sign randomization.
Calibration evaluated scales 4, 8, 16, 32 and 48 and selected the smallest
scale satisfying every frozen gate. Development used that scale once.

## Calibration and frozen selection

| Scale | Correct candidate | Correct exact generation | Wrong selects new | Wrong selects old | Zero selects new | Gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 77.33% | 30.81% | 3.10% | 94.38% | 30.43% | fail |
| **8** | **97.09%** | **78.10%** | **0.00%** | **100.00%** | **30.43%** | **pass** |
| 16 | 99.61% | 98.64% | 0.00% | 100.00% | 30.43% | pass |
| 32 | 99.61% | 100.00% | 0.00% | 100.00% | 30.43% | pass |
| 48 | 99.61% | 100.00% | 0.00% | 100.00% | 30.43% | pass |

The registered smallest-passing rule selected scale 8. Calibration exact and
rephrase candidate accuracy was 96.90% and 97.29%; length-conditioned token
exact generation was 79.46% and 76.74%.

## Subject-disjoint development result

| Arm | Updated-value candidate accuracy | Updated-value mean NLL | Injected-old accuracy |
|---|---:|---:|---:|
| **Correct** | **97.66%** | **0.2991** | — |
| Wrong | 0.00% | 4.4463 | 100.00% |
| Zero | 27.15% | 3.4113 | — |
| Random | 28.32% | 3.4265 | — |

Correct candidate accuracy was 97.66% for both exact and rephrased queries.
Fixed-correct-length autoregressive token exact match was 83.59% for both query
types. Correct improved candidate accuracy over the strongest control by 70.31
points on exact queries and 68.36 points on rephrases.

The updated-answer NLL gains were 4.1472 over Wrong, 3.1122 over Zero and
3.1274 over Random. Their source-fact bootstrap 95% intervals were respectively
`[3.8639, 4.4380]`, `[2.9044, 3.3276]` and `[2.9187, 3.3446]`. Every frozen
calibration and development check passed.

Length-stratified development candidate accuracy remained 97.22--100% across
the observed 1, 2, 3--4, 5--8 and 9--18 token buckets. Generation exact was
82.14%, 86.36%, 87.10%, 77.08% and 50.00%; the final bucket contains only four
queries and is descriptive rather than a reliable length estimate.

## Execution and audit

The calibration two-GPU critical path was 117.28 seconds and development was
49.64 seconds on two RTX 5090 cards. Peak allocated/reserved memory was
7.665/7.705 GiB. The independent audit passed all eight checks, including exact
corpus reconstruction, source isolation, prompt/token reconstruction, row and
resource hashes, independent bootstrap and gate recomputation, conditional
development release, and sealed test/external flags.

The first launch is retained under `failed_zero_row_launch/`. It stopped before
writing any score row because a trailing-space prompt boundary did not preserve
the Qwen tokenizer prefix. Recovery changed only that boundary to the existing
newline form, added an all-row tokenizer preflight, and used a new output root.

## Interpretation and claim boundary

The result establishes that a frozen Qwen LM-head residual can bind real
multi-token counterfactual values and causally control fresh-KV candidate
selection and fixed-length generation when the relevant record key is already
selected. The Wrong arm's 100% old-value selection demonstrates content
specificity; Zero and norm-matched Random controls rule out base-model recall
and residual magnitude alone.

This does not establish unrestricted generation, automatic fact extraction,
automatic key selection, native `ttt_conv`/`ttt_proj` writing, or official ZsRE
editing performance. Per the preregistered decision rule, it releases a natural
answer causal-ledger sequence experiment that composes the validated value
writer with explicit lifecycle state and the independently validated natural
scope resolver.
