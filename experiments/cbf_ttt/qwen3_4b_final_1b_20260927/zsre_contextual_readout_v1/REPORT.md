# ZsRE contextual memory readout v1

## Objective

Test whether the frozen `context_last` representation selected by the synthetic geometry study can distinguish natural factual updates, paraphrases, and same-subject questions about another relation. This is a candidate-memory readout study on a constructed subject-disjoint split of ZsRE; it is not official ZsRE editing accuracy or open-vocabulary generation.

## Protocol

- Source: `ZsRE-test-all.json`, SHA-256 `0e0214dda853a906ef02bef58c7ce2978af5a962e91fc4a10827e0125459905c`.
- Select one deterministic canonical row for each of 1,068 unique subjects.
- Subject-disjoint calibration/development/test sizes: 267/267/534 subjects.
- Session width: eight facts, yielding 34/34/67 sessions.
- Frozen final Qwen3-4B checkpoint after the 1B-token continual-pretraining run.
- Frozen final-layer hidden state at the last prompt or value token.
- Three independent 256-dimensional projections and calibration-only covariance whitening.
- Calibration alone chooses the regularizer and unknown threshold.
- The test split is loaded once only if every development gate passes.

Each session is evaluated before and after replacing every original object with its ZsRE update. Exact prompts and rephrases must retrieve the active object. The two same-subject, different-relation locality prompts attached to each fact must return `unknown`.

## Calibration selection

Calibration selected `context_last_r0.01` with whitening regularizer 0.01 and unknown score 0.35. Mean calibration accuracy was 98.14% across the three representation seeds.

## Development result

| Query type | Accuracy | NLL |
|---|---:|---:|
| Original exact | 100.00% | 0.1799 |
| Updated exact | 100.00% | 0.1802 |
| Original rephrase | 98.75% | 0.5101 |
| Updated rephrase | 98.63% | 0.5133 |
| Original locality | 88.58% | 1.4317 |
| Updated locality | 87.95% | 1.4302 |

Overall development accuracy was 93.80%, the minimum representation-seed accuracy was 93.49%, and mean NLL was 0.8884. Combined locality accuracy was **88.26%**, below the preregistered 90% threshold. Every other gate passed.

The development gate therefore failed and the 534-subject test split remained sealed. No test embedding, prediction, or aggregate was produced.

## Interpretation

The contextual representation transfers strongly from synthetic identifiers to natural exact prompts and paraphrases. The remaining error is concentrated in same-subject questions about another relation: subject identity can dominate relation identity and spuriously activate the subject's stored object.

This result motivates relation-aware key learning rather than a larger forgetting controller. The registered follow-up compares an identity map, prompt/rephrase contrastive alignment, and contrastive alignment with same-subject locality hard negatives. Values, memory construction, backbone, split, and release gate remain frozen.

## Runtime and audit

- Formal run: 2026-10-12 16:58:10 to 16:58:28 Asia/Shanghai.
- Task time: 15.70 seconds.
- Peak CUDA allocated/reserved: 8.027/8.811 GiB.
- Formal code commit: `5d3f83c12178575a3e4a142574c17828309faf63`.
- Independent construction, split, hash, aggregation, gate, and conditional-test audit: passed.

## Artifacts

Compact summaries, manifests, timestamps, and the execution audit are archived in this directory. Projection tensors and raw development JSONL files remain under `/home/ctj/cbf_ttt_zsre_contextual_readout_v1` on `hku-gpu2`.

## Claim boundary

- Supported: frozen Qwen contextual keys reliably retrieve exact and paraphrased natural facts in the constructed candidate-memory protocol.
- Supported: the preregistered failure is localized to same-subject, different-relation locality discrimination.
- Not supported: official ZsRE knowledge-editing performance, open-vocabulary generation, native dense `ttt_proj`/`ttt_conv` memory writing, or end-to-end forgetting benefit on ZsRE.
