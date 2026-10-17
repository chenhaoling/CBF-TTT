# In-Place TTT Level-B formal RULER execution addendum

Status: **frozen before formal Level-B training or RULER evaluation**

Freeze date: 2026-10-18 (Asia/Shanghai)

Parent public-corpus plan SHA256: `c5019a28b9fa7fa27052ddf1d69eb029a2ec5b3b6446f1527e9ceb7db1a12bab`

Training addendum SHA256: `c33310cae2c196b06a15eca2462d009d9ca34674adc11f55a167048dd6d8d03b`

## 1. Scope

This addendum fixes the executable evaluation protocol for the public-data
matched replication. It does not change the model, writer, training corpus,
optimizer, training schedule, task set, sample count, context lengths or claim
boundary in the parent plans. It does not authorize update-admission or
forgetting experiments.

Formal evaluation remains sealed until the no-TTT and native-TTT Stage-2
training runs both pass their independent audits and publish their final HF
checkpoints. Each evaluation arm must resolve to the exact `final_hf_path` in
its Stage-2 audit.

## 2. Registered matrix

Both arms run the same 13 official OpenCompass RULER tasks with 100 examples
per task at 4K, 8K, 16K, 32K, 64K, 128K and 256K. The dataset generator seed
is explicitly `42`. The original Qwen3-4B tokenizer is used with
`fix_mistral_regex=false`; the converted model tokenizer must be byte-identical
to it. Generation is greedy, batch size is one and `max_new_tokens=64`, matching
the released repository model settings.

The 4K--128K arithmetic mean is the registered summary. The 256K result is
reported separately as extrapolation because Stage 2 trains with a 131,072
position configuration. Every task's ordered `origin_prompt` and `references`
are hashed. The no-TTT and TTT hashes must be identical at every context before
a paired delta may be reported.

## 3. Frozen evaluation environment

| Package | Version |
|---|---|
| OpenCompass | `0.5.4` |
| Transformers | `4.57.3` |
| Datasets | `2.21.0` |
| Torch | `2.8.0+cu128` |

The five generator configs and five dataset implementations are bound to:

| Module | SHA256 |
|---|---|
| `opencompass.configs.datasets.ruler.ruler_qa_gen` | `2c48ffb84f1ef858bd02ba333254defb67c528d3bd5142fe720c84591d4e1f06` |
| `opencompass.configs.datasets.ruler.ruler_niah_gen` | `e9c51a74a75428af7d1b6c2aafabbf1c93f817a1197e4b03301d66169837a77d` |
| `opencompass.configs.datasets.ruler.ruler_vt_gen` | `ce54adedba8ab4548809b4e3f0bce848f30acac87e6a1c2bdeffdb1de343aaa6` |
| `opencompass.configs.datasets.ruler.ruler_fwe_gen` | `9c54073a4c0fe687f0ed51e559ceb79041d28ab04258111017341111d8851c8d` |
| `opencompass.configs.datasets.ruler.ruler_cwe_gen` | `c8c0f87acc8f28b4ca6b3580223180ec06a30fdc140809d81fb66a1074efd24f` |
| `opencompass.datasets.ruler.ruler_qa` | `a753601befe8e6f4341b1c40ff4ebc710429e8f2098ead00e79aab8929a63da7` |
| `opencompass.datasets.ruler.ruler_niah` | `397b2b023c9c88fdc36041d82a71de21e7b71979893862173e628af832bcef7b` |
| `opencompass.datasets.ruler.ruler_vt` | `0501a09f88ec6bd919019a5be8516febee1bffe5229b7b174a1d35df2607c73a` |
| `opencompass.datasets.ruler.ruler_fwe` | `c900418620df5fb25363d9a0eda4fa5cf42ca4fb272772ee03ef006880ace194` |
| `opencompass.datasets.ruler.ruler_cwe` | `bf4a5b2e9168dc90030f420f9afda2a7f8c59cfe31306c46f55ee2dc4269edc3` |

OpenCompass resolves the three external source files through its own
`get_data_path(..., local_mode=True)` function. Their frozen hashes are:

| Logical path | Bytes | SHA256 |
|---|---:|---|
| `data/ruler/dev-v2.0.json` | 4,370,528 | `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8` |
| `data/ruler/hotpotqa.json` | 46,320,117 | `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea` |
| `data/ruler/PaulGrahamEssays.jsonl` | 1,968,927 | `78ad877447201471578c69dd4fc11e6f489970966329468c5d0ac32292d53056` |

Any package, generator, source-data or tokenizer drift stops the matrix.

## 4. Execution and resources

The registered matrix runner receives seven distinct GPUs with at least
80,000 MiB each and no existing compute process. It starts all seven context
lengths in parallel for the no-TTT arm, waits for all seven to pass, then does
the same for native TTT. Each context uses exactly one GPU. A failed context,
OOM, nonzero exit, missing task, missing example, non-finite score, tokenizer
warning or source/hash mismatch stops publication.

Single-context and matrix roots are built under new `.partial` directories and
renamed only after their independent audits pass. Raw predictions, raw task
results, official OpenCompass summaries, logs, model manifests, GPU traces,
per-task input hashes and all audit files are retained.

The training auditor now records both `final_hf_relative_path` and the absolute
path under the final published root. This corrects the otherwise stale path
that would result when a successful training directory is renamed from
`.partial`; it does not alter training behavior or checkpoint contents.

## 5. Frozen implementation hashes

| File | SHA256 |
|---|---|
| `eval_config/baseline_reproduction/qwen3_4b_formal_ruler.py` | `c00be4f034f83722bc8bd49329f1839e839a4e9d4ba2711938e6b6036532ae75` |
| `scripts/run_in_place_ttt_level_b_ruler_single.sh` | `962003b52d8a24f2da0d529ae35e8b857e8af3166ab29b99796e158652c13801` |
| `scripts/summarize_in_place_ttt_level_b_ruler.py` | `d82ad17beb3c82788a2b4f91e3f698353bcb4289999194f9545511db15079de7` |
| `scripts/audit_in_place_ttt_level_b_ruler.py` | `d278d3204c39c3f601b098c5584768511712aeb87df29a19e1c3ab5f5409fba0` |
| `scripts/run_in_place_ttt_level_b_ruler_matrix.sh` | `227a8efe1dffe5b02fef4183dd1045524341cc8dd212e4337fc91095ace35f30` |
| `scripts/summarize_in_place_ttt_level_b_ruler_matrix.py` | `2a74800f2f40009f11bc617b75db80da56476a82fc7ebabeb0e7c2d1b987b1ee` |
| `scripts/audit_in_place_ttt_level_b_ruler_matrix.py` | `cc630878c336f1857458b83ee33fb91a6c33b3ad1efdd5403383e4baea091925` |
| `scripts/run_in_place_ttt_level_b_training.sh` | `b9576a40d1ed8644a333be127b072be2d8c47d6bc11ed5d22421b088844d34d9` |
| `scripts/audit_in_place_ttt_level_b_training.py` | `51cb52a2cc704b591cc4688e2f134bd0556df62c0ef037fa893bb61c3fa9370b` |

## 6. Independent audit and claim boundary

The single-run audit independently recomputes the exact 13-task set, ordered
100-example set, raw scores, macro, official aggregate, input hashes, result
hashes, prediction hashes, model files, tokenizer, training-audit binding,
dependencies, source files and GPU trace. The matrix audit independently
recomputes all 14 runs, verifies that prior single-run audits have not become
stale, checks paired input identity and recomputes every context delta and the
registered 4K--128K mean.

Only a fully passed matrix may support a baseline comparison. It remains a
**public-data matched replication**, not an exact reproduction of the private
paper corpus or unreleased internal training recipe. Update-admission and
forgetting experiments remain sealed until this matrix passes.
