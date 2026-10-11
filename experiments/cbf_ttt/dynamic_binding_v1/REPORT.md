# Dynamic Random-Value Binding Experiment

## 1. Goal

The preceding dynamic-memory experiment showed that same-context joint training lowered answer NLL but produced zero correct random codes on held-out worlds. This experiment tested whether an explicit value-binding objective could make the writer preserve the session-specific value rather than merely learn the answer format.

The fixed comparison was:

- `joint_ce`: mean answer-plus-EOS cross entropy for four questions sharing one context write.
- `joint_binding`: digit-weighted positive loss plus an anchor contrast between correct and twin-session memories.

This is a writer prerequisite experiment. It does not train a forgetting policy, construct forgetting labels, or evaluate a controller.

## 2. Interpretation and assumptions

Each world contains two twin sessions. They share the same entities and questions but change the anchor value, so a model that only learns the output format or a global answer prior cannot pass the correct-versus-twin comparison. The train, development, and test worlds are disjoint. Natural 4,096-token backgrounds alternate equally between FineWeb-Edu and LongCrawl64.

The frozen Qwen3-4B backbone is used as the read/write substrate. Only the seven `ttt_conv`/`ttt_proj` writer pairs are optimized. Fresh-KV evaluation isolates the learned writer memory. Full-KV evaluation is a readability control and Empty/Wrong/twin memories detect format learning and content confusion.

The two arms match question exposures, order, initialization, optimizer, and evaluation. They are not compute matched: `joint_binding` creates both correct and twin memories on every update.

## 3. Fixed protocol

| Item | Value |
|---|---:|
| Model | Qwen3-4B |
| Semantic seed / writer seed / order seed | 20261013 / 301 / 607 |
| Train / dev / sealed test worlds | 128 / 32 / 32 |
| Questions per context | 4 |
| Training rounds | 4 |
| Exposures per arm | 4,096 |
| Optimizer updates per arm | 1,024 |
| Checkpoints | 0 / 2,048 / 4,096 exposures |
| Learning rate | `1e-7` |
| Optimizer | AdamW, FP32 state, no weight decay |
| Forward dtype | BF16 |
| Gradient clip | 1.0 |
| Safety limit | `max ||delta W|| / ||W|| <= 1` |
| Train probe / dev questions | 64 / 256 |
| Test scored | No |

`joint_binding` uses

```text
positive = mean(0.75 * digit_nll + 0.125 * prefix_nll + 0.125 * eos_nll)
contrast = softplus(0.2 + correct_anchor_digit_nll - twin_anchor_digit_nll)
loss = positive + 0.5 * contrast
```

Both correct and twin memory paths remain differentiable.

## 4. Execution result

The run completed on two RTX 5090 GPUs from 2026-10-11 11:07:50 to 11:59:29 +08:00. This 51 minute 39 second interval includes data construction, two-arm smoke, formal training, three evaluation checkpoints, summary, and independent audit.

- Focused launch tests: 7/7 passed on the server.
- Smoke: both arms completed; protocol audit passed.
- Formal: both arms completed all 1,024 updates and 4,096 exposures.
- Independent execution audit: passed.
- Frozen backbone hashes were identical and unchanged.
- AdamW state and saved writer parameters were FP32.
- No NaN, OOM, safety stop, or incomplete evaluation occurred.
- Test remained semantic text only and was not tokenized or scored.

## 5. Primary results

### 5.1 Final development metrics

Lower NLL is better. Code accuracy checks the complete random code and is the primary content metric.

| Arm | Memory | Code accuracy | Digit NLL | Answer NLL |
|---|---|---:|---:|---:|
| joint CE | Correct | 0.0000 | 2.6741 | 2.1421 |
| joint CE | Wrong | 0.0000 | 2.6726 | 2.1276 |
| joint CE | Empty | 0.0000 | 2.6962 | 4.2833 |
| joint CE | Full-KV | 0.9570 | 0.0377 | 0.0572 |
| joint binding | Correct | 0.0000 | 2.5776 | 2.0801 |
| joint binding | Wrong | 0.0000 | 2.6170 | 2.0972 |
| joint binding | Empty | 0.0000 | 2.6962 | 4.2833 |
| joint binding | Full-KV | 0.9570 | 0.0377 | 0.0572 |

At the fixed endpoint, binding minus CE on Correct memory was:

- dev digit NLL: **-0.0965**;
- dev answer NLL: **-0.0620**;
- dev code accuracy: **0.0000** difference;
- train-probe digit NLL: **-0.0362**;
- train-probe answer NLL: **-0.0307**.

The binding objective improved likelihood but did not recover a single complete random code on the 256 development questions.

### 5.2 Content discrimination

The fixed group-level gain is `control digit NLL - Correct digit NLL`.

| Arm | Control | Mean gain | Bootstrap 95% CI | Positive world fraction |
|---|---|---:|---:|---:|
| joint CE | Wrong | -0.0015 | [-0.1387, 0.1274] | 0.4375 |
| joint CE | Empty | 0.0221 | [-0.1076, 0.1152] | 0.7188 |
| joint binding | Wrong | 0.0394 | [-0.0917, 0.1707] | 0.5625 |
| joint binding | Empty | 0.1186 | [0.0224, 0.2010] | 0.8125 |

The binding arm reliably distinguishes a populated memory from Empty under this sample, but it does not reliably distinguish the correct memory from a wrong memory: the effect is small and its confidence interval crosses zero. The final dev anchor correct-versus-twin digit gain is only 0.0090.

Correct-memory first-token accuracy was 0.9141 for CE and 0.8828 for binding, while complete code accuracy was zero. The writer therefore learned to begin the answer in the expected `code_` format without preserving enough digits to identify the session value. Full-KV code accuracy of 0.9570 shows that the prompt and answer are readable when the original keys and values remain available.

### 5.3 Checkpoint trajectory

Development Correct-memory digit NLL:

| Exposure | joint CE | joint binding |
|---:|---:|---:|
| 0 | 2.6962 | 2.6962 |
| 2,048 | 2.6251 | 2.5290 |
| 4,096 | 2.6741 | 2.5776 |

Both arms were best at the midpoint and weakened slightly by the fixed endpoint. The midpoint was diagnostic only and was not used for checkpoint selection. Neither checkpoint produced nonzero code accuracy.

### 5.4 Training and resources

| Arm | First 50 loss | Last 50 loss | Mean update time | Peak allocated | Peak reserved | Max relative delta |
|---|---:|---:|---:|---:|---:|---:|
| joint CE | 3.9556 | 1.4433 | 0.4495 s | 12.0963 GiB | 12.3359 GiB | 0.7371 |
| joint binding | 4.7068 | 2.1729 | 0.7290 s | 14.1159 GiB | 14.8555 GiB | 0.7252 |

The higher binding loss contains the contrast term and is not directly comparable in scale to CE. Binding was about 62% slower per update because it performs two differentiable writes.

## 6. Gate decision and conclusion

The preregistered content-memory gate **failed for both arms**.

- Full-KV readability passed.
- Correct-memory code accuracy failed: 0% versus the required 60%.
- Correct did not exceed Wrong or Empty code accuracy by 20 percentage points.
- Binding passed the Empty digit-NLL gain criterion but failed the Wrong criterion.
- All numerical and execution safety checks passed.

The result supports a narrow conclusion: digit weighting plus one anchor twin contrast improves held-out likelihood and memory-versus-empty separation, but does not establish robust random-value binding. The current writer can recognize that a learned memory is present and can learn the answer format, yet it cannot reliably select the correct value from the memory content.

This experiment does not test whether a trained forgetting policy is effective. Starting controller training now would confound memory-content failure with forgetting decisions, so the preregistered protocol stops before that stage.

## 7. Files and implementation mapping

### Code added before execution

- `tasks/train_cbf_dynamic_binding.py`: four-question schedule, CE and binding objectives, two differentiable memory writes, streamed evaluation, summary, and fixed endpoint gate.
- `scripts/audit_cbf_dynamic_binding.py`: independently checks data hashes, schedules, loss accounting, FP32 optimizer/writer state, identical initialization, evaluation matrices, summary recomputation, and test sealing.
- `scripts/run_cbf_dynamic_binding.sh`: builds data, runs two-GPU smoke and formal stages, applies resource/safety checks, summarizes, and audits.
- `tests/test_cbf_dynamic_binding.py`: tests four-question scheduling, CE averaging, and the sign of the correct-versus-twin contrast gradient.
- `DYNAMIC_BINDING_PLAN.md`: preregistered question, arms, fixed data, gate, and command.

### Result files in this directory

- `formal/summary.json`: complete formal aggregate and final gate.
- `formal/execution_audit.json`: independent audit result.
- `formal/*/complete.json`: optimization, resource, dtype, and safety summaries.
- `formal/*/evaluation_*.summary.json`: per-checkpoint aggregates and hashes.
- `formal/*/training_manifest.json`: exact model, loss, data, seed, and source-file configuration.
- `smoke/summary.json` and `smoke/execution_audit.json`: smoke outcome and audit.
- `source/manifest.json` and `data/manifest.json`: semantic and packed-data provenance.
- `RUN_STATUS.json` and `RUN_STATUS.md`: terminal state.
- `ARTIFACTS_SHA256.txt`: hashes for the archived small artifacts.

Raw tokenized rows, per-update JSONL, per-question evaluation JSONL, and writer checkpoints remain on `hku-gpu2` under `/home/ctj/cbf_ttt_dynamic_binding_v1`. They are not committed because of size. The downloaded transport archive and combined run log are locally ignored after hash verification.

## 8. Reproduction

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927

ROOT=/home/ctj/cbf_ttt_dynamic_binding_v1 \
MODEL=/home/ctj/models/Qwen3-4B \
CORPUS=/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
bash scripts/run_cbf_dynamic_binding.sh
```

Independent audit:

```bash
/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
  scripts/audit_cbf_dynamic_binding.py \
  --root /home/ctj/cbf_ttt_dynamic_binding_v1 \
  --stage formal
```

## 9. Baseline compatibility

The original In-Place TTT model, pretraining configs, runtime, dataset loaders, evaluation configs, and baseline entry points are unchanged. All new behavior lives in opt-in task and script entry points. There are no new Python dependencies. Existing baseline configurations therefore retain their original behavior.

## 10. Completed, not completed, and next work

Completed:

- constructed and packed 128/32 train/dev worlds with equal FineWeb-Edu and LongCrawl64 backgrounds;
- kept 32 test worlds sealed;
- completed smoke and formal two-arm training;
- evaluated fixed checkpoints under Correct, Wrong, Empty, twin-anchor, and Full-KV conditions;
- recorded per-update/per-evaluation time and peak memory remotely;
- independently audited all formal accounting and provenance;
- archived compact results and hashes in Git.

Not completed by design:

- test-set scoring;
- forgetting-label construction;
- forgetting-controller training;
- end-to-end forgetting evaluation;
- multi-seed confirmation.

Recommended next experiment:

1. Keep the same sealed split and joint-context schedule.
2. Change the memory representation/readout so the writer is explicitly trained to retrieve entity-conditioned values, for example a key-query contrast over all four facts or a small associative retrieval head on the writer state.
3. Compare correct, within-world permuted, cross-world Wrong, twin, and Empty memories. Require correct-versus-wrong confidence separation before exact generation.
4. Use the 2,048-exposure checkpoint as a diagnostic only, then repeat the chosen architecture with a fresh seed and a fixed endpoint.
5. Resume forgetting-oracle and policy experiments only after the writer passes the content gate.

## 11. Risks

- The experiment uses one semantic seed and 32 development worlds; small likelihood effects can be unstable.
- The binding and CE arms are exposure matched but not compute matched.
- Only the anchor receives a twin contrast; the other three facts receive digit weighting without a direct negative memory.
- Exact random-code generation is harder than ranking candidate values. A future candidate-ranking diagnostic may localize failure, but it must not replace the preregistered generation gate.
- Natural backgrounds are sampled from the available mixed 1B corpus, so conclusions do not automatically transfer to the full 20B pretraining distribution.
- A frozen original backbone with newly trained writers is a controlled prerequisite test, not a complete end-to-end In-Place TTT training result.
