# In-Place TTT 2xRTX5090 100M-tier RULER plan

Status: **frozen before any 100M-tier RULER score is produced**

Freeze date: 2026-10-18 (Asia/Shanghai)

Public-corpus plan SHA256:
`c5019a28b9fa7fa27052ddf1d69eb029a2ec5b3b6446f1527e9ceb7db1a12bab`

Parent Level-B RULER addendum SHA256:
`a5762933f8c6f9eee2863e42ba6823bbad922433fb00cf6e872630fe56543e7b`

Registered OpenCompass config SHA256:
`4c67903efb85881488ba2e8edb648199d404ad989df4dca6085621ab0f60b7db`

## 1. Scope and checkpoints

Evaluate only the paired final HF checkpoints released by the independent
`in_place_ttt_2x5090_100m_stage2_audit_v1` audit. The no-TTT and native-TTT
arms must retain their complete audited file manifests, original Qwen3-4B
tokenizer, registered `ttt_mode`, 131,072 maximum positions and identical
evaluation inputs. Resource or timing checkpoints are forbidden.

All GPU execution is restricted to `hku-gpu2`. A registered context pairs the
no-TTT arm on physical GPU 0 with the native-TTT arm on physical GPU 1. Both
GPUs must be idle RTX 5090 devices with at least 32,000 MiB total memory before
the pair starts. Require at least 32 GiB host `MemAvailable` and no active
public-corpus builder/auditor. Contexts run sequentially to avoid
cross-context contention.

## 2. Registered RULER matrix

Use the existing frozen OpenCompass config and source data. For each arm and
each context 4K, 8K, 16K, 32K, 64K and 128K, evaluate all 13 registered RULER
tasks with exactly 100 examples, seed 42, deterministic decoding,
`enable_thinking=false`, 64 maximum new tokens, the original tokenizer and
`fix_mistral_regex=false`.

Report all 13 task accuracies, the macro accuracy per context, `TTT - no-TTT`
paired deltas, and the mean over the six registered contexts. Raw predictions,
references, prompts, OpenCompass results, official summaries, logs, GPU traces
and complete dependency/model hashes are retained and independently audited.

## 3. Resource ladder and 256K boundary

Before opening the six-context matrix, run a one-example-per-task paired
resource ladder at 4K, 32K, 64K and 128K using the same final checkpoints and
decoding path. The resource score is discarded. Stop if either arm exits
nonzero, OOMs, changes the model/input binding or exceeds device capacity.

The paper-facing registered matrix ends at 128K because the Stage-2 model is
registered to 131,072 positions. A 256K run is an explicitly labelled YaRN
extrapolation and requires a separate resource decision. Failure at 256K must
be reported as a resource boundary; it must not remove samples, shorten inputs
or alter the task set to manufacture a score.

## 4. Release and claim boundary

The matrix remains sealed until both formal Stage-2 arms and their independent
audit pass, the paired RULER resource ladder passes through 128K, and all six
contexts can retain at least 5% GPU memory headroom. Any task/input mismatch,
missing example, tokenizer warning, nonfinite score, OOM or dependency drift
invalidates the affected pair and stops longer contexts.

Only the complete independently audited paired matrix supports a baseline
claim. Update-admission experiments remain sealed until this baseline result
is available; a smoke/resource score or a subset of contexts cannot release
the gate study.
