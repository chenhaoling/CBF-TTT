# Native Support-Aware Fast-Weight Update Admission V1

## Status

**Stopped at the preregistered resource/oracle pilot.** The final, independently
audited pilot contains 48 source groups, six mechanisms, and 144 sparse
boundaries. Every boundary has negative native-write benefit. The safe oracle
selects reject for all 144 boundaries and exactly equals never-write. Per the
frozen stop rule, no controller was trained, formal train/dev data was not
labeled, and sealed test was not generated or opened.

This is a negative result for learning a nontrivial admission policy with the
current checkpoint, native writer, and constructed V1 corpus. It does not show
that support-aware admission is impossible after a writer or data-distribution
change.

## 1. Objective and implementation assumptions

The experiment tests the binary native update rule

```text
W_fast,t = W_fast,t-1 + g_t * delta_W_t,  g_t in {0, 1}.
```

The only session memory is the dense weight in
`TTTDynamicCache.ttt_states[layer][2]`. The Qwen3-4B backbone and the original
seven-layer `ttt_conv/ttt_proj` writer remain frozen. No plaintext memory or
teacher is used.

At each complete 4,096-token boundary, two native caches are cloned from the
same parent. Reject consumes the chunk with native `ttt_lr=0`; admit consumes
the same chunk with the checkpoint value `ttt_lr=0.3`. Both use the original
Qwen forward. Queries start from fresh attention KV and receive only the
branch's native `W_fast`. The current output, parent cache, reject weight, and
branch identities are checked for every row.

## 2. Frozen plan and identities

The plan was committed and pushed before V1 scoring:

* plan: `SUPPORT_AWARE_WRITE_GATE_V1_PLAN.md`;
* plan commit: `f7ea4117`;
* plan SHA256: `3556f5e3a221ce2a4cd171bf3ac1c61f0402ccc7879a5822ba8a97feef1188b1`;
* final data manifest SHA256:
  `abb279ab2d99f2edbfbf7870f3eff98e37b3ffb447164226a454cc1f0816a3fe`;
* checkpoint: final 1B-token Qwen3-4B, global step 81,381;
* checkpoint parameter count: 4,068,432,896;
* checkpoint parameter SHA256:
  `1947114f49a948c0f40202abb2d098fc8d68b339bc46c95333ab4221dfe00eb4`;
* native configuration: layers `[0,6,12,18,24,30,35]`, chunk 4,096,
  `ttt_lr=0.3`, hidden-state target, bfloat16, SDPA;
* hardware: two NVIDIA RTX 5090 GPUs, 32,607 MiB each.

The independent audit also fixes SHA256 for `merges.txt`, `tokenizer.json`,
`tokenizer_config.json`, `vocab.json`, both checkpoint safetensor shards, source
episodes, features, labels, and summaries.

## 3. Constructed data

The builder derives source-disjoint worlds from the existing independently
audited LAMA/ParaRel semantic-record pilot. The source episode hash is
`22c9f007c305426d2d26e4bc7066645e7427d25ef02a8d06d61198bc329c86dc`.
It creates eight groups for each mechanism:

1. `novel_supporting`;
2. `useful_repeat`;
3. `exact_duplicate`;
4. `irrelevant_distractor`;
5. `conflicting_overwrite`;
6. `stale_repetition`.

Each group contains one setup chunk and three incoming chunks. Statements are
placed repeatedly at deterministic positions until the native 4,096-token
boundary. Each boundary has two related and three protected queries. Semantic
text stays in `data/semantic_audit.jsonl`; model inputs contain token ids and
opaque hashes. Controller features contain no token ids or text.

Smoke uses two groups per mechanism. The resource/oracle pilot uses all eight
groups per mechanism. Pilot data is excluded from every proposed formal split.

## 4. Allowed feature path

A read-only MLP forward pre-hook observes the actual input activation at every
TTT layer without editing the Qwen model file. A fixed signed modulo CountSketch
with seed `2026101601` maps each layer to 128 dimensions. The stored support
statistics are current centroid, `log1p` diagonal variance, cosine to the most
recent centroid, diagonal-Gaussian support likelihood, and a recent-support
indicator. Recent support contains numeric centroid/variance only.

The other features are current token NLL, predictive entropy, top-1 probability,
top-1/top-2 margin, boundary position, and five native fast-weight geometry
statistics per layer. Total feature width is 1,854.

Online features and offline counterfactual labels are physically separated:

* `*_features.jsonl`: only allowed decision-time numeric features;
* `*_labels.jsonl`: `J(0)`, `J(1)`, `B`, query outcomes, resource use, and a
  SHA256 binding to the matching feature row.

The independent audit reports zero plaintext online-memory fields.

## 5. Smoke and resource profile

| Stage | Groups | Labels | Mean seconds/label | Peak allocated | Peak reserved | Execution audit |
|---|---:|---:|---:|---:|---:|---|
| Smoke | 12 | 36 | 5.789 | 10.433 GiB | 11.266 GiB | pass |
| Pilot | 48 | 144 | 5.818 | 10.433 GiB | 11.363 GiB | pass |

Pilot aggregate label time is 837.80 GPU-seconds. The two-GPU label critical
path is 418.91 seconds, excluding process/model startup and final audit. No OOM,
nonfinite score, model-parameter mutation, branch mismatch, or missing row
occurred.

Focused tests passed on `hku-gpu2`:

* six new data/feature/native-branch tests;
* three original native TTT train/inference tests;
* ten original CBF runtime/controller tests.

## 6. Oracle/headroom result

The frozen objective is

```text
J(a) = 0.5 * mean related answer NLL
     + 0.5 * mean protected answer NLL
B = J(0) - J(1).
```

| Quantity | Result |
|---|---:|
| Positive boundaries (`B > 1e-4`) | 0 / 144 (0%) |
| Negative boundaries (`B < -1e-4`) | 144 / 144 (100%) |
| Ties | 0 / 144 |
| Mean `B` | -4.43129 |
| Median `B` | -3.67078 |
| `J(always-write)` | 11.99533 |
| `J(never-write)` | 7.56404 |
| `J(safe-oracle)` | 7.56404 |
| Oracle gain over stronger baseline | 0.00000 |
| Paired source-group bootstrap 95% CI | [0.00000, 0.00000] |
| Safe-oracle actions | reject 144, admit 0 |

All six mechanisms independently contain 24/24 negative boundaries:

| Mechanism | Mean B | Never-write J | Always-write J |
|---|---:|---:|---:|
| novel supporting | -3.91132 | 8.70671 | 12.61803 |
| useful repeat | -4.65003 | 7.26476 | 11.91479 |
| exact duplicate | -5.05206 | 6.66450 | 11.71656 |
| irrelevant distractor | -4.67347 | 6.70407 | 11.37755 |
| conflicting overwrite | -4.02746 | 8.79121 | 12.81866 |
| stale repetition | -4.27341 | 7.25297 | 11.52638 |

Admitting updates also reduces overall candidate accuracy from 55.69% to
41.39%, reduces protected accuracy from 63.66% to 46.06%, raises protected NLL
from 6.9221 to 11.6267, and raises first-token EOS rate from 10.97% to 44.31%.

## 7. Frozen gate decision

The pilot fails three mandatory checks:

* positive-benefit support is 0%, below 15%;
* safe oracle does not beat never-write;
* the oracle-gain bootstrap lower bound is not strictly positive.

Safety bookkeeping and all engineering/audit checks pass, but they cannot
override the missing headroom. Therefore:

* utility regressor training: **not run**;
* three controller seeds: **not created**;
* heuristic and learned controller comparison: **not run**;
* formal 384/96/96 construction and labeling: **not run**;
* development release gate: **not reached**;
* sealed test release count: **0**;
* formal success criteria: **not evaluated**.

The correct scientific conclusion is that the current native writer does not
produce a useful mixture of admissible and rejectable updates on this V1
corpus. A learned support gate has no oracle gap to close beyond the trivial
never-write action. This result cannot support a claim that adaptive TTT works.

The plan's conditional wording remains the claim boundary for a future setting:

> support gate减少了有害更新，但当前native writer尚未产生可验证的净正内容收益

For this run, even that learned-gate comparison was not authorized because the
oracle stage selected reject everywhere.

## 8. Invalidated engineering runs

Three preliminary directories were retained remotely and excluded from every
metric:

1. the first partial pilot colocated features and oracle labels in one JSONL;
2. the next smoke exposed unstable canonical hashes caused by integer layer
   keys changing to strings during JSON round trip;
3. the next partial pilot fixed separation/hashing but lacked file-level
   tokenizer hashes.

Each issue was detected before accepting the formal pilot. The final `r3`
directory rebuilt inputs, reran smoke, reran all 144 labels, and passed the
independent audit. No preliminary efficacy result was used to alter the frozen
objective or gate.

## 9. Reproduction

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927

export PYTHON_BIN=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python
export REPO=$PWD
export MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
export TOKENIZER=/home/ctj/models/Qwen3-4B
export SOURCE_EPISODES=/home/ctj/cbf_ttt_utility_semantic_record_pilot_data_20261016_retry1/pilot_episodes.jsonl
export OUTPUT_ROOT=/home/ctj/cbf_ttt_support_write_gate_v1_reproduction

STAGE=smoke NUM_SHARDS=2 bash scripts/run_native_support_write_gate_v1.sh
STAGE=pilot NUM_SHARDS=2 bash scripts/run_native_support_write_gate_v1.sh

$PYTHON_BIN -m scripts.detail_native_support_write_gate_v1 \
  --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/pilot_detail.json"
```

The runner refuses to overwrite existing outputs. Use a new output directory
for every reproduction.

## 10. Compatibility, limitations, and next work

All V1 behavior is opt-in. No baseline model, native Qwen implementation,
training entry point, configuration default, or checkpoint was modified.

Main limitations:

* the corpus repeats compact relation statements to fill native 4,096-token
  chunks; this is controlled and auditable, but differs from natural documents;
* query evaluation is candidate NLL under fresh KV and does not cover open-ended
  generation quality;
* the result is specific to this final 1B-token checkpoint and its current
  native writer scale;
* no learned gate can be assessed without both positive and negative update
  utility.

The next defensible experiment must change the source of `delta_W`, its scale,
or the chunk/data construction in a newly frozen protocol and then repeat the
oracle/headroom gate on fresh source groups. Reusing these 48 groups to tune a
new writer would turn them into development data and require a new held-out
pilot for any claim.

## 11. Subsequent layerwise diagnostics

The V1 conclusion for a scalar whole-update gate remains unchanged.  V2 tested
whether useful layer updates were hidden inside the harmful bundled write.  On
the V1 development groups, a safe singleton oracle improved J by 0.08592 with
source-group bootstrap interval `[0.06418, 0.11321]`; 91.67% of boundaries had
singleton headroom.  All-layer admit remained harmful.  V2 passed as a
development diagnostic only.

V3 then generated 624 new source-disjoint groups and locked 48/384/96/96
confirmation/train/development/test partitions.  Fresh confirmation reproduced
headroom: reject J 7.55667, all-layer admit J 11.95417, singleton oracle J
7.46663, gain 0.09005, bootstrap interval `[0.07071, 0.11229]`.  The registered
V3 gate nevertheless failed because oracle EOS rose by 1.25 points, above the
1-point limit.  V3 stopped before controller training and left train/dev/test
unscored.

A post-stop diagnostic added a rowwise EOS nonincrease barrier.  It changed
7/144 actions and retained gain 0.08059 with interval `[0.06412, 0.09991]`
while reducing EOS by 0.278 point.  This result motivated the separately frozen
V4 EOS-risk controller protocol; it does not retroactively pass V3.
