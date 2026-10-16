# Native Support-Aware Fast-Weight Update Admission V1: frozen plan

Status: **frozen before any V1 model scoring**  
Freeze date: 2026-10-16 (Asia/Shanghai)  
Protocol id: `native_support_aware_fastweight_admission_v1`

## 1. Question and claim boundary

This experiment asks whether information available at the instant of a native
TTT update can predict whether that update should be committed:

```text
W_fast,t = W_fast,t-1 + g_t * delta_W_t,  g_t in {0, 1}.
```

The Qwen backbone, `ttt_conv`, `ttt_proj`, and their learned checkpoint weights
remain frozen. The online memory is only the native dense fast weight stored in
`TTTDynamicCache.ttt_states[layer][2]`. No plaintext record store, teacher,
answer, future outcome, mechanism id, subject, relation, value, or oracle action
is available to the controller.

The primary claim is limited to **admission of native fast-weight updates**.
Forgetting an already committed update is outside this V1 action space. If the
learned gate beats always-write but not never-write, the required conclusion is:

> support gate减少了有害更新，但当前native writer尚未产生可验证的净正内容收益

## 2. Native branch intervention

Every scored boundary starts from one immutable native cache `C_(t-1)` at a
complete 4,096-token boundary. Two deep copies consume the exact same incoming
chunk:

* reject: temporarily set every native TTT layer's `ttt_lr` to zero;
* admit: use the checkpoint's frozen native `ttt_lr=0.3`.

Both branches use the repository's original Qwen forward and original
`ttt_conv/ttt_proj` writer. The learning rate is restored after each forward.
The parent cache is not mutated. The branches must have identical token input,
pre-boundary fast-weight hash, current-chunk hidden-state score, KV sequence
length, and non-fast-weight native state. Reject must retain the exact previous
fast-weight hash; admit must equal the writer-produced native update. A mismatch
invalidates the row.

Future queries are scored from a fresh attention KV cache initialized with the
branch's native `W_fast`. Query prompts and candidate answers are identical
between branches and shorter than one TTT chunk, so scoring cannot create
another native update. This isolates fast-weight content from plaintext context.

## 3. Data and mechanisms

The data builder reuses the repository's audited LAMA/ParaRel source inventory
and templating utilities. It constructs independent synthetic session worlds
from real relation facts. Source rows, normalized subjects, relation partitions,
and document/world identifiers are disjoint across pilot, train, development,
and sealed test.

There are exactly six mechanisms:

1. `novel_supporting`: a new fact needed by future queries;
2. `useful_repeat`: a correct fact is repeated and remains useful;
3. `exact_duplicate`: a redundant exact repeat adds no new target;
4. `irrelevant_distractor`: an unrelated record arrives while old facts remain
   the future target;
5. `conflicting_overwrite`: a new value supersedes an earlier value;
6. `stale_repetition`: an obsolete value is repeated after a correction.

Each world contains one setup chunk and three sparsely selected update
boundaries. Every chunk has exactly 4,096 tokens. The target statement and its
fixed paraphrases are placed at deterministic separated offsets among filler
drawn from that world's source document. The three boundaries are fixed before
scoring. Each boundary has two related queries and three protected/unrelated
queries, with identical candidate sets across actions.

Stages:

| Stage | Source groups | Boundaries/group | Use |
|---|---:|---:|---|
| smoke | 2 per mechanism (12 total) | 3 | path, invariants, time and VRAM only |
| resource/oracle pilot | 8 per mechanism (48 total) | 3 | headroom gate |
| formal train | 384 | 3 | labels and controller fitting |
| formal development | 96 | 3 | frozen success gate |
| sealed test | 96 | 3 | one release after development passes |

Smoke groups are included in the 48 pilot groups and cannot enter a formal
split. Pilot source material cannot enter any formal split.

## 4. Allowed online features

All feature extraction occurs after the incoming chunk has been observed and
before choosing whether its native `delta_W` is retained. Features are detached
and never backpropagated into Qwen.

For each of the seven TTT layers, a frozen signed CountSketch projection maps
the MLP input activation to 128 dimensions. Projection seed, signs, bucket map,
and normalization are fixed by `(protocol id, layer index, seed=2026101601)`.
Only these summaries are retained:

* current projected centroid (128 values);
* current projected diagonal variance (128 values, stored as `log1p(var)`);
* cosine to the most recent activation-support centroid;
* diagonal-Gaussian mean log likelihood under the most recent support summary;
* a `has_recent_support` indicator.

The recent-support state contains only centroid and diagonal variance. It is
updated after observing every chunk regardless of the admission action. It
contains no token ids, strings, hidden sequence, subject, relation, or value.

Additional allowed features are:

* current next-token mean NLL;
* current predictive entropy, top-1 probability, and top-1/top-2 margin;
* per-layer relative old-fast-weight norm, relative update norm,
  old/update cosine, update stable rank proxy, and max/mean singular-value
  proxy computed without future data;
* boundary position and a constant bias.

The full model receives all allowed features. `no-support` receives current
loss, uncertainty, geometry, and position. `support-only` receives only the
support summaries, recent-support statistics, position, and bias. Mechanism
names and all semantic audit fields remain in a separate offline manifest and
are never joined into controller tensors.

Online feature/decision artifacts may contain hashes, numeric features, resource
statistics, and native weight geometry only. Fields named `memory`,
`online_memory`, or `support_state` must contain numeric sufficient statistics
and must never contain token ids or text. The independent audit recursively
rejects strings or integer token sequences in these fields.

## 5. Labels and objective

For action `a`, define:

```text
J(a) = 0.5 * mean related answer NLL
     + 0.5 * mean protected/unrelated answer NLL.
B = J(0) - J(1).
```

Positive `B` favors admit; negative `B` favors reject. Every row stores complete
per-query outcomes, `J(0)`, `J(1)`, and `B`, plus action-specific candidate
accuracy, unrelated accuracy/NLL, first-token EOS rate, timing, and peak CUDA
allocated/reserved memory. Future outcomes and `B` exist only in the offline
label file. They are forbidden from feature and inference artifacts.

`safe_oracle` selects the lower-J action only if it does not reduce that row's
protected-query accuracy relative to the safer action; ties select reject.
Always-write and never-write are rolled out from the same source groups. For
later boundaries, the canonical label state follows always-write, so all labels
measure one-step admission utility on a fixed, policy-independent state
distribution.

A boundary is positive when `B > 1e-4`, negative when `B < -1e-4`, and tied
otherwise. The tolerance is fixed before observing scores.

## 6. Pilot gate and stopping rule

Smoke has no efficacy decision. It passes only when all branch/hash/fresh-KV
invariants hold, all scores are finite, all required fields exist, no forbidden
field leaks, and per-label time plus peak allocated/reserved VRAM are reported.

The 48-group resource/oracle pilot passes only if all conditions hold:

1. positive boundaries are at least 15% and negative boundaries are at least
   15% of the 144 boundaries;
2. safe-oracle mean J is lower than the stronger aggregate baseline among
   always-write and never-write;
3. the paired source-group bootstrap 95% interval for that J improvement has a
   strictly positive lower bound (20,000 draws, seed 2026101602);
4. protected accuracy is no lower than the stronger baseline, protected NLL
   rises by at most 0.01, and EOS rate rises by at most 1 percentage point;
5. execution and independent pilot audit both pass.

If any pilot condition fails, execution stops. No controller is trained, no
formal data is labeled, and sealed test is not generated or opened. Runtime
projections, label throughput, and observed VRAM determine the formal shard
count; no scale decision is made before the resource pilot.

## 7. Controller and baselines after a passing pilot

Three utility regressors use seeds `2026101611`, `2026101612`, and
`2026101613`. Each predicts continuous `B`; action is admit iff predicted
benefit is greater than zero. Feature standardization is fit on train only.
The regressor is a two-layer MLP with widths 128 and 32, GELU, dropout 0.1,
Huber loss, AdamW, group-balanced batches, and early stopping selected by a
source-group split inside train. No threshold is tuned on development.

Frozen comparisons are:

* always-write and never-write;
* train-calibrated loss threshold and update-norm threshold;
* `support-in` (train-calibrated support likelihood threshold);
* `novelty-only` (train-calibrated recent-support cosine threshold);
* `duplicate-skip` (reject above a train-calibrated cosine threshold);
* learned `no-support`, learned `support-only`, learned full model;
* safe oracle.

Threshold direction and value are chosen by grouped cross-validation on train
only and then frozen. All policies act on identical cached boundary rows.

Oracle-gap closure for a policy is

```text
(J(stronger always/never) - J(policy)) /
(J(stronger always/never) - J(safe oracle)).
```

## 8. Development gate and sealed test

The test partition is represented before release only by source allocation
commitments, counts, and salted hashes. Its raw examples, labels, and metrics
must not be loaded by training, calibration, or development commands.

Development passes only when all formal success conditions below hold on the
96 development source groups. Only then may a one-shot release command record
the plan hash, data commitment hash, checkpoint hashes, development decision,
operator timestamp, and `test_release_count=1`, construct/label the 96 test
groups once, and score the already frozen checkpoints. A failed development
gate permanently records `test_release_count=0` for V1.

## 9. Formal success conditions

Success requires all of the following on the one-shot sealed test:

* every one of three full-model seeds has lower paired J than both always-write
  and never-write;
* for each seed, the two paired source-group bootstrap 95% intervals versus
  always-write and versus never-write are strictly positive;
* every seed closes at least 30% of the safe-oracle gap;
* protected/unrelated accuracy falls by no more than 0.5 percentage point from
  the stronger always/never baseline;
* protected/unrelated NLL increases by no more than 0.01;
* first-token EOS rate is no more than 1 percentage point above never-write;
* the full model is better than the corresponding learned `no-support` model
  in paired mean J, with a strictly positive grouped bootstrap interval;
* all execution and independent audit checks pass.

`support-only` and heuristic baselines are diagnostic and do not replace any
success condition.

## 10. Independent audit and immutable outputs

The independent audit recomputes, without importing the collector's summary
functions:

* plan, source, tokenizer, data, feature, label, model, and checkpoint hashes;
* split/world/document/subject/fact disjointness;
* native cache parent immutability and reject/admit branch consistency;
* projection definitions and every numeric feature from raw numeric hook
  summaries;
* `J(0)`, `J(1)`, `B`, actions, metrics, grouped bootstrap intervals, resource
  totals, and test release state;
* frozen-backbone parameter hashes and parameter version counters;
* absence of forbidden controller fields and plaintext online memory.

All new behavior is opt-in under
`experiments/cbf_ttt/support_write_gate_v1/`. Baseline configuration, training
entry points, original checkpoints, and default native inference behavior are
unchanged. Final results, including failures and stopped stages, are written to
`experiments/cbf_ttt/support_write_gate_v1/REPORT.md` and summarized in
`MODIFICATION_LOG.md`.
