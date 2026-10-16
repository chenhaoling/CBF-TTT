# Native Layerwise Fast-Weight Admission V3: frozen fresh-source plan

Status: **frozen before V3 corpus materialization and scoring**  
Freeze date: 2026-10-16 (Asia/Shanghai)  
Protocol: `native_layerwise_fastweight_admission_v3`

## Registered question and claim boundary

V1 established that the scalar action which admits all seven native TTT-layer
updates is harmful on the development pilot.  V2 subsequently found audited
development headroom when exactly one layer update is admitted.  V3 asks
whether that layer-selective headroom replicates on source-disjoint worlds and,
only if it does, whether decision-time support features can predict the useful
layer.

V3 changes the intervention from one scalar gate to seven binary gates:

```text
W_fast,t,l = W_fast,t-1,l + g_t,l * delta_W_t,l,
g_t,l in {0, 1}, l in {0, 6, 12, 18, 24, 30, 35}.
```

The Qwen3-4B backbone and the native `ttt_conv/ttt_proj` writer remain frozen.
No positive V3 result can rescue the failed claim that a single whole-update
gate is useful.  The strongest possible V3 claim is selective admission of
individual native TTT-layer updates.

## Source-locked corpus and partitions

Before loading the model, construct and independently audit 624 source groups
from the already acquired LAMA/ParaRel train partition using the existing
semantic-record lifecycle generator.  The allocator must exclude every V1/V2
pilot source fact and subject and must make all material fact hashes and subject
identifiers disjoint across groups.  Each group has three sparse boundaries;
only twin zero is converted to model input.

The six mechanisms and source mappings remain fixed:

```text
novel_supporting       <- same_subject_different_relation
useful_repeat          <- semantic_duplicate
exact_duplicate        <- explicit_reset_revoke, converted to literal duplicate
irrelevant_distractor  <- different_subject_same_relation
conflicting_overwrite  <- semantic_overwrite
stale_repetition       <- stale_repetition
```

Within each mechanism, groups are ordered by SHA256 of
`(2026101701, source_group)` and assigned without replacement:

* fresh confirmation: 8 per mechanism, 48 groups;
* controller train: 64 per mechanism, 384 groups;
* development: 16 per mechanism, 96 groups;
* sealed test: 16 per mechanism, 96 groups.

The manifest stores only opaque group/fact hashes for the online path.  Plain
statements, answers and provenance live in a separate offline audit artifact.
No V3 model output is consulted while constructing or partitioning the corpus.

## Fixed actions, branches and objective

At every boundary, reconstruct the same `M_(t-1)` and make nine fresh-KV
branches:

```text
reject_all,
admit_layer_0, admit_layer_6, admit_layer_12, admit_layer_18,
admit_layer_24, admit_layer_30, admit_layer_35,
admit_all.
```

The seven singleton branches take one layer from the native admit branch and
six layers from the reject branch.  Hash audits must prove this composition.
V3 performs no pair, arbitrary-subset or per-layer-scale search.  Such a search
would require a separately frozen experiment after V3 and is outside this
protocol.

For action `a`, using the same five future queries:

```text
J(a) = 0.5 * mean related answer NLL
     + 0.5 * mean protected answer NLL
B_l  = J(reject_all) - J(admit_layer_l).
```

The safe oracle considers a singleton only if its protected accuracy is no
lower than reject-all.  Ties reject.  Bootstrap resamples source groups with
20,000 draws and seed `2026101702`.

## Fresh-confirmation gate and stopping rule

Run a 12-group smoke (two groups per mechanism), record per-label time and peak
CUDA memory, then score all 48 confirmation groups.  Continue to controller
label construction only when all conditions hold on the fresh confirmation:

1. at least 15% of boundaries have `max_l B_l > 1e-4`;
2. safe singleton oracle mean J is below reject-all;
3. paired source-group bootstrap 95% CI for reject-all minus oracle J has a
   strictly positive lower bound;
4. protected accuracy is no lower than reject-all;
5. protected NLL rises by at most 0.01 and EOS rate rises by at most 1 point;
6. branch, composition, corpus, dependency, resource and independent audits
   all pass.

Failure stops V3 before train/dev/test label construction.  The sealed test is
not opened under any confirmation failure.

## Controller inputs and labels

The decision-time feature contract is frozen before confirmation scoring.  It
reuses the V1 numeric feature extractor:

* a fixed 128-dimensional signed projection at every TTT layer;
* support centroid and diagonal variance;
* cosine and diagonal-Gaussian likelihood against numeric recent support;
* current chunk loss and prediction uncertainty;
* per-layer fast-weight norm, update norm, cosine and relative geometry;
* numeric position/support counts.

The controller may not receive plaintext, token ids, subject/relation/value,
mechanism, source id, future score, answer, label, oracle action, candidate
rank, or semantic audit data.  Online feature rows and offline action labels
are physically separate and joined by row SHA256.

For every singleton, the offline targets are benefit `B_l`, protected-NLL
change and protected-accuracy harm.  Full models use all permitted features.
The no-support ablation removes projection, centroid, variance, cosine and
likelihood fields.  The support-only ablation retains those fields and numeric
position/count fields while removing current-loss, uncertainty and geometry.

## Frozen controller training

Train action-conditioned utility/risk regressors with seeds
`2026101703`, `2026101704`, and `2026101705`.  Inputs are standardized from the
384 training groups only.  The network has shared widths `[256, 64]`, GELU,
dropout 0.1, seven benefit outputs, seven protected-NLL outputs and seven harm
logits.  Use AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 128,
80 epochs and gradient-norm cap 1.0.  No hyperparameter search is allowed.

For each boundary, a singleton is eligible when predicted benefit is positive,
predicted protected-NLL change is at most zero and predicted harm probability
is below 0.5.  Choose the eligible layer with maximum predicted benefit;
otherwise reject.  Development is used only for the registered release gate,
not to tune thresholds, epochs or features.

## Registered baselines and metrics

Evaluate the following on identical branches:

* always-write (`admit_all`) and never-write (`reject_all`);
* train-fitted loss-threshold and update-norm-threshold rules;
* support-in, novelty-only and duplicate-skip numeric heuristics;
* no-support learned model;
* support-only learned model;
* full learned model;
* safe singleton oracle.

Threshold baselines choose their threshold and direction by training-group J
only and are then frozen.  Heuristics may use only registered numeric online
features.  Report mean J, related/protected accuracy and NLL, EOS rate, action
rate, layer frequencies, oracle-gap closure and paired source-group bootstrap.

## Development release and one-time sealed test

All three full-model seeds must satisfy every condition on development before
test release:

1. lower J than always-write and never-write;
2. both paired source-group bootstrap 95% intervals are strictly positive;
3. close at least 30% of the safe-oracle gap from the stronger fixed baseline;
4. protected accuracy falls by at most 0.5 percentage point;
5. protected NLL rises by at most 0.01;
6. EOS rate is at most 1 percentage point above never-write;
7. full model is better than the no-support learned model;
8. all execution and independent audits pass.

Only then may the 96 sealed-test groups be scored once.  Test success requires
the same conditions for all three seeds.  Test data cannot select a seed,
threshold, epoch, feature set, action set or retry.  A failed development gate
leaves test model inputs and semantic audit unopened by scoring code.

If a learned gate beats always-write but not never-write, the conclusion must
be: “support gate减少了有害更新，但当前native writer尚未产生可验证的净正内容收益”.

## Independent audit and compatibility

A second program must rebuild and verify corpus hashes and split isolation;
feature/label separation; parent/reject/admit and singleton hashes; action J
and targets; checkpoint/config hashes; actions; all metrics and bootstraps;
CUDA resources; unchanged backbone parameters; and the test-release decision.

All V3 code and configuration are opt-in.  Baseline entry points, native
writer code, Qwen weights, optimizer state and prior artifacts are immutable.
