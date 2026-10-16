# Native EOS-Safe Layerwise Fast-Weight Admission V4: frozen formal plan

Status: **frozen before V4 train/development scoring**  
Freeze date: 2026-10-16 (Asia/Shanghai)  
Protocol: `native_eos_safe_layerwise_fastweight_admission_v4`

## Motivation and evidence status

V3 fresh-source confirmation is formally stopped.  Its accuracy-safe singleton
oracle retained strong headroom but raised EOS rate by 1.25 percentage points,
above the registered 1-point limit.  No V3 controller was trained and no V3
train, development or test group was scored.

A post-stop diagnostic, which is design evidence rather than confirmation,
added a rowwise condition that a singleton's EOS rate may not exceed reject.
It changed 7 of 144 actions while retaining mean J gain 0.080585 and a paired
source-group bootstrap interval `[0.064116, 0.099907]`; EOS changed by -0.278
point.  V4 freezes that safety mechanism before opening any formal outcome.

V3 confirmation remains a method-design set.  V4 claims must come from the
source-disjoint development and, if released, sealed-test partitions.

## Frozen data and native intervention

V4 reuses the already built and independently audited corpus with manifest
SHA256 `27c712ac9d23f958583313eb59f5a0debde0045cdd789ac17226c1b19d70c6f0`.
The 384 train, 96 development and 96 sealed-test source groups have not been
scored.  Their material fact hashes and subjects are globally disjoint from
each other and from the 48 V3 design groups.

The backbone, checkpoint, seven native `ttt_conv/ttt_proj` writers, chunk size,
writer learning rate and fresh-KV objective stay frozen.  Actions remain:

```text
reject_all,
admit_layer_0, admit_layer_6, admit_layer_12, admit_layer_18,
admit_layer_24, admit_layer_30, admit_layer_35,
admit_all.
```

No pair/subset or scale search is allowed.  Every singleton must differ from
reject at exactly one audited native fast-weight layer.

## Objective and EOS-safe oracle

```text
J(a) = 0.5 * mean related answer NLL
     + 0.5 * mean protected answer NLL.
B_l  = J(reject_all) - J(admit_layer_l).
```

The V4 safe oracle admits a singleton only when all hold:

1. `J(singleton) < J(reject_all)`;
2. protected accuracy is no lower than reject-all;
3. EOS rate is no higher than reject-all.

Ties reject.  This oracle labels headroom; its future metrics never enter a
controller input.

## Decision-time features and offline targets

The 1,854-dimensional input contract is unchanged: fixed 128-dimensional
activation projections, support centroid/diagonal variance, recent support
cosine and likelihood, current loss, prediction uncertainty, fast-weight
geometry, numeric position and bias.  Plaintext, token ids, source/mechanism
identity, subject/relation/value, future loss, answer, oracle action and all
outcome fields are prohibited.

Feature and label files stay physically separate and are bound by SHA256.  For
each singleton the offline targets are:

* benefit `B_l`;
* protected-NLL change relative to reject;
* protected-accuracy harm indicator;
* EOS-increase harm indicator.

## Frozen models and action rule

Train no-support, support-only and full variants with seeds `2026101801`,
`2026101802`, `2026101803`.  All use shared widths `[256,64]`, GELU, dropout
0.1, AdamW learning rate `1e-3`, weight decay `1e-4`, batch size 128, 80 epochs
and gradient-norm cap 1.0.  Inputs and continuous targets are standardized from
the 384 training groups only.  The output has seven benefit values, seven
protected-NLL changes, seven accuracy-harm logits and seven EOS-harm logits.

A singleton is eligible only when predicted benefit is positive, predicted
protected-NLL change is at most zero, predicted accuracy-harm probability is
below 0.5 and predicted EOS-harm probability is below 0.5.  Select the eligible
layer with maximum predicted benefit; otherwise reject.  Thresholds, epochs,
features and seeds cannot be selected on development.

## Baselines and development release

Compare always-write, never-write, train-fitted loss threshold, train-fitted
norm threshold, support-in, novelty-only, duplicate-skip, no-support learned,
support-only learned, full learned and EOS-safe oracle.  Heuristics use numeric
decision-time features only.

For every full-model seed, development must satisfy all conditions:

1. lower J than always-write and never-write;
2. paired source-group bootstrap 95% lower bounds versus both are positive;
3. close at least 30% of the EOS-safe oracle gap from the stronger fixed
   baseline;
4. protected accuracy falls by at most 0.5 percentage point from never-write;
5. protected NLL rises by at most 0.01;
6. EOS rate is at most 1 percentage point above never-write;
7. full is better than the matched-seed no-support model;
8. all corpus, branch, feature, label, checkpoint, metric, bootstrap and
   resource audits pass.

All three seeds must pass.  Development cannot tune or retry a model.

## Sealed test and stopping rules

If any development condition fails, stop V4 and leave all 96 test groups
unscored.  If development passes, score sealed test exactly once with all
checkpoints and thresholds frozen.  Test success uses the same eight
conditions for every seed.

If the controller beats always-write but not never-write, report exactly:
“support gate减少了有害更新，但当前native writer尚未产生可验证的净正内容收益”.

All V4 paths are opt-in.  Baseline configuration, training entry points, Qwen
weights, optimizer state, native writer source and all V1--V3 artifacts remain
immutable.
