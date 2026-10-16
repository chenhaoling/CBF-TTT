# Native Layerwise Fast-Weight Admission V2: frozen diagnostic plan

Status: **frozen before V2 scoring**  
Freeze date: 2026-10-16 (Asia/Shanghai)  
Protocol: `native_layerwise_fastweight_admission_v2`

## Question

V1 found that admitting the simultaneous native update at all seven TTT layers
was harmful at every one of 144 boundaries. V2 tests a narrower causal
hypothesis: the full update may be harmful because useful and harmful layer
updates are bundled together.

For TTT layer `l`, V2 evaluates

```text
W_fast,t,l = W_fast,t-1,l + g_t,l * delta_W_t,l,
g_t,l in {0, 1}.
```

This is an oracle/headroom diagnostic before controller training. It does not
train a layer gate and cannot establish a held-out effectiveness claim.

## Frozen substrate and interventions

The Qwen3-4B backbone, native `ttt_conv/ttt_proj`, checkpoint, chunk size,
writer learning rate, future queries, objective, and V1 data remain unchanged.
The only memory is `TTTDynamicCache.ttt_states[layer][2]`.

At each boundary, V2 deterministically reconstructs the same V1 parent and
reruns the same reject/full-admit native branches. Parent, reject, and admit
fast-weight hashes must exactly equal the independently audited V1 hashes.
V2 then composes seven singleton actions using the reject branch at six layers
and the full-admit branch at one layer. The action set is fixed:

```text
reject_all, admit_layer_0, admit_layer_6, admit_layer_12,
admit_layer_18, admit_layer_24, admit_layer_30, admit_layer_35, admit_all.
```

Every singleton query uses a fresh attention KV cache and the same candidate
answers. Reject-all and admit-all outcomes are reused from the immutable V1
labels only after exact branch-hash matching. This avoids rescoring known arms
while preserving the same causal state.

## Data and stages

V2 reuses V1's already observed 48 source groups. This makes V2 a development
diagnostic. A positive result would require a fresh-source confirmation before
any paper claim.

* smoke: two groups per mechanism, 12 groups, 36 boundaries;
* diagnostic pilot: eight groups per mechanism, 48 groups, 144 boundaries;
* each boundary: seven newly scored singleton actions plus the two audited V1
  reference actions.

No formal or sealed-test source is generated in V2 unless the diagnostic gate
passes and a separate confirmation plan is frozen.

## Objective and safe oracle

The action objective is identical to V1:

```text
J(a) = 0.5 * mean related answer NLL
     + 0.5 * mean protected answer NLL.
B_l = J(reject_all) - J(admit_layer_l).
```

The safe singleton oracle chooses the lowest-J singleton only when its protected
accuracy is no lower than reject-all; otherwise it rejects. Ties reject.

A boundary has singleton headroom when `max_l B_l > 1e-4`. Bootstrap resamples
the 48 source groups with 20,000 draws and seed `2026101603`.

## Diagnostic gate and stopping rule

Proceed to a separately frozen sparse layer-gate confirmation only if all
conditions hold:

1. at least 15% of boundaries have singleton headroom;
2. safe singleton oracle mean J is below reject-all;
3. paired source-group bootstrap 95% CI for reject-all minus oracle J has a
   strictly positive lower bound;
4. protected accuracy is no lower than reject-all;
5. protected NLL rises by at most 0.01 and EOS rate rises by at most 1 point;
6. all branch, composition, dependency, resource, and independent audits pass.

If the gate fails, do not search layer pairs, tune per-layer scales, train a
controller, or open any test source. Report that layerwise decomposition did
not expose usable native-write headroom.

If the gate passes, V2 itself still stops after reporting the diagnostic. A new
plan must freeze fresh source groups, pair/subset search, controller features,
training seeds, baselines, and release rules before further scoring.

## Features and privacy

V2 references the immutable V1 1,854-dimensional decision-time feature rows by
SHA256. Those rows contain activation support, current loss, uncertainty,
fast-weight geometry, position, and numeric recent-support state only. V2 does
not add semantic identifiers, plaintext history, token ids, future outcomes,
answers, mechanism labels, or oracle actions to controller inputs.

## Audit and compatibility

An independent program must recompute action objectives, singleton benefits,
safe actions, per-layer counts, grouped bootstrap, safety metrics, resources,
V1 dependency hashes, and composition invariants. It must verify that every
singleton differs from reject at exactly one native fast-weight layer and
matches full-admit at that layer.

All code is opt-in. Baseline training, Qwen model files, native writer modules,
V1 artifacts, configurations, and checkpoints are immutable.
