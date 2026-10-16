# Native Closed-Loop Fast-Weight Gate V5: frozen diagnostic plan

Status: **frozen before V5 model scoring**
Freeze date: 2026-10-16 (Asia/Shanghai)
Protocol: `native_closed_loop_fastweight_gate_v5`

## Motivation and evidence boundary

V4 trained a decision-time layerwise admission controller, but its development
metrics were computed from counterfactual actions whose parent state always
followed the native full-admit behavior trajectory.  Those metrics do not prove
that a gate remains useful after its own decisions alter later fast weights.

V5 evaluates the deployed recurrence directly:

```text
M_t^pi = Apply(M_{t-1}^pi, action_t^pi)
action_t^pi = pi(features(M_{t-1}^pi, x_t))
```

Every policy owns a separate native cache.  Its selected reject or singleton
state becomes the parent of its next boundary.  A policy may never reset to the
always-write trajectory between boundaries.

V5 is a post-stop method diagnostic on already observed design data.  It cannot
revive V4, release the V4 sealed test, or support a held-out efficacy claim.

## Frozen data and policies

Phase A uses the twelve `smoke_groups` already fixed in the independently
audited V3 corpus manifest: two groups from each of the six registered
mechanisms and three sparse boundaries per group.  The online scorer reads only
`confirmation_model_inputs.jsonl` plus those opaque group ids.  It does not read
the semantic audit or mechanism field.

The following eight policies are frozen:

1. `always_write`;
2. `never_write`;
3. V4 full seeds `2026101801`, `2026101802`, `2026101803`;
4. matched V4 no-support seeds `2026101801`, `2026101802`, `2026101803`.

All six learned checkpoints, feature standardizers, four-head outputs,
thresholds, and action rules are loaded without retraining or calibration.
Support-only models are excluded from Phase A because the registered causal
comparison is full versus matched no-support.

## Native intervention and decision features

The Qwen3-4B checkpoint, native `ttt_conv/ttt_proj` writer, writer learning
rate, chunk size, seven TTT layers, and fresh-KV query objective remain frozen.
Actions are reject-all or one admitted layer from `{0,6,12,18,24,30,35}`.
Always-write admits all seven layers only as a fixed baseline.

Learned policies use exactly the V4 1,854-dimensional contract: current loss,
prediction uncertainty, native fast-weight geometry, fixed 128-dimensional
activation projections, support centroid and diagonal variance, recent-support
cosine and likelihood, numeric position, and bias.  Recent support is maintained
separately along each policy trajectory.  Plaintext, token ids, source or
mechanism identity, subject/relation/value, answers, future outcomes, and oracle
actions are prohibited controller inputs.

Feature artifacts and post-action query outcomes are stored separately and
bound by canonical SHA256.  No online-memory field may contain text or token
sequences.

## Phase-A smoke gate

For each policy and source group, aggregate J and safety metrics across the
three sequential boundaries.  Phase A passes only if all conditions hold:

1. at least two of three full seeds have lower mean J than closed-loop
   never-write;
2. the mean gain of the three full seeds over never-write is positive;
3. at least two of three full seeds have lower J than their matched no-support
   policy;
4. every full seed loses at most 0.5 percentage point protected accuracy from
   never-write;
5. every full seed raises protected NLL by at most 0.01;
6. every full seed raises EOS rate by at most 1 percentage point;
7. every recorded action is reconstructed from its frozen checkpoint and
   decision-time feature;
8. every policy transition satisfies
   `next_fast_sha256[t] == parent_fast_sha256[t+1]`;
9. branch, composition, backbone, dependency, resource, and plaintext-leakage
   audits pass.

The smoke gate uses point estimates because it contains only twelve source
groups.  Bootstrap intervals are descriptive and cannot override a failed
condition.

## Conditional continuation and stopping

If Phase A fails, stop V5.  Do not run the 96-group closed-loop development
diagnostic, do not train or tune a controller, and do not open sealed test.

If Phase A passes, run the same frozen eight policies once on the already
observed 96-group V4 development split.  Before that run, freeze a separate
Phase-B gate requiring all three full seeds to beat never-write with paired
source-group bootstrap lower bounds above zero, all three to beat matched
no-support, and all safety/audit conditions to pass.

Only a successful nonconfirmatory Phase B may justify constructing a new
source-disjoint V5 formal corpus with newly frozen train/development/test
partitions.  Existing V4 sealed-test inputs remain untouched in every V5 phase.

## Resource protocol and audit

Run Phase A before choosing any larger scale.  Record seconds and peak CUDA
allocated/reserved memory for every policy-boundary transition.  Use four
source-group shards across the two RTX 5090 GPUs only if the smoke implementation
tests pass.

The independent auditor must reload all six checkpoints, reconstruct actions,
recompute J and safety metrics from stored query outcomes, verify feature hashes
and numeric-only online memory, verify layer composition and sequential state
continuity, recompute paired source-group bootstraps and the Phase-A gate, and
confirm that model parameters and all V4/V3 dependencies are unchanged.

All V5 behavior is opt-in.  Baseline entry points, configurations, checkpoints,
native writer source, V1--V4 artifacts, and sealed-test files must not be
overwritten.
