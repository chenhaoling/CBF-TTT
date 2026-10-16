# Native Closed-Loop Fast-Weight Gate V5 Phase B: frozen development plan

Status: **frozen before Phase-B model scoring**
Freeze date: 2026-10-16 (Asia/Shanghai)
Protocol: `native_closed_loop_fastweight_gate_v5_phase_b`

## Phase-A evidence and scope

The registered twelve-group Phase-A smoke completed with 288 native policy
transitions and passed independent reconstruction.  Two of three full models
beat never-write and their matched no-support policy by point estimate; the
mean full-model gain versus never-write was 0.018659.  Seed 2026101802 was
worse than never-write by 0.016884.  All registered safety and trajectory
checks passed.

Phase B determines whether the smoke signal is stable on the full 96-group V4
development split.  These groups were already used for V4 release decisions,
so Phase B is nonconfirmatory method evidence.  It cannot release the existing
sealed test or support a formal efficacy claim.

## Frozen data, policies, and recurrence

Use all 96 source groups in the independently audited
`development_model_inputs.jsonl`, with three sparse boundaries per group.  The
online scorer receives no semantic audit, mechanism, or plaintext identity.

Run the same eight frozen Phase-A policies with no retraining, calibration,
threshold change, checkpoint selection, or seed removal:

1. always-write;
2. never-write;
3. full seeds 2026101801, 2026101802, 2026101803;
4. matched no-support seeds 2026101801, 2026101802, 2026101803.

Every policy starts from the same setup fast weights within a source group and
then owns an independent native cache.  Its selected state at boundary `t`
must exactly equal its parent at boundary `t+1`.  The Qwen3-4B backbone,
`ttt_conv/ttt_proj` writer, writer learning rate, action space, V4 feature
contract, four-head controller rule, and fresh-KV queries remain frozen.

## Registered Phase-B gate

Phase B passes only if all conditions hold:

1. every full seed has lower mean J than closed-loop never-write;
2. for every full seed, the paired 96-source-group bootstrap 95% lower bound
   for gain versus never-write is strictly positive;
3. every full seed has lower mean J than its matched no-support policy;
4. every full seed loses at most 0.5 percentage point protected accuracy from
   never-write;
5. every full seed raises protected NLL by at most 0.01;
6. every full seed raises EOS rate by at most 1 percentage point;
7. all frozen-checkpoint action, feature/label separation, layer composition,
   sequential-state, backbone, dependency, resource, and plaintext-leakage
   audits pass.

Report paired bootstrap intervals versus both never-write and matched
no-support.  The no-support interval is descriptive; condition 3 uses the
preregistered point comparison.  Always-write remains a harm baseline and
cannot substitute for never-write.

## Stopping and claim boundary

If any condition fails, stop V5.  Do not tune the V4 models, choose a seed,
change thresholds, construct a new formal corpus, or open any sealed test.

If every condition passes, the only permitted conclusion is that the frozen
gate merits a newly registered, source-disjoint formal V5 corpus.  Before any
new corpus scoring, freeze its construction, train/development/test partition,
sample size, controller training rule, formal success thresholds, and one-shot
test release protocol.

## Resources and independent audit

Use four shards across two RTX 5090 GPUs, two workers per GPU.  Record seconds
and peak CUDA allocated/reserved memory for every policy-boundary transition.
Based on Phase A, 2,304 transitions are expected and the run is expected to
take roughly 100--120 minutes.

The independent auditor must reload all six checkpoints, reconstruct all 2,304
actions and outcome metrics, recompute both paired source-group bootstraps,
verify exact three-step cache continuity per policy/group, verify all artifact
and dependency hashes, and confirm `test_opened=false`.

All Phase-B behavior is opt-in.  V1--V4 artifacts, baseline entry points,
checkpoints, native writer source, and all sealed-test files remain immutable.
