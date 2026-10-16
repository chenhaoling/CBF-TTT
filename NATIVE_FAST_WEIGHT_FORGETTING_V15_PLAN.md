# V15 native fast-weight forgetting correction and decision gate

## Corrected research question

The paper studies forgetting of the actual session-local dense fast weights in
In-Place TTT. For every configured TTT layer, the only admissible mainline
state update is

```text
M_t = (1 - alpha_t) * M_{t-1} + DeltaW_t,
DeltaW_t = native ttt_proj/ttt_conv candidate,
g_t = 1.
```

Here `alpha_t` is the forgetting fraction: zero retains all old session memory
and one clears all old session memory. The frozen backbone weight remains
unchanged. Attention KV follows the normal causal path. The controller cannot
select a text record, construct an external value residual, gate the current
write, or access a future query.

## Evidence correction

The utility record controller in V6--V13 uses a list of explicit records with
per-record strengths and an external value residual at readout. Those results
can be cited only as evidence that selective deletion can be useful when memory
is addressable and readable. They do not validate native dense fast-weight
forgetting. V14 REBEL was withdrawn before source acquisition because it would
replicate the same wrong substrate.

The native evidence is kept separate:

- `CBFSession` stages the real `ttt_proj/ttt_conv` update and applies one shared
  forgetting coefficient to all configured dense layer memories.
- Native accumulation previously showed that repeated writes can damage NLL
  and create an EOS bias.
- Correct/Wrong/Zero intervention previously found no reliable factual content
  binding in one native update.
- The natural joint pilot already contains the exact `g=1` slice required for
  an initial forgetting-only audit, so V15 first reanalyzes those frozen traces
  without another model run.

## Stage A: frozen-trace headroom gate

Use all 48 already observed natural scenarios from 12 source groups. For each
state, retain only joint-grid actions whose current write gate is exactly one.
Convert runtime retention `r` to method forgetting `alpha=1-r`, yielding the
three-point grid `{0, 0.5, 1}`. Compare:

1. each fixed forgetting coefficient;
2. a rowwise counterfactual oracle over the same three points;
3. the best `g=1` action against the best `g=0` action as a writer-utility
   diagnostic.

The controller is authorized only if all preregistered checks pass:

- at least 25% of rows have more than 0.005 NLL spread across forgetting;
- oracle headroom over the strongest fixed coefficient is at least 0.005 NLL
  and the source-group bootstrap 95% lower bound is positive;
- full forgetting is selected in no more than 75% of oracle rows;
- at least two source groups require a non-full-forgetting action;
- the native current write has positive mean gain over the best no-write
  action and its source-group bootstrap 95% lower bound is positive.

These checks separate a useful state-dependent forgetting policy from a
controller that merely suppresses a damaging writer.

## Stage A result and stop decision

Stage A failed. Full details are archived in
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/native_forgetting_v15_reanalysis`.
The strongest fixed action is full forgetting (`alpha=1`), selected by the
oracle in 39/48 rows. Oracle headroom is only 0.00110 NLL. The best native
`g=1` action is worse than the best `g=0` action by 0.04045 NLL on average,
with source-group bootstrap 95% interval `[-0.05825, -0.02460]` for the
write gain. No native forgetting controller is trained from these labels.

## Next authorized experiment

Repair and validate the native writer before returning to controller training.
Freeze the Qwen backbone and train only the existing seven layers'
`ttt_proj/ttt_conv` modules. Use source-disjoint natural continuation examples
and optimize the same next-token objective used at deployment, plus a bounded
contrastive term that requires the correct update to beat a source-matched
wrong update. Run a held-out causal intervention with identical query and KV:
Correct versus Wrong, norm-matched Random, and Zero fast weights.

The writer passes only if Correct improves continuation NLL over all three
controls with a positive source-group bootstrap lower bound, repeated writes
remain numerically stable, and at least one retain-sensitive condition shows a
benefit from old memory. Only then construct fresh `{0,0.5,1}` native forgetting
labels, train the continuous controller, and compare it with full forgetting,
no forgetting, fixed 0.5, periodic reset, norm threshold, and oracle.

## Compatibility

All changes are opt-in. Baseline training and inference are unchanged. New
generic CBF labels and checkpoints carry
`memory_substrate=native_dense_ttt_fast_weight`; controller fitting rejects
labels from a different or unspecified substrate. Historical checkpoints can
still be loaded for inference when they predate this metadata, but they cannot
be silently retrained as native evidence.
