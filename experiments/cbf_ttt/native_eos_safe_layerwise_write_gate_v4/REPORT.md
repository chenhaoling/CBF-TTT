# Native EOS-Safe Layerwise Fast-Weight Admission V4

## Status

The protocol was frozen in
`NATIVE_EOS_SAFE_LAYERWISE_WRITE_GATE_V4_PLAN.md` before formal outcome
scoring.  Train label collection is active on the audited 384-group split.
Development and sealed test remain unopened.

## Registered method

The Qwen3-4B backbone and native `ttt_conv/ttt_proj` writer remain unchanged.
At each boundary, the action is reject-all or admission of exactly one of the
seven registered native fast-weight layer updates.  Decision inputs contain
only current loss, prediction uncertainty, fast-weight geometry, and numeric
activation-support statistics.  Features and future-derived labels are stored
separately and bound by SHA256.

The controller predicts seven benefits, seven protected-NLL changes, seven
protected-accuracy harm probabilities, and seven EOS-harm probabilities.  It
admits the eligible layer with the largest predicted benefit, otherwise it
rejects the update.  Full, no-support, and support-only variants use three
fixed seeds.

## Release and audit controls

- Development requires a passed train summary and independent audit.
- Controller training requires independently audited train and development
  labels.
- Sealed test requires all three full-model development seeds to pass every
  registered efficacy, bootstrap, gap-closure, safety, and ablation condition.
- First test access creates an atomic one-shot release marker bound to the
  controller and result-directory hashes.
- A separate program reloads checkpoints and recomputes test predictions,
  actions, metrics, paired bootstraps, baselines, release decisions, and hashes.

## Current resource observation

Four collector workers run across two RTX 5090 GPUs.  Early labels take about
17.4--19.3 seconds each under two-worker GPU contention.  Per-worker peak CUDA
allocation is about 10.76 GiB and reservation about 11.26--11.35 GiB.  Final
resource statistics will be taken from the complete independently audited
sidecars.

## Pending decision

After all 1,152 train boundary labels complete, the runner must produce and
independently audit `train_summary.json`.  Only a passed result authorizes the
288 development labels.  If any development release condition fails, V4 stops
and this report will record the failure without opening test.
