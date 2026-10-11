# Norm-Constrained Candidate-Aligned Binding

## Motivation

The unbounded listwise run stopped safely near update 41 because retrieval-only fast memory exceeded `||delta W||/||W|| = 1`. Its four-way retrieval loss had not improved. The failure directory is preserved and is not a formal result.

This recovery keeps the registered listwise comparison and adds one architectural constraint: every layer's context-generated fast-weight delta is scaled to at most 0.3 times the frozen down-projection norm before any query read. The scale is detached; gradients still train the delta direction. Both arms use the identical constraint during training and evaluation.

## Fixed protocol

- Arms: `retrieval_only` and `retrieval_plus_lm` (`retrieval + 0.25 * balanced LM`).
- Data: first 32 train and first 8 development worlds; test sealed.
- Four rounds, 1,024 question exposures, 256 updates, fixed 0/512/1,024 checkpoints.
- One context write and four four-candidate listwise reads per update; both arms compute the same auxiliary LM reads.
- Optimizer, order, initialization, learning rate `1e-7`, gradient clip 1.0, and all evaluation controls remain matched.
- A one-update GPU smoke must remain below 28 GiB and produce finite gradients before formal execution.

The development and optimization gates remain exactly those in `LISTWISE_BINDING_PLAN.md`. No checkpoint selection or test scoring is allowed.

## Command

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_norm_listwise_binding_v1 \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_norm_listwise_binding.sh
```

