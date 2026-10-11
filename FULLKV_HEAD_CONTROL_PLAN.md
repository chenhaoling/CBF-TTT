# Full-KV Retrieval Head Positive Control

## Question

The explicit retrieval-head pilot failed both the compressed-memory gate and its untrained Full-KV check. This control directly trains the identical head on frozen Full-KV query states. It separates an unusable head from a writer bottleneck before any larger memory experiment.

## Fixed protocol

- Arms: `same_lr` (`1e-4`) and `high_lr` (`1e-3`).
- Same head initialization, balanced paired assignment schedule, 32 train / 8 development worlds, four cycles, 512 updates, and 4,096 question exposures.
- Backbone, writer, token embeddings, and cached Full-KV features are frozen. Test stays sealed.
- Features are the final query hidden state after the full 4,096-token context. Candidate vectors are the same frozen mean embeddings of five digit tokens used by the failed pilot.
- Evaluation uses assignment shifts 0 and 2 on the first eight train worlds and all eight development worlds, with context4 and global16 ranking.

The primary `high_lr` arm passes only if last-32 training accuracy is at least 80%, development context4 top 1 is at least 75%, development global16 top 1 is at least 50%, and the group-bootstrap lower confidence bound for development global16 top 1 exceeds the 6.25% chance rate.

If either arm passes, the head is usable and a writer-path rerun with the successful head settings is justified. If both fail, the head representation is rejected; subsequent work moves to candidate-conditioned autoregressive scoring or structured key-value memory.

## Command

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_fullkv_head_control_v1 \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_fullkv_head_control.sh
```

This is an opt-in diagnostic and changes no baseline path.
