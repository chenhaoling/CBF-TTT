# Run status

- State: completed
- Fixed interpretation: `no_localized_binding`
- Execution audit: passed
- Test split: sealed and unscored
- Server: `hku-gpu2`, two RTX 5090 GPUs
- Runtime: 2026-10-11 22:43:19 to 22:50:28 +08:00, about 7 minutes 10 seconds
- Source commit: `887c3d0b8daf141a79892121decec65ba2114e7b`
- Remote root: `/home/ctj/cbf_ttt_candidate_ranking_v1`
- Remote tmux: `cbf-candidate-ranking-v1` exited normally
- GPU state after completion: idle

Both arms scored the fixed 0/2,048/4,096-exposure checkpoints without retraining or selecting a checkpoint. At the fixed final `joint_binding` checkpoint, Correct-memory `global16` top-1 was 0.0625, exactly the 1/16 chance level, and did not exceed twin or Wrong memory. The current writer therefore contains no usable localized entity-to-value binding under this diagnostic. No forgetting labels or controller were produced.

