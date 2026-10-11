# Run status

- State: completed
- Fixed pilot gate: failed
- Execution audit: passed
- Test split: sealed and unscored
- Server: `hku-gpu2`, two RTX 5090 GPUs
- Runtime: 2026-10-11 23:17:49 to 23:38:50 +08:00, about 21 minutes
- Source commit: `f5b35eb`
- Remote root: `/home/ctj/cbf_ttt_associative_binding_v1`
- Remote tmux: `cbf-associative-binding-v1` exited normally
- GPU state after completion: idle

Both arms completed 256 optimizer updates and 1,024 question exposures. The all-fact permutation arm created a significant Correct-versus-permuted digit-NLL difference at the midpoint, but the effect disappeared at the fixed endpoint and never changed candidate top-1. The endpoint development `global16` Correct top-1 was 0.0469 and complete-code accuracy was zero. No formal generation run, forgetting labels, or controller training was started.

