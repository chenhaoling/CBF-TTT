# Run status

- State: completed
- Stage result: failed the preregistered content-memory gate
- Execution audit: passed
- Test split: sealed and unscored
- Server: `hku-gpu2`, two RTX 5090 GPUs
- Runtime: 2026-10-11 11:07:50 to 11:59:29 +08:00, about 51 minutes 39 seconds including construction, smoke, formal training, evaluation, summary, and audit
- Source commit: `91a0ef9`
- Remote root: `/home/ctj/cbf_ttt_dynamic_binding_v1`
- Remote tmux: `cbf-dynamic-binding-v1` exited normally
- GPU state after completion: idle

Both formal arms completed 1,024 optimizer updates and 4,096 question exposures. The final development code accuracy was zero for both arms. The binding objective improved development correct-memory digit NLL by 0.0965 relative to joint CE, but did not establish reliable correct-versus-wrong memory discrimination. No forgetting controller was trained.
