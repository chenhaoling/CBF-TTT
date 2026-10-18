# Corrected Base Stage-1 resource watcher evidence

This directory archives compact evidence for the corrected 16K/32K resource
watcher. It contains no source text, token IDs, checkpoints or predictions.

- Tool commit: `97c4d786e0c6c9d38de665312a92d41b26d63ab1`
- tmux: `baseline_qwen3_4b_base_stage1_ladder_97c4d786_20261019`
- watcher root:
  `/home/ctj/in_place_ttt_qwen3_4b_base_stage1_ladder_97c4d786_20261019.watch`
- future result root:
  `/home/ctj/in_place_ttt_qwen3_4b_base_stage1_ladder_97c4d786_20261019`
- initial state: `waiting_for_corrected_level_a`
- initial result-root existence: false
- remote focused suite: 81/81 passed

`launch.json` proves the exact clean commit, live watcher handle, live Level-A
handle, upstream states, absence of a ladder result root and idle
`hku-gpu2` GPUs. `static_inputs.sha256` binds the watcher, runner, auditor,
model config/identity and frozen plans. The watcher may create the result root
only after corrected Level A and its terminal independent audit pass.

A waiting watcher is not resource evidence. Completion requires four passing
children in the registered order and a passing
`independent_ladder_audit.json`.
