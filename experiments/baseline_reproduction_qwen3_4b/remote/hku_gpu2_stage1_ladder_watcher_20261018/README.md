# hku-gpu2 Stage-1 ladder watcher evidence

This directory records the initial deployment snapshot for the CPU-only
handoff watcher pinned to tool commit
`bae87e6c75183a73f5ed2cd8150e0cc228ed19f0` on `hku-gpu2`.

- tmux session:
  `baseline_2x5090_stage1_ladder_bae87e6c_20261018`
- watcher sidecar:
  `/home/ctj/in_place_ttt_2x5090_stage1_ladder_bae87e6c_20261018.watch`
- future ladder output:
  `/home/ctj/in_place_ttt_2x5090_stage1_ladder_bae87e6c_20261018`
- 8K prerequisite root:
  `/home/ctj/in_place_ttt_2x5090_activation_8k_f14cb715_20261018`

At `2026-10-18T10:49:49+08:00`, the watcher state was
`waiting_for_audited_8k`; the 8K watcher tmux pane was live; the public-corpus
builder PID 1397987 was live; the ladder output did not exist; and GPU 0/1
used 46/18 MiB at 0% utilization. Therefore this deployment did not launch a
GPU process.

`frozen_inputs.sha256` is the watcher's own dependency ledger. The copied
state and handle files are point-in-time observations. `remote_snapshot.txt`
contains the clean tool revision, live process evidence, corpus size, watcher
states and GPU snapshot collected in one SSH command.
