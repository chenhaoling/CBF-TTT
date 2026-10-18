# Corrected Qwen3-4B-Base Level-A watcher evidence

This directory contains compact, text-free evidence for the corrected Base
resource-input smoke and the waiting Level-A execution chain. Raw corpus rows,
checkpoints and future RULER predictions remain on `hku-gpu2`.

## CPU validation

Commit `6a3bb2bf54c803d4bc9a9fb789e5f11c7f1a6b65` passed all 78 focused
In-Place-TTT tests in a clean detached worktree. The new opt-in resource-input
path was then exercised with the audited Base pilot and an explicitly supplied
manifest SHA/profile. Build and independent audit exited zero; all seven audit
checks pass. The manifest and audit hashes are recorded in `launch.json`.

The pilot invocation exists only to test the generalized code path. Formal
corrected inputs will be rebuilt from the `full` Base corpus after its
canonical audit passes.

## Waiting chain

- tmux: `baseline_qwen3_4b_base_level_a_6a3bb2bf_20261019`
- output root:
  `/home/ctj/in_place_ttt_qwen3_4b_base_level_a_6a3bb2bf_20261019`
- initial state: `WAITING_BASE_CORPUS`
- prerequisite corpus state: `RUNNING_BUILD`
- training children at launch: zero
- GPU restriction: `hku-gpu2_only`

`launch.json` verifies the exact clean tool commit, watcher/corpus process
handles, live tmux sessions, zero training children and idle GPU snapshot.
`watcher_static_inputs.sha256` binds all frozen plans and scripts. A live
watcher is not a completed result: Level A remains incomplete until both
resource arms, conversion, two RULER smokes and the terminal independent audit
all pass.
