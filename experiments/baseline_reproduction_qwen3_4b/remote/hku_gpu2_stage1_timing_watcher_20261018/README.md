# hku-gpu2 Stage-1 timing watcher evidence

This directory records the initial deployment of the CPU-only Stage-1 timing
handoff watcher pinned to commit
`56094b8906ad670ec2be0d21ab75d698216082d4` on `hku-gpu2`.

- tmux session:
  `baseline_2x5090_stage1_timing_56094b89_20261018`
- watcher sidecar:
  `/home/ctj/in_place_ttt_2x5090_stage1_timing_56094b89_20261018.watch`
- future timing output:
  `/home/ctj/in_place_ttt_2x5090_stage1_timing_56094b89_20261018`
- prerequisite ladder:
  `/home/ctj/in_place_ttt_2x5090_stage1_ladder_bae87e6c_20261018`

At `2026-10-18T10:59:40+08:00`, the timing watcher was
`waiting_for_audited_stage1_ladder`, its recorded ladder-watcher pane was live,
the future timing output was absent, and GPU0/GPU1 used 46/18 MiB at 0%.
The public-corpus builder and both upstream watchers were also observed live.

The separately recorded negative preflight used implementation commit
`18c639324b0600a3db02a087799a5c4ae89dda38`. It exited 1 because the formal
corpus and ladder were not published, created neither output nor partial
directory and left both GPUs unchanged. The empty `negative_preflight.log` is
expected because the failing `test -s` guard is silent; the authoritative exit
and absence checks are in `negative_preflight_result.txt`.
