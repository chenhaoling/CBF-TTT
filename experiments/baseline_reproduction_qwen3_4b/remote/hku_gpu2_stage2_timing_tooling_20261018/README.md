# hku-gpu2 Stage-2 timing tooling verification

- Verification time: 2026-10-18 11:17--11:19 HKT.
- Host: `user-Super-Server`, reached only through SSH alias `hku-gpu2`.
- Tool commit: `69ee323af5e942ebe7df65d8f7445556d894b4d9`.
- Checkout: clean detached `/home/ctj/cbf_ttt_tool_69ee323a`.
- CPU-side focused baseline suite: 51/51 passed.
- Stage-2 timing Bash syntax and auditor CLI import: passed.
- Negative preflight: exit 1 before CUDA, with neither final nor partial output.
- GPUs after verification: GPU 0 at 46 MiB/0%, GPU 1 at 18 MiB/0%.
- Formal corpus builder remained live; the three existing upstream watchers
  remained alive and waiting. No Stage-2 timing GPU process was launched.

`focused_tests.txt`, `auditor_help.txt`, `file_hashes.txt`,
`negative_preflight_result.txt`, and `status_snapshot.txt` are the retained
machine evidence. The negative runner log is empty because the shell `test`
guard that detected the missing unpublished prerequisite exits silently.
