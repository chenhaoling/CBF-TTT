# hku-gpu2 RULER tooling verification

This directory records CPU-side verification of commit
`52775cb307b84de89751891f2ee99251be5ceb6f` on `hku-gpu2`
(`user-Super-Server`) on 2026-10-18.

- The complete focused In-Place-TTT baseline suite passed 59/59.
- All three RULER Bash runners passed `bash -n`.
- All four RULER auditor CLIs loaded and printed `--help` successfully.
- The resource ladder and formal matrix negative preflights both exited 1
  because the paired formal Stage-2 root does not exist yet.
- Both negative preflights stopped before creating final or partial output,
  and no CUDA process was launched.
- At the final snapshot, GPU 0/1 used 46/18 MiB at 0% utilization. The public
  corpus builder remained live and all three upstream watchers remained
  registered.

`bash_syntax.txt` and `git_status.txt` are intentionally empty: Bash emitted
no diagnostics and the fixed checkout was clean. The two
`*_negative_preflight.txt` files are intentionally empty because the failing
shell `test` command emitted no text; `negative_preflight_result.txt` records
the exit codes, missing prerequisite, output absence and CUDA status.
