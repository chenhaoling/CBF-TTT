# Public-corpus memory-monitor process-identity correction addendum

Status: **frozen before corrected persistent monitoring**

Freeze date: 2026-10-19 (Asia/Shanghai)

Parent: `IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_MEMORY_MONITOR_ADDENDUM.md`

## Observed failure

The first identity-correct persistent attempt from tool commit `32aa45e0...`
failed closed before writing a sample. The original classifier found two
processes with the exact builder filename and `--output-root`: the
`/usr/bin/time` parent wrapper and its Python builder child. It wrote
`FAILED_MULTIPLE_BUILDERS` with PIDs 1673441 and 1673442. The corpus builder
remained alive and unchanged. The failed output root must be preserved.

## Frozen correction

Retain every original host, source, output-root, polling, read-only and
termination rule. Add one process-identity condition: the matching process's
`/proc/<pid>/exe` basename must begin with `python`. This selects the actual
interpreter executing the registered builder and excludes `/usr/bin/time`
without using PID order, parent heuristics or a hard-coded PID.

The corrected run must use a new output root and a new clean committed tool
checkout. Tests must include both a Python child and a time wrapper carrying
the same command arguments, and must select only the Python child.

This correction changes no corpus data, builder, wrapper, sampling field,
threshold, claim or downstream release condition.
