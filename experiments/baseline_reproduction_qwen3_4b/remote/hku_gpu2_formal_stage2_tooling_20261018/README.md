# hku-gpu2 formal Stage-2 tooling verification

- Verification time: 2026-10-18 11:26 HKT.
- Host: `user-Super-Server`, reached only through SSH alias `hku-gpu2`.
- Tool commit: `d2bc98e8a34d8e66c43117f6cf0dcedd298d911d`.
- Checkout: clean detached `/home/ctj/cbf_ttt_tool_d2bc98e8`.
- CPU-side focused baseline suite: 53/53 passed.
- Formal Stage-2 Bash syntax and auditor CLI import: passed.
- Negative preflight: exit 1 before CUDA, with neither final nor partial output.
- GPUs after verification: GPU 0 at 46 MiB/0%, GPU 1 at 18 MiB/0%.
- Formal corpus builder remained live and reached a 51-GB partial directory;
  the activation, Stage-1 ladder and Stage-1 timing watchers remained waiting.
- No formal Stage-2 release was created and no GPU process was launched.

The retained files contain the complete test output, dependency hashes,
auditor CLI, negative-preflight result and remote process/resource snapshot.
