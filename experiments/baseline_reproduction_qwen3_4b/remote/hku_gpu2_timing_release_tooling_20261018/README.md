# hku-gpu2 timing-release tooling verification

This directory records CPU-only verification of commit
`2ac3a890d664ada87801b2b1d15232853b6e1557` on `hku-gpu2` on 2026-10-18.

The SSH gateway repeatedly reset while materializing a new approximately
600-MB worktree. The commit object had already been fetched, so the two new
files were extracted directly from that exact Git object with `git show` into
an isolated verification directory. Their hashes are recorded in
`file_hashes.txt`.

- The four new timing-release safety tests passed 4/4.
- Direct CLI loading and `--help` passed.
- The preceding exact clean commit passed the other 59 focused baseline tests;
  this additive change therefore brings the focused count to 63.
- No GPU process was launched. The corpus builder and repaired activation
  watcher remained live; GPU 0/1 used 46/18 MiB at 0% utilization.
