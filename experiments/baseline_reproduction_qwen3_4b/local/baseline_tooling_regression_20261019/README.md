# Baseline tooling regression suite

This directory records a CPU-only regression pass across the registered
In-Place-TTT baseline reproduction chain after the exact Qwen3-4B FSDP2
audits were added.

## Local suite

From tracked-clean commit `6ccfdba12c8eb10e9caf22ddf929f730d1339a58`,
the command

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tests -p 'test_in_place_ttt*.py' -v
```

ran **133 tests successfully** in 9.219 seconds. One test was skipped because
the local macOS Python did not contain the frozen Torch/Transformers inference
environment. Coverage includes public-corpus atomicity and continuity,
resource-runner fail-closed behavior, corrected Base Level-A ordering,
32K/128K training and release gates, checkpoint bindings, RULER 4K--256K,
Table-1 reporting, and the invariant that update admission remains disabled.

## Frozen inference-environment test

The skipped runtime test was then run separately with CUDA hidden on
`hku-gpu2` in the frozen evaluation environment and a clean tool checkout.
It passed in 5.296 seconds. The test constructs a tiny released Qwen3 TTT MLP
and verifies that every inference delta is clipped to the paper's
`tau=1e-5`, zero deltas remain unchanged, the setting serializes through the
config, and nonpositive/nonfinite clipping values fail closed.

The remote checkout was commit `2b10fd17062687778bcb74b624368c7d5814eff4`.
`source_equivalence.txt` is empty by construction and proves that the runtime
test and overlay builder/auditor sources did not change between that commit
and the locally tested archival commit `6ccfdba1`.

## Evidence

- `local_unittest.log`: complete 133-test local output
- `local_identity.txt`: local commit, tracked status and Python identity
- `remote_runtime_unittest.log`: environment-specific runtime test output
- `remote_runtime.time.txt`: independent runtime resource record
- `remote_identity.txt`: remote host, clean source identities and hidden-CUDA
  condition
- `SHA256SUMS`: hashes created on `hku-gpu2` for the remote files
- `source_equivalence.txt`: expected-empty source diff between the two tested
  commits
- `ARCHIVE_SHA256.txt`: hashes of the complete archive

This suite validates tooling behavior and frozen gates. It does not substitute
for full-corpus construction, optimizer steps, checkpoint conversion or
RULER scores.
