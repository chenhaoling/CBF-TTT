# Active baseline-chain integrity audit on `hku-gpu2`

This CPU-only audit captures the immutable execution chain while the full
public corpus is still building. It verifies the clean upstream checkout,
eleven frozen tool checkouts, thirteen live watcher processes, the exact
watcher script in every process command line, the two expected RTX 5090
devices, and continued absence of all three explicit release files.

The audit ran on `user-Super-Server` with an empty `CUDA_VISIBLE_DEVICES`.
It did not load a model, reserve a GPU, alter a process, or release an
experiment. All 55 registered checks passed.

## Files

- `audit_in_place_ttt_active_chain_integrity.py`: frozen read-only auditor,
  SHA256 `2f8fdc682ebf874e0af25e61ef0e0987e727c329202ef2c9c157ea995db3f831`.
- `in_place_ttt_active_chain_integrity.json`: structured result, SHA256
  `6962c87fa39cbed91523bc551d8b7aa1e9da4602e8f5ae1b7178424b0b8e6315`.

This snapshot proves integrity at `2026-10-18T20:43:23Z`. Later training and
evaluation artifacts still require their own frozen child and aggregate
audits; a live watcher is not a completed result.
