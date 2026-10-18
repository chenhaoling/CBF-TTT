# Qwen3-4B-Base identity correction evidence

This directory archives compact evidence for the 2026-10-19 correction from
the post-trained `Qwen/Qwen3-4B` checkpoint to the paper-required
`Qwen/Qwen3-4B-Base` checkpoint. Large checkpoints, source data and packed
JSONL files remain on `hku-gpu2` and are not stored in Git.

## Decision

- Required model: `Qwen/Qwen3-4B-Base` at revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- The old checkpoint and all model-bound waiting chains are superseded for the
  paper-baseline claim. The old artifacts are preserved for diagnosis only.
- The Base and post-trained tokenizers share the same vocabulary and merges,
  and the registered ordinary-text and packed-edge probes agree. The
  post-trained tokenizer nevertheless has four additional reserved tokens
  (`<think>`, `</think>`, `<tool_response>`, and `</tool_response>`), whose
  literal strings occur in both old packed stages. Reusing the old payload
  could therefore change packing boundaries. The corpus is rebuilt from the
  source inventory with the Base tokenizer.
- Every GPU run is restricted to `hku-gpu2`. The full corpus build and audit
  are CPU-only and launch with an empty `CUDA_VISIBLE_DEVICES`.

## Files

- `base_identity.json`: exact model revision, asset hashes and nine passing
  identity checks.
- `tokenizer_compatibility.json`: vocabulary/merges equality and hash-only
  fixed/edge probes. It explicitly rejects reuse of old training hashes.
- `base_inventory.json` and `base_inventory_audit.json`: Base-bound source
  inventory and its seven-check independent audit.
- `base_pilot_manifest.json` and `base_pilot_audit.json`: Base-tokenizer
  two-stage pilot and its ten-check independent audit.
- `containment.json`: absence of all superseded tmux sessions, the active
  corrected CPU build, the `hku-gpu2_only` constraint and idle GPU snapshot.
- `full_build_frozen_inputs.sha256` and `full_build_wrapper.sha256`: immutable
  inputs and wrapper binding for the active complete-corpus build.

The live corrected full-corpus root is
`/home/ctj/in_place_ttt_public_corpus_qwen3_4b_base_906bfd4b_20261019`; its
run root is
`/home/ctj/in_place_ttt_public_corpus_qwen3_4b_base_run_20261019` and tmux
session is `public_corpus_qwen3_4b_base_full_20261019`. A live session is not a
completion result. Completion requires atomic publication and a passing
`independent_audit.json`.
