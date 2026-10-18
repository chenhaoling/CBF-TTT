# Qwen3-4B-Base reviewed-release transition addendum

Status: frozen before any formal release or 100M training outcome exists.

## Purpose

The non-authorizing review packet fixes the exact timing and selected-stream
evidence a human sees.  This addendum defines the only supported transition
from that packet to an accepted formal release, so a later command cannot
silently substitute different audit files.

## Transition contract

`create_in_place_ttt_qwen3_4b_base_release_from_review.py` must:

1. require a passing immutable review packet with
   `ready_for_human_review=true` and `formal_release_created=false`;
2. require its output to equal the packet's registered target release path;
3. reload the packet's timing, corrected-timing and selected-stream paths,
   verify their current SHA-256 values, and rerun the applicable Stage-1 or
   Stage-2 release validation functions;
4. require the independently recomputed review dictionary to equal the packet
   byte-for-byte at the JSON-value level;
5. require explicit `--accept`, nonempty `--reviewed-by`, and nonempty
   `--decision-note` directly from the human release action;
6. preserve the existing release protocol and fields while additionally
   binding the review packet path, SHA-256 and this transition addendum;
7. atomically create a previously absent target and refuse every overwrite.

The tool is never called by a watcher.  Preparing or passing a review packet,
an elapsed timeout, or a prior general instruction does not supply
`--accept`.  Formal GPU execution remains restricted to `hku-gpu2`, while the
transition itself is CPU-only and must run with GPUs hidden.

## Stop rules

No release is written if the packet, any source audit, model revision,
selected-stream digest, release-tool source, target path, reviewer, decision
note, host or GPU policy differs from the frozen contract.  A failed attempt
does not authorize retry with weaker validation.
