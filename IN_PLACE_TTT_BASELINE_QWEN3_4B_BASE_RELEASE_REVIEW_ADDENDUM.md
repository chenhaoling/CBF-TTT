# Qwen3-4B-Base 100M formal-release review addendum

Status: frozen before either corrected timing result, selected-stream result,
formal release, or formal 100M training outcome exists.

## Purpose

Formal Stage 1 and Stage 2 already require a separately written human release.
This addendum prepares the concrete material for that decision without
accepting it.  A CPU-only watcher combines the measured timing audit and the
corrected selected-stream audit into an immutable review packet.  It never
writes the release path, never supplies `--accept`, and never launches a
training process.

## Inputs and order

Each stage-specific watcher runs only on `hku-gpu2` and hides all GPUs.  It
waits for both of these terminal prerequisites:

1. the applicable corrected Base timing watcher has completed and its
   underlying and wrapper audits pass;
2. the corrected Base selected-stream watcher has completed and its
   independent audit passes.

The review builder calls the same validation functions used by the applicable
formal release creator.  It binds:

- underlying and corrected timing audit paths, protocols and SHA-256 values;
- measured per-arm timing and sequential formal-stage projection;
- exact Qwen3-4B-Base revision;
- corrected selected-stream audit path, protocol and SHA-256;
- corrected corpus-manifest SHA-256;
- the applicable stage's ordered record-index and training-token-hash
  digests;
- the review builder, release creator and this addendum SHA-256 values;
- the intended future release path.

## Output contract

The immutable packet has
`in_place_ttt_qwen3_4b_base_100m_release_review_v1`, the applicable stage,
`ready_for_human_review=true`, and `formal_release_created=false`.  The
watcher independently reloads the packet and verifies those fields and all
input hashes before reporting completion.

The packet is evidence for a later decision.  It is not authorization.  A
human must still inspect the measured projection and run the stage-specific
release creator with explicit `--accept`, nonempty `--reviewed-by`, and a
nonempty `--decision-note`.  Existing formal watchers remain responsible for
validating that separately written release before any GPU child starts.

## Failure and stop rules

The watcher stops without a packet if the host differs, either prerequisite
fails or loses its handle before a terminal result, an audit/hash is stale,
the target release already exists, or an output path would be overwritten.
It also stops if any GPU becomes visible to the review builder.  It does not
infer acceptance from a passing timing threshold, elapsed time, or an earlier
conversation instruction.
