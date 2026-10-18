# Qwen3-4B-Base live partial-corpus smoke addendum

Status: frozen before inspecting any live partial-corpus record.

Freeze date: 2026-10-19 (Asia/Shanghai)

## Purpose

The full Base-tokenized public corpus takes roughly one day to construct.  This
CPU-only smoke detects a malformed Stage-1 stream while the immutable builder
is still running; it neither replaces nor weakens the complete independent
audit after atomic publication.

## Frozen checks

Run only on `hku-gpu2` with all GPUs hidden.  Require the registered builder
PID to be live, its command line to name the exact builder and final output
root, the `.partial` root to exist, and the final root and smoke output to be
absent.

Using the exact Qwen3-4B-Base tokenizer, inspect:

1. the first eight FineWeb/LongCrawl record pairs (16 data and provenance
   rows), re-tokenizing every `content_split` and independently checking the
   32,767 content tokens, appended EOS, training-token SHA-256, source order,
   stage/source indices, span accounting and absence of plaintext in
   provenance; and
2. the latest 32 complete provenance rows, requiring consecutive global
   indices, alternating sources, matching source indices, valid token counts,
   span accounting, valid 64-character token hashes and no plaintext fields.

The smoke artifact may contain paths, file sizes, counts, hashes, indices and
resource measurements.  It must not serialize corpus text, token IDs, document
text or source identities.

## Interpretation and stop rules

A pass is early evidence that the active builder is emitting the registered
schema and Base-tokenized records.  It does not establish full record counts,
the complete ordered curriculum hash, cross-stage continuity, source
exhaustion behavior or final artifact hashes.

Stop the build only on a reproducible smoke failure tied to a stable completed
row.  A transient final-line write, process-observation race or SSH timeout is
not a failure and must be retried without restarting the builder.  Formal
training remains sealed until atomic publication and the complete independent
audit pass.
