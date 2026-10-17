# In-Place TTT public-data construction pilot addendum

Date: 2026-10-17<br>
Parent plan: `IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`<br>
Scope: deterministic 32K data-pipeline validation only

## 1. Purpose

Level-B matched replication requires a public substitute with fixed source
revisions, exact Qwen token counts, document-boundary provenance, and an
identical ordered stream for no-TTT and native-TTT arms. The existing 1B
engineering corpus uses 6,144-token plaintext rows and records neither source
file hashes nor an ordered-token-stream hash. It remains immutable and is not
promoted to the baseline corpus.

The pinned VeOmni version does not expose a usable
`process_pretokenized_example`. Its official plaintext transform encodes each
`content_split`, appends one tokenizer EOS, and then splits at
`max_seq_len`. The new builder must therefore emit plaintext whose independent
Qwen re-encoding plus VeOmni's terminal EOS is exactly the requested length.

## 2. Frozen sources

- FineWeb-Edu repository: `HuggingFaceFW/fineweb-edu`
- FineWeb-Edu config: `sample-10BT`
- FineWeb-Edu revision:
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
- LongCrawl64 repository: `manifestai/longcrawl64`
- LongCrawl64 revision:
  `8e7af027525b3597852746ee11e4baa0e98492d7`
- LongCrawl64 pilot input: publisher file `train/0-of-256.parquet`
- Tokenizer: the unmodified local Qwen3-4B tokenizer used by Level A

FineWeb is read from the pinned Hugging Face streaming revision. LongCrawl is
read from the already downloaded publisher parquet; its byte size and SHA256
must be computed before packing. GPT-2 token-ID-only LongCrawl copies may be
decoded with the publisher's GPT-2 encoding, but their IDs must never be fed
directly to Qwen.

## 3. Pilot matrix

Build two independent source files:

| Source | Records | VeOmni training tokens/record | Total tokens |
|---|---:|---:|---:|
| FineWeb-Edu | 8 | 32,768 | 262,144 |
| LongCrawl64 | 8 | 32,768 | 262,144 |

No training is permitted in this addendum. The pilot only validates data
construction, resource use, and audit artifacts before a formal corpus is
defined on a larger-GPU host.

## 4. Packing and provenance rules

1. Traverse source rows in their deterministic pinned order.
2. Encode each nonempty source document with the original Qwen tokenizer and
   insert one Qwen EOS at every source-document boundary.
3. Decode candidate packed tokens to plaintext and independently re-encode
   them exactly as VeOmni does. If round-trip normalization shortens a row,
   add explicit Qwen EOS tokens inside the plaintext until the content has
   exactly 32,767 tokens; VeOmni adds the final 32,768th EOS.
4. Never duplicate source text to fill a row and never silently drop a partial
   token span. Any unconsumed raw tokens remain at the front of the buffer.
5. Save a provenance JSONL with source file/stream position, document SHA256,
   consumed token spans, synthetic-EOS count, and per-record training-token
   hash. It must not contain source plaintext.
6. Save a manifest binding the parent plan, builder code, source revision or
   parquet hashes, tokenizer assets, output/provenance hashes, ordered training
   token hash, record/token/document counts, wall time, and peak host RSS.
7. Write all artifacts through new `.partial` paths and atomically publish
   only after every invariant passes. Refuse to overwrite any existing final
   or partial output.

## 5. Success and stop conditions

The pilot passes only if both sources satisfy all of the following:

- exactly 8 records are emitted;
- every record independently re-encodes to 32,767 content tokens and becomes
  exactly 32,768 tokens after the terminal EOS;
- the manifest reports exactly 262,144 training tokens;
- all source revisions/file hashes and tokenizer hashes are nonempty;
- output, provenance, document-order and ordered-token hashes are present;
- provenance contains no `text` or `content_split` field;
- a second independent audit reconstructs all record hashes from the emitted
  plaintext and agrees with the manifest;
- the builder exits normally with finite wall time and recorded peak RSS.

Stop without publishing final outputs on a source/hash mismatch, incomplete
record count, token-length mismatch, round-trip instability, malformed source
row, existing output path, or network/data read failure.

## 6. Claim boundary

Passing this pilot validates a deterministic 32K public-data construction
path. It does not define the final 20B/15B source mixture, prove 128K packing,
start Level-B training, or reproduce the paper's private corpus. Formal scale
and stage composition remain gated on the selected H800-class execution host
and a separately frozen full-corpus manifest.
