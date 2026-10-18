# In-Place TTT 2xRTX5090 resource-input addendum

Status: **frozen before any 16K/32K activation-offload probe**

Freeze date: 2026-10-18 (Asia/Shanghai)

Parent 2x5090 execution addendum SHA256:
`0b70a098f8ee5cadf12f8d371dcc0020124069f5365aa5279d706cde4322e5c6`

## 1. Purpose

The sparse resource ladder must measure the declared sequence length. Passing
a 32K plaintext record with `--data.max_seq_len 16384` is insufficient evidence
that the iterable plaintext adapter actually consumed a 16K example. This
addendum registers exact, independently audited two-record inputs for 8K,
16K, 32K, 64K and 128K resource boundaries. It changes no training data for
the later 100M tier and makes no efficacy claim.

The already launched 8K waiting job remains bound to the previously audited
8K pilot with SHA256
`2eb03326ec87bc831ee1441e812bb5bef843f8e906656787cd2ddafc69d9d823`.
The new input suite is mandatory for every later 16K/32K/64K/128K probe and
may be used for a future fresh 8K confirmation.

## 2. Frozen source and ordering

The only source is the completed public-corpus pilot whose manifest SHA256 is
`3b2e7f4821692dfe75bae48e4cb52c6ac39f4b9b3b4c75f96aeb32ab7dec7206`.
Its independent streaming audit must pass and bind that manifest. For each
output, record 0 is the first FineWeb-Edu record and record 1 is the first
LongCrawl64 record:

| Output length | Parent artifact |
|---:|---|
| 8,192 | `stage1_32k.jsonl` |
| 16,384 | `stage1_32k.jsonl` |
| 32,768 | `stage1_32k.jsonl` |
| 65,536 | `stage2_128k.jsonl` |
| 131,072 | `stage2_128k.jsonl` |

Each output contains exactly two JSONL rows and only a `content_split` field.
No historical or source plaintext enters its manifest or audit.

## 3. Deterministic length derivation

For target length `L`, the content must re-tokenize to exactly `L-1` Qwen
tokens; VeOmni appends the final EOS to produce `L` training tokens. When the
parent content already has `L-1` tokens, preserve its text byte-for-byte.
Otherwise take the longest parent-token prefix whose decode then encode length
does not exceed `L-1`. Append the tokenizer EOS string only as needed to reach
exactly `L-1`. The builder records the consumed prefix count, synthetic EOS
count and token hashes. The independent auditor reimplements this procedure.

The ordered training-token hash is SHA256 over little-endian uint32 token IDs
for FineWeb followed by LongCrawl. Inputs are written under a `.partial`
directory and atomically published only after all five lengths validate.
Existing final or partial paths are never overwritten.

## 4. Acceptance gate

The independent audit must recompute and pass all of the following before an
input can enter a resource probe:

- source manifest/audit, parent artifact, plan, code and tokenizer hashes;
- the FineWeb/LongCrawl record order from parent provenance;
- deterministic prefix derivation and byte-for-byte preservation at 32K/128K;
- exactly two rows, `L-1` content tokens and `L` training tokens per row;
- every record hash, ordered stream hash and output artifact hash;
- absence of plaintext fields from manifest and audit.

Failure at any length invalidates the whole suite. Passing this gate proves
only input construction; GPU feasibility still requires the separately
frozen resource runner and independent resource audit.
