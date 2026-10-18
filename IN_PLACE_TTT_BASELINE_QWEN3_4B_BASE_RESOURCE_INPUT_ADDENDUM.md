# Corrected Qwen3-4B-Base resource-input addendum

Status: **frozen before corrected-Base resource-input construction**  
Freeze date: 2026-10-19 (Asia/Shanghai)

## Purpose

The earlier sparse resource inputs were bound to a pilot packed with the
post-trained Qwen3-4B tokenizer. They cannot be used by the corrected Base
chain. This addendum registers a new opt-in invocation of the existing sparse
input builder and auditor after the full Base-tokenizer corpus is complete.

## Source gate

The source root must be the atomically published `full` corpus with exactly
724,800 records and 34,999,369,728 training tokens. Its canonical independent
audit must pass every check and bind the exact source manifest SHA256 supplied
to both builder and auditor. A `pilot` corpus, `.partial` root, failed audit or
caller-supplied mismatched hash stops construction.

The tokenizer must be the exact identity-audited `Qwen/Qwen3-4B-Base`
revision `906bfd4b4dc7f14ee4320094d8b41684abff8539` with EOS 151643.

## Outputs

Reuse the existing deterministic derivation: take the first ordered
FineWeb-Edu/LongCrawl64 pair from Stage 1 for 8K, 16K and 32K, and from Stage 2
for 64K and 128K. Each output has two records and exactly twice its sequence
length in training tokens. Full-length 32K/128K rows remain byte-identical to
their parents; shorter rows use the stable tokenize/decode prefix rule.

The manifest stores hashes, sizes, source labels, parent indices and token
counts but no text or token IDs. The independent auditor re-derives every
record and verifies source order, parent artifact hashes, tokenizer assets,
stream hashes and absence of plaintext metadata. Passing input construction
does not prove GPU feasibility or release a training stage.

## Compatibility

The builder and auditor retain their legacy default pilot hash/profile. The
corrected chain must explicitly pass `--required-source-profile full` and the
observed, independently audited full-manifest SHA256. Existing legacy outputs
and baseline entry points are not overwritten.
