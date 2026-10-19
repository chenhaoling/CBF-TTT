# Preliminary 100M public-corpus evidence

This directory archives compact, text-free evidence from the
`preliminary_100m` Qwen3-4B-Base corpus build on `hku-gpu2`. The packed
training JSONL and provenance rows remain on the server and are not committed.

## Result

- Remote corpus root:
  `/home/ctj/in_place_ttt_public_corpus_qwen3_4b_base_preliminary_100m_0a33dbd5_20261019`
- Wrapper state: `COMPLETE`
- Profile: `preliminary_100m`
- Records: 2,048
- Training tokens: 98,566,144
- Synthetic EOS tokens: 0
- Stage 1: 1,728 records and 56,623,104 tokens at 32K
- Stage 2: 320 records and 41,943,040 tokens at 128K
- Build wall time: 335.83 seconds
- Build peak host RSS: 1,156.17 MiB
- Independent-audit wall time: 160.14 seconds
- Independent-audit peak host RSS: 813,588 KiB
- Audit: all ten registered checks passed

The tokenizer emitted a maximum-length warning while preprocessing one source
document with 188,570 tokens. No model forward used that untruncated document;
the packer sliced it into registered sequence lengths, and the independent
record/token/hash audit passed.

## Full-corpus deferral

After this audit passed, the optional approximately 35B-token builder was
stopped under the frozen deferral procedure. Its 255,345 complete provenance
records and 36,818,954,678-byte partial root were atomically preserved at:

`/home/ctj/in_place_ttt_public_corpus_qwen3_4b_base_906bfd4b_20261019.deferred_user_priority_20261019_255345`

The original wrapper reports `FAILED_BUILD` because the child was deliberately
terminated after preservation. This is a user-priority deferral and is not a
published or resumable full corpus.

## Integrity

- Manifest SHA256:
  `522ec14bd1e05af172895c85250173edee5a1f6d8134cc0d34cf57a5aa5b548c`
- Independent audit SHA256:
  `9e6e92423eb095d92c972ab8f88d42d5d66937dbeab50190b3cb1ae20f11e0b2`
- Deferral record SHA256:
  `bdea3e318e8bc311c65c76d4ac3f5ab9a3180118405312d4cc2e94a2d400c226`

This evidence establishes corpus construction and audit only. It does not
establish model quality, a paper-scale reproduction or gate efficacy.
