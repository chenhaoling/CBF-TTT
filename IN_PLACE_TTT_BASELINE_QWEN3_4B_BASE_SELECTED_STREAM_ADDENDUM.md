# Qwen3-4B-Base 100M Selected-Stream Audit Addendum

Status: frozen before the corrected corpus finishes and before either formal
100M release is created.

## Purpose

The public-data reproduction must demonstrate that the no-TTT and native-TTT
arms consume the same deterministic records in the same order.  The earlier
selected-stream result used a superseded corpus and iterated the Hugging Face
stream directly.  It is therefore non-authoritative for the corrected
Qwen3-4B-Base run.

This addendum freezes a replacement audit that replays the installed training
runtime's iterable-dataset and `StatefulDataLoader` path on `hku-gpu2` before a
human can release either formal 100M stage.

## Frozen inputs

- Host: `hku-gpu2`, whose expected hostname is `user-Super-Server`.
- GPU use: none; the audit runs with `CUDA_VISIBLE_DEVICES=`.
- Corrected corpus profile: `full`, independently audited, with 724,800
  records and 34,999,369,728 training tokens.
- Model/tokenizer identity: `Qwen/Qwen3-4B-Base` at revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Seed: 42.
- Hugging Face iterable shuffle buffer: 10,000 records, as fixed by the
  installed VeOmni `build_iterable_dataset` implementation.
- Data-parallel world size: 2.
- Per-rank dataloader batch: 32 records (`global_batch_size=64 / 2 ranks`).
- Native dataloader settings: `num_workers=2`, `prefetch_factor=2`,
  `drop_last=true`.  Pinning is disabled only in the CPU replay because it
  cannot alter iterable record order.
- Each stage is one JSONL shard.  The installed Hugging Face runtime therefore
  accepts the requested two workers and stops one worker at iteration time.
  The artifact must record both `num_workers=2` and
  `effective_num_workers=1`; this mirrors the formal training input layout.
- Stage 1: 32,768 tokens per record, 27 steps, 864 records per rank,
  1,728 records and 56,623,104 tokens globally.
- Stage 2: 131,072 tokens per record, 5 steps, 160 records per rank,
  320 records and 41,943,040 tokens globally.

The replay reads only the first 15,000 provenance rows from each stage.  This
is larger than the 10,000-record shuffle reservoir plus the maximum selected
record count.  The compact replay file contains only stage index, source name,
and training-token SHA-256.  It never copies plaintext or token arrays.

## Required checks

1. Both no-TTT and TTT YAML files within a stage have identical data and
   dataloader fields, differing only in model/update behavior and output path.
2. The replay uses the installed `datasets` shuffle and node split followed by
   the installed `torchdata.stateful_dataloader.StatefulDataLoader` with the
   frozen worker settings.
3. Two independent executions produce exactly the same ordered stage indices,
   token hashes, and sources for both ranks.
4. Rank schedules have the exact frozen lengths, are mutually disjoint, and
   contain unique stage indices and training-token hashes.
5. The corrected corpus manifest, independent audit, provenance prefixes,
   model identity, four formal configs, and installed runtime source files are
   SHA-256 bound into the result.
6. A separate audit process rebuilds both compact prefixes, replays both
   schedules, and matches every ordered digest and aggregate count.
7. Neither the replay nor the independent audit may expose corpus text or
   token IDs in its JSON artifacts.

## Release rule

The Stage-1 and Stage-2 release creators must reject a release unless supplied
with the passing corrected selected-stream audit.  A release binds its path,
SHA-256, protocol, corrected-corpus manifest SHA-256, and the applicable
stage's ordered schedule digests.  Passing this audit does not itself release
training: the measured timing result still requires an explicit human
`--accept`, reviewer identity, and decision note.

## Stop conditions

Stop and do not create a formal release if any of the following occurs:

- the corrected corpus or its independent audit fails;
- the expected host, model revision, corpus counts, config values, or runtime
  path differs;
- either replay is nondeterministic, truncated, overlapping across ranks, or
  disagrees with the independent audit;
- any selected-stream artifact contains plaintext or token arrays;
- the corpus-builder tmux handle disappears before a valid complete corpus is
  published.

All outputs are immutable: the watcher, replay, auditor, and release tools
refuse to overwrite existing roots or JSON results.
