# Preliminary 100M corrected Qwen3-4B-Base Level-A addendum

Status: **frozen before preliminary-corpus Level-A GPU execution**  
Freeze date: 2026-10-19 (Asia/Shanghai)

## Scope

This addendum permits the existing corrected Qwen3-4B-Base Level-A closure to
consume an independently audited `preliminary_100m` corpus rather than waiting
for the optional approximately 35B-token `full` profile. All GPU work remains
restricted to the two RTX 5090 devices on `hku-gpu2`.

The source corpus must contain exactly 2,048 records and 98,566,144 training
tokens with zero synthetic EOS tokens: 1,728 records at 32K followed by 320
records at 128K. Its manifest, independent audit, exact Qwen3 tokenizer and
plan hashes must pass. No symlink, relabeling or fabricated full-profile
manifest is permitted.

## Code-path change

The resource-input builder/auditor and corrected Level-A watcher/auditor accept
an explicit `REQUIRED_SOURCE_PROFILE=preliminary_100m`. Their default remains
`full`, preserving the old formal path. The watcher also accepts an explicit
`CORPUS_SESSION`; its default remains the registered full-builder tmux name.

The Level-A workload is unchanged: derive the same FineWeb/LongCrawl pair at
8K, execute no-TTT before native-TTT with activation offload and a real AdamW
step, enforce the registered GPU/host/swap gates, save DCP checkpoints, convert
through the official path, reload, run one example for each of the 13 RULER-4K
tasks, and independently audit every binding. It does not train a gate.

## Stop and claim rules

Stop on profile/count/hash drift, a non-finite loss or gradient, missing
optimizer step, conversion/reload mismatch, RULER smoke failure, less than 5%
free GPU memory, less than 8 GiB host available memory, more than 3.5 GiB swap,
or any concurrently active corpus builder/auditor. A passing result only proves
the preliminary corpus can close the corrected Base Level-A engineering path;
it does not establish efficacy and does not release natural-text gate training
or its sealed test.
