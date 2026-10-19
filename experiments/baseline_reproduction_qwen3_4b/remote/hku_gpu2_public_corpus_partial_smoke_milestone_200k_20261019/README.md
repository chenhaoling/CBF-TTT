# Qwen3-4B-Base public-corpus 200K milestone smoke

This directory is the immutable local copy of the CUDA-hidden, read-only smoke
run performed on `hku-gpu2` against the live partial full public corpus. The
remote source directory was:

`/home/ctj/in_place_ttt_qwen3_4b_base_partial_smoke_milestone_200k_38d167c3_20261019`

The audit checkout was clean at
`38d167c34641e498aef20c29b677ddfc47c589f7`; the live builder checkout was
clean at `3e09b18727ae6fcda17684da44abf53c59754465`. The observed builder PID was
`1673442` and remained live after the audit.

`result.json` records five passing checks and no errors. It re-tokenized the
first 16 complete records to 32,767 content tokens, validated hashes and
FineWeb/LongCrawl alternation, and checked 32 consecutive recent provenance
records ending at index 203,304. No source text, token array, activation or
model tensor was copied into this archive.

The script reported 3.31 seconds wall time and 650.45 MiB peak RSS; GNU time
reported 3.80 seconds and 744,448 KiB maximum RSS, with zero swaps. This is
only a partial-corpus consistency smoke. It does not replace atomic corpus
publication, complete independent audit, training, checkpoint conversion or
RULER evaluation.

## Copied artifact SHA256

| File | SHA256 |
|---|---|
| `exit_code.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `result.json` | `f825e0148e615d6aceb1d8f37602a8fbfa321a437478d0a9647b6d2e7fc98842` |
| `run.time.txt` | `578ccbd7926cd3dab7eda24d56f6ce960f95ddaa97c70a860ed5b2fd76a23d83` |
| `stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `stdout.log` | `03ba9538d2e500581fe05b5e7d5c333ea48a901925605425556ef1739bb6dd50` |
