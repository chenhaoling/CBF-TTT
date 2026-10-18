# 100M selected-stream audit on `hku-gpu2`

This directory records a CPU-only replay of the exact data selection used by
the registered 98,566,144-token training tier. The audit uses seed 42, the
native Datasets buffer-10,000 streaming shuffle, data-parallel world size two,
and 32 examples per rank and optimizer step. It selects 27 Stage-1 steps and
five Stage-2 steps from the full-corpus provenance prefixes.

The process ran with an empty `CUDA_VISIBLE_DEVICES` on host
`user-Super-Server`; it did not load a model or reserve a GPU. It retained only
stage indices, source labels and training-token hashes. No source text or token
IDs are present in the artifacts.

## Result

All determinism, count, uniqueness, rank-disjointness and prefix-sufficiency
checks passed.

| Stage | FineWeb-Edu records | LongCrawl64 records | selected tokens |
|---|---:|---:|---:|
| 32K, 27 steps | 827 | 901 | 56,623,104 |
| 128K, 5 steps | 161 | 159 | 41,943,040 |
| Combined | 988 | 1,060 | 98,566,144 |

Because record lengths are fixed within each stage but differ between stages,
the combined token counts are 48,201,728 FineWeb-Edu tokens (48.9029%) and
50,364,416 LongCrawl64 tokens (51.0971%). This is deterministic finite-sample
variation. The registered full 35B corpus remains exactly token-balanced; the
100M resource tier must be described as approximately, rather than exactly,
1:1.

## Files

- `audit_in_place_ttt_100m_selected_stream.py`: frozen replay/audit script,
  SHA256 `834874e9ab013f6c8e29d09459aba1adffb8775449f975b9fffab9a7083641d4`.
- `in_place_ttt_100m_selected_stream_audit.json`: structured result, SHA256
  `7134c35d98fd24f4781a32aa6ea3713b2f73033218a2dc762a3f04ee4e6fd7f0`.

This audit establishes sample composition and order reproducibility. It does
not replace the canonical full-corpus independent audit, execute training, or
release a formal result.
