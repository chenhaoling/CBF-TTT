# Native dataloader closure probe

This directory records a CPU-only, metadata-only probe executed on
`hku-gpu2` at 2026-10-19 04:18 HKT. `CUDA_VISIBLE_DEVICES` was empty, so the
probe did not start or reserve a GPU process.

The probe loaded the frozen public-corpus pilot through the exact tokenizer
builder and `process_pretrain_example` implementation used by the official
training entry point. It checked the first FineWeb-Edu and LongCrawl64 record
from both the 32K and 128K stages. It did not emit source text or token IDs.

For every record, the native transform produced exactly one chunk, preserved
the registered sequence length, appended exactly one terminal EOS, set labels
equal to input IDs and produced the same little-endian uint32 token SHA256 as
the corpus provenance. The result and exact code/data bindings are stored in
`probe.json`.

The SHA256 of `probe.json` is
`1e31a02bf8b623ad119347f1c680ee4a392a4ba18a835fb8d3371494bc118502`.

This proves the pilot corpus representation is closed under the pinned native
plaintext transform. It does not replace the full-corpus canonical audit and
does not release a training or evaluation gate.
