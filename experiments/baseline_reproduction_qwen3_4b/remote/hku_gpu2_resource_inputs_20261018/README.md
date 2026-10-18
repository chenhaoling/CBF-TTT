# hku-gpu2 exact resource-input evidence

This directory records the CPU-only construction and independent audit of the
five exact-length two-record inputs used by the 2xRTX5090 resource ladder.
The generated plaintext JSONL files remain at
`/home/ctj/in_place_ttt_2x5090_resource_inputs_099d58cd_20261018` on
`hku-gpu2`; they are intentionally omitted from Git. `output_sha256_after_audit_r2.txt`,
the manifest and the independent audit retain their hashes and counts.

The first audit invocation failed before reading data because commit `099d58cd`
mis-mapped the CLI name `source_audit`. The build itself had exited zero and
atomically published all files. Commit `8019cc55` fixed the CLI mapping and
added a regression test; the second independent audit recomputed all outputs
and passed. Both attempts and their resource logs are preserved.
