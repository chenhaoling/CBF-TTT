# Native dataloader pairing probe

This directory records two independent CPU-only replays of the pinned native
VeOmni dataloader on `hku-gpu2`. The probe used two Gloo ranks, seed 42, the
32K pilot corpus, global batch 64, microbatch one, `rmpad=false` and the
configured two workers. `CUDA_VISIBLE_DEVICES` was empty. It did not load a
model, emit source text or persist token IDs.

Each rank produced 32 one-example microbatches. Both independent launches
produced the same ordered hashes on rank 0 and rank 1. The rank streams were
disjoint, all 64 hashes were unique and every hash matched a registered pilot
provenance row. The two run JSON files are byte-identical.

The pinned Datasets implementation reports that a single JSONL file has one
data shard, so one of the configured two workers exits without consuming
data. This leaves one effective worker per data-parallel rank. It does not
change sample order or rank separation, but measured timing must include this
actual worker behavior.

Artifacts:

- `probe_in_place_ttt_dataloader_pairing.py`:
  `23864c40bef8b260b2a883a7694e9d611a2e1dd93139912c07a84706b30d59bf`
- `in_place_ttt_dataloader_pair_a.json`:
  `1030083c664521b56decdb8a9415af1c13c3f002c8544db2c34dbd80c590e0db`
- `in_place_ttt_dataloader_pair_b.json`:
  `1030083c664521b56decdb8a9415af1c13c3f002c8544db2c34dbd80c590e0db`
- `in_place_ttt_dataloader_pair_comparison.json`:
  `a8a93b7e8924b73ed24c4a40280db259be4dfc83d952d67a27a934a207c0c5e0`

This establishes deterministic paired ordering for the registered pilot and
the pinned loader implementation. The formal runs remain responsible for
binding the published full corpus, exact configs and checkpoints.
