# Corrected Qwen3-4B-Base RULER-256K release addendum

Status: frozen before any corrected-Base 256K resource outcome, review packet,
accepted release, or formal 256K score exists.

Freeze date: 2026-10-19 (Asia/Shanghai)

## Purpose

The generic 256K release binds the underlying paired resource audit.  This
addendum additionally binds that audit to the corrected Qwen3-4B-Base wrapper,
the exact Base identity, and the corrected primary chain before a human can
authorize formal 256K scoring.  It does not authorize a run.

## Non-authorizing review packet

A CPU-only watcher on `hku-gpu2` waits for
`completed_corrected_ruler_256k_resource`.  It verifies the underlying and
corrected resource audits, their SHA-256 relationship, model revision
`906bfd4b4dc7f14ee4320094d8b41684abff8539`, the exact model identity, and the
generic release validator.  It then writes one immutable packet with:

- protocol `in_place_ttt_qwen3_4b_base_ruler_256k_release_review_v1`;
- `ready_for_human_review=true` and `formal_release_created=false`;
- the intended future release path;
- exact paths and hashes for both resource audits, model identity, release
  validator, review builder, transition tool, and this addendum;
- required human fields `accept`, `reviewed_by`, and `decision_note`.

The watcher never passes `--accept`, never invokes the transition tool, never
writes the release path, and never starts a GPU process.

## Human transition

The only supported release transition reloads the packet, rehashes every
bound file, reruns the generic and corrected resource validation, and requires
the recomputed review dictionary to equal the packet.  It then requires an
explicit `--accept`, nonempty `--reviewed-by`, and nonempty `--decision-note`.
On acceptance it preserves the generic formal-release protocol needed by the
existing runner and adds the corrected Base protocol, model revision,
corrected-resource hash, review-packet hash, and this addendum hash.  Creation
is atomic and every overwrite is rejected.

## Stop and claim rules

Both tools require hidden GPUs and hostname `user-Super-Server`.  Missing or
changed evidence, another model revision, a failed prerequisite, a missing
live predecessor handle, a pre-existing target, or visible CUDA fails closed.
A passing review packet is not acceptance.  Formal 13-task, 100-example 256K
scoring remains a public-data YaRN extrapolation, stays outside the primary
4K--128K mean, and cannot authorize update-admission exploration by itself.
