# Corrected Qwen3-4B-Base formal RULER-256K execution addendum

Status: frozen before an accepted corrected-Base 256K release or formal 256K
score exists.

Freeze date: 2026-10-19 (Asia/Shanghai)

## Entry gate

Formal 256K scoring may start only after the corrected resource wrapper and
its non-authorizing review watcher pass, and a human separately creates the
registered release using the review-bound transition.  The watcher must
recompute the review content, verify every current hash, require the generic
and corrected release protocols, and require nonempty reviewer identity and
decision note.  It must not infer acceptance from a passing resource result.

## Execution

Run only on `hku-gpu2`, with no-TTT on physical GPU 0 and native TTT on
physical GPU 1.  Reuse the generic audited runner unchanged: all 13 official
RULER tasks, 100 examples per task, 262,144 tokens, seed 42, deterministic
decoding, at most 64 generated tokens, and the independently audited final
Stage-2 checkpoints.  Pass the frozen paper inference overlay so every native
TTT update is clipped to Frobenius norm `tau=1e-5`.

## Corrected final audit

After the generic paired audit passes, a separate wrapper must bind the result
to the exact Qwen3-4B-Base identity and tokenizer, corrected Stage-2 audit,
corrected 4K--128K matrix, corrected 256K resource audit, immutable review
packet, accepted release, both 256K plans, this addendum, and the paper
inference overlay.  Both child audits must carry the same overlay hash.  Any
missing artifact, input/hash drift, task/sample mismatch, failed generic
audit, or stale release fails closed.

## Claim boundary

Report the paired scores and differences from Table 1's 41.7 matched baseline
and 43.9 TTT values as a public-data YaRN extrapolation.  Do not merge 256K
into the primary 4K--128K mean and do not describe it as an exact reproduction
of the paper's private continual-pretraining corpus.  The 256K result alone
does not release update-admission exploration.
