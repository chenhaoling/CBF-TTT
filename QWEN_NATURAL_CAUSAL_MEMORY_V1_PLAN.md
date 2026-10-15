# Preregistered Qwen natural causal-memory composition V1

## Freeze point and scientific question

The structured causal ledger passed matched confirmation and lifecycle OOD,
the decomposed natural scope gate passed its one-time subject-disjoint test, and
the natural-value residual bridge passed every calibration/development gate at
the frozen scale 8. The remaining gap is compositional: can an automatic
forgetting policy maintain natural, variable-length counterfactual values over
time and make the frozen Qwen model emit the currently valid answer without
placing session facts in the query KV context?

This protocol is frozen before producing any score for the composed natural
memory. It uses only the already observed ZsRE counterfactual-forgetting
calibration and development splits. Its result is architecture-development
evidence, not a new held-out ZsRE claim. The already observed ZsRE test is not
rescored; LongMemEval-S, LoCoMo, MMLU, RULER and RULER/QA remain unopened.

## Frozen dependencies

- Qwen3-4B 1B-token checkpoint at step 81,381;
- counterfactual sequence manifest SHA256
  `c5056824efa867532a33ff460976aaaf6d709ff1900acc562fc26c4b5ae1026f`,
  calibration/dev SHA256 `04bcc19b...b3243d` / `79f6c29b...b4301d`;
- decomposed-scope checkpoint SHA256
  `0ed4dab8b77fd0fb91fdd49720c783261ab2fd5c64cd705ebe36346d99615619`;
- contextual transforms SHA256
  `6f6a696005185d80d6b8f468fa6442146dc8d9683e232e7a591676ba9515c378`;
- forgetting checkpoint SHA256
  `fc766618db0e6847da893c4db0866eaafa98bb94889f58d6ed0b9fc6b77538e1`;
- audited development-selected forgetting threshold `0.675`;
- natural-value development summary/audit SHA256
  `f13dd631...8208a` / `c27933a0...b49ee`, which fix value scale 8.

Use decomposed-scope seed 7301 and the frozen forgetting head trained with seed
8601. Neither checkpoint, threshold, scale, feature, prompt nor split may be
changed after calibration scores are observed.

## Natural online sequence

Each session contains four independent primary facts and four same-subject,
different-relation auxiliary facts. It first writes all eight initial records,
then applies four counterfactual updates, one per primary fact. The current fact
is always written. A forgetting policy decides only which old session-memory
record, if any, is deleted.

Score five registered checkpoints in every full-width session: after the eight
initial writes and after each of the four counterfactual updates. The final
two-pair calibration remainder has the corresponding three checkpoints after
four initial writes and two updates; it is retained rather than dropped. At
every checkpoint, query the exact and rephrased surface of every active primary
and auxiliary chain. The same two primary query surfaces therefore require the
original value before their update and the counterfactual value after it. This
directly measures replacement and collateral retention. Calibration has 65
sessions; development has 64 subject-disjoint sessions.

Every session receives one deterministic 12-way natural-answer candidate set:
all distinct original, updated and auxiliary targets in the session, filled if
needed with hash-selected split-local targets. Candidate construction cannot
use model scores. Case/whitespace-equivalent target surfaces map to the single
registered session candidate representative before tokenization; this prevents
semantically duplicate answer classes while preserving their factual value. No
session fact is included in the Qwen query prompt.

## Frozen memory and policy arms

Every selected record is read through the position-specific normalized frozen
Qwen LM-head residual at scale 8. Automatic arms use the same frozen contextual
candidate ranker and decomposed retrieve/defer gate.

1. `learned_auto`: the frozen state-dependent forgetting controller at threshold
   0.675, followed by automatic scope;
2. `cosine_auto`: frozen cosine local-deletion heuristic, followed by automatic
   scope;
3. `retain_auto`: never delete an old session record;
4. `global_auto`: clear all session records before every write;
5. `oracle_auto`: correct local lifecycle deletion with automatic scope;
6. `oracle_oracle`: correct lifecycle and direct active-record selection;
7. `oracle_dense`: correct lifecycle but sum every active value without key
   selection;
8. `zero`: no memory residual.

All arms share the identical prompt, candidates, frozen model and fresh KV
cache. Primary scoring is mean-token candidate NLL and 12-way candidate
accuracy at all five checkpoints. At the final checkpoint, additionally run
fixed-correct-length autoregressive generation for every arm. Record policy
actions, selected record, stale selection, memory size, query type, chain type,
checkpoint, per-row time and peak GPU memory.

## Calibration release gate

Calibration releases development only if all checks pass:

- `oracle_oracle` overall and final candidate accuracy are at least 95%;
- `oracle_auto` overall candidate accuracy is at least 88% and its gap to
  `oracle_oracle` is at most 8 points;
- `learned_auto` overall candidate accuracy is at least 68%, and primary and
  auxiliary accuracy are each at least 65%;
- `learned_auto` final fixed-length generation exact match is at least 55%;
- `learned_auto` beats the strongest fixed non-oracle arm among cosine, retain,
  global, dense and zero by at least two points, with a subject-group bootstrap
  95% lower bound above zero;
- `oracle_auto` beats `retain_auto`, `global_auto`, `oracle_dense` and `zero` in
  candidate accuracy, with every paired lower confidence bound above zero.

Use 20,000 deterministic subject-group bootstrap draws. Development uses the
identical frozen system once. It applies the same absolute checks and requires
a positive lower confidence bound for `learned_auto` over the strongest fixed
arm. Test and external benchmarks remain sealed regardless of the outcome.

## Decision rule and claim boundary

A development pass establishes end-to-end composition of learned session
forgetting, automatic natural scope and causal natural-value realization on an
already observed architecture-development split. It releases construction of a
new source-disjoint LongMemEval-S/LoCoMo forgetting benchmark and a one-time
external evaluation.

If `oracle_oracle` fails, stop and repair value realization or candidate
construction. If `oracle_oracle` passes but `oracle_auto` fails, repair scope
transfer. If both oracle arms pass but `learned_auto` fails its comparison,
improve only the forgetting controller using calibration data; do not tune the
writer or scope resolver. No outcome supports claims about the unchanged native
`ttt_conv`/`ttt_proj` writer or unrestricted open-vocabulary generation.
