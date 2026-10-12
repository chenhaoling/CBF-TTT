# ZsRE contrastive key alignment plan

## Trigger

The frozen contextual readout passed original/updated exact and rephrase checks on development, but locality unknown accuracy was 88.26%, below the preregistered 90% threshold. Test remained sealed. This follow-up is frozen before training any alignment head or reading test.

## Objective

Learn a small calibration-only key transform that keeps a factual prompt close to its valid rephrase while separating same-subject questions about another relation. Values, memory construction, candidate sets, subject splits and the final Qwen3-4B backbone remain frozen.

## Alignment head

- One independent head for each projection seed 5101/5102/5103.
- Residual rank-16 map: `normalize(x + W_up(tanh(W_down x)))`.
- `W_up` starts at zero, so checkpoint 0 is mathematically the frozen contextual key (up to the final FP32 normalization roundoff). The separate `identity` arm bypasses the head and must reproduce the v1 raw records byte for byte.
- Train for 300 full-batch epochs on the 267 calibration subjects.
- AdamW learning rate `0.01`, weight decay `1e-4`.
- Symmetric prompt/rephrase InfoNCE temperature `0.07`.
- Relation-aware arm adds `softplus(0.2 + sim(prompt, locality) - sim(prompt, rephrase))` for both locality prompts, weighted 0.5.
- Gradient clipping at 1.0.

## Arms

1. `identity`: no learned transform; exact bridge to the failed v1 readout.
2. `positive_only`: symmetric prompt/rephrase InfoNCE with in-batch negatives.
3. `relation_aware`: the same loss plus explicit same-subject locality negatives; this is the primary arm.

After head training, choose `unknown_raw` independently for each arm from `{0.1, 0.2, 0.35, 0.5, 0.75}` using calibration accuracy only. No development result changes the head or threshold.

## Development and test rule

Evaluate all three preregistered arms on development. The primary `relation_aware` arm must meet the unchanged v1 gate:

- original exact at least 90%;
- updated exact at least 90%;
- original rephrase at least 70%;
- updated rephrase at least 70%;
- locality unknown at least 90%;
- minimum representation-seed overall at least 75%.

Only if every check passes, load the sealed 534-subject test once and score all three arms. Report primary-versus-identity source-session bootstrap intervals for overall, rephrase and locality accuracy.

## Stop rule

If relation-aware alignment fails development, keep test sealed. The next method must jointly learn a reference resolver and retrieval objective; do not lower the gate or add test-informed thresholds.

## Claim boundary

A positive result establishes calibration-trained semantic key discrimination on a subject-disjoint natural factual benchmark. It remains candidate retrieval on a constructed split of a public test resource, not official ZsRE knowledge-editing accuracy or open-vocabulary generation.
