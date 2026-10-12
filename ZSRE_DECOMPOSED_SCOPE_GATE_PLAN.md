# ZsRE decomposed scope gate plan

## Trigger

The structured three-seed bi-encoder control reached 100.00% exact and 98.38% rephrase accuracy, with only two wrong-key rephrase errors across 801 seed-level queries. Its remaining error was primarily 163 locality false retrieves. The nine-way frozen cross-linear head instead created 474 false deferrals. Candidate-key selection and update-scope rejection should therefore be modeled separately.

This protocol is frozen before fitting the decomposed gate or loading the sealed test split.

## Frozen candidate retrieval

- Reuse the three frozen contextual projections from ZsRE readout v1.
- Compute query-to-key cosine similarity for every valid session key under each projection.
- Average similarities across the three projections and select the top key.
- The candidate index is immutable: the learned gate cannot change or rerank it.

## Scope features

For the ensemble top key, concatenate:

1. the frozen normalized Qwen joint-pair hidden used in scope-resolver v1;
2. ensemble top cosine;
3. standard deviation of the top-key cosine across projections;
4. ensemble top1-minus-top2 cosine margin;
5. fraction of projections whose own argmax agrees with the ensemble argmax.

The four geometry features are standardized with fit-calibration statistics only. The model receives no answer, subject ID, relation label, split name, row ID, future event, or supplied locality ground truth.

## Arms

### `ensemble_threshold` control

Retrieve the ensemble top key if its mean cosine exceeds a threshold; otherwise defer. Select the threshold from `{0.1,...,0.9}` on the seven calibration-tune sessions.

### `decomposed_gate` primary

Train one affine binary retrieve/defer classifier for each seed 7301/7302/7303. Exact and rephrase queries target retrieve; locality queries target defer. Use balanced query groups, full-batch BCE, AdamW learning rate 0.01, weight decay 1e-4, gradient clipping 1, and 200 epochs. At epochs 10/25/50/100/200, select a probability threshold from `{0.1,...,0.9}` on calibration-tune, then select the epoch by final action accuracy, BCE NLL, and earlier epoch in that order.

The 27/7 calibration session split, frozen pair template, Qwen checkpoint, tokenizer, contextual transforms, development subjects, and test subjects remain unchanged.

## Development release gate

The primary must satisfy every check:

- exact accuracy at least 95%;
- rephrase accuracy at least 90%;
- locality/defer accuracy at least 92%;
- minimum gate-seed overall accuracy at least 93%;
- overall gain over `ensemble_threshold` at least one percentage point;
- source-session paired bootstrap 95% lower bound for the gain above zero.

Only a full pass loads the 534-subject test once. Report both arms, all query types, gate seeds, candidate-selection upper bound, source-session bootstrap intervals, NLL, MRR, and margins.

## Stop rule

If the decomposed linear gate fails, keep test sealed. The next admissible method is a calibration-trained rank-16 MLP gate over the same frozen inputs, with the same candidate selection and gates; do not change labels, data splits, or use development to select architecture or thresholds.

## Claim boundary

A positive result demonstrates automatic retrieve/defer scope gating over an explicit structured memory. It does not establish that deferred answers are factually correct, official ZsRE generation, native dense TTT writing, or end-to-end natural forgetting gains.
