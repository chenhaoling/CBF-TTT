# Final-checkpoint writer binding recovery

## Question

Can the official `ttt_conv`/`ttt_proj` writer in the 1B-token Qwen3-4B checkpoint be fine-tuned to bind the current session fact into `Delta W`, while the Qwen backbone and base `down_proj` remain frozen?

The preceding `fastweight_content_v2` intervention established a valid full-KV readout but found no native Correct-versus-Wrong/Zero/Random content effect. Earlier writer experiments started from newly initialized adapters and also exposed fixed-assignment shortcuts. This experiment starts from the actually pretrained final writer and makes every query take all eight possible values across episodes.

## Frozen data split

The audited `fastweight_content_v2` scenes are expanded only at the single differing color token. All other tokens, the query, entity, natural background, and source identity remain fixed.

- tuning train: groups 0,1,2,3,4,7 (three FineWeb-Edu and three LongCrawl64 groups);
- tuning development: groups 5,6 (one group per domain);
- final writer train: groups 0--7;
- independent Q: groups 8--15;
- sealed V: groups 16--23, scored only if Q passes.

For every group the same query is paired with all eight colors. The train schedule uses opposite color pairs and balances every value. Q/V raw sources are disjoint from train/tuning development.

## Model and optimization

- load the final step-81381 checkpoint including its pretrained TTT adapters;
- freeze every non-writer parameter;
- optimize the seven `ttt_conv`/`ttt_proj` pairs in FP32 with AdamW and BF16 forward;
- preserve the official update equation and `ttt_lr=0.3`;
- compute the write from the second 4096-token block after a frozen first-block prefix;
- discard write-time attention KV and score the query with fresh KV;
- cap each generated delta at `0.3 * ||W_base||` as a safety constraint.

Each paired update minimizes eight-way candidate CE and full-vocabulary CE for both memories, plus symmetric Correct-versus-counterfactual and Correct-versus-Zero margins. Tuning compares learning rates `3e-7` and `1e-6` at a fixed 192-update budget. The selected learning rate is retrained from the untouched final checkpoint on all groups 0--7 for 256 updates; tuning checkpoints are not promoted.

## Interventions and gates

Every evaluation uses fresh query KV and scores Correct, opposite-label Wrong, Zero, and a per-layer norm-matched Rademacher Random memory. The Q gate is fixed before scoring:

1. Correct eight-way accuracy at least 75% overall and in each domain;
2. Correct full-vocabulary first-token accuracy at least 50% overall;
3. Correct-minus-Wrong and Correct-minus-Zero eight-way accuracy at least 25 points;
4. Wrong-minus-Correct, Zero-minus-Correct, and Random-minus-Correct full-vocabulary NLL gains at least 0.02, with source-group bootstrap 95% lower bound above zero;
5. all deltas finite, no frozen parameter change, and EOS first-token rate no worse than 5%.

Only a passing Q unlocks V. V uses the same fixed gates without changing the writer, format, learning rate, thresholds, or seed. A Q failure stops the run and motivates a representation change; it does not authorize threshold relaxation or V inspection.

## Interpretation

- Correct > Wrong but not Zero: content specificity without useful memory;
- Correct > Zero but not Wrong: generic writer perturbation rather than binding;
- tuning development pass but Q fail: source/template overfitting;
- Q and V pass: the recovered writer becomes the fixed substrate for multi-block accumulation and forgetting-controller experiments;
- failure after balanced current-value supervision: stop tuning the same dense writer and move to an explicit low-rank/key-value memory representation.

The experiment is opt-in and does not change baseline configurations or default inference.
