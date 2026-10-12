# CBF-TTT Forgetting Paper Evidence Summary

## Paper-ready controlled claim

For additive session fast weights `M_t = alpha_t M_{t-1} + Delta_t`, a state-dependent continuous retention coefficient improves the preservation–overwrite tradeoff relative to any fixed coefficient and to a learned 0/1-only controller when both old and current information remain useful.

The claim is supported by a staged evidence chain:

1. Structured exact memory: learned local/global forgetting reached 99.74% test query accuracy versus 75.00% for the strongest fixed policy.
2. Low-rank additive fast weights: learned local/global control reached 96.68% versus 82.13% for the strongest fixed policy across three projections.
3. Continuous IID tradeoff: 56.71% of probe labels were interior; test NLL improved by 0.0839 over the best fixed alpha and 0.0698 over endpoint control.
4. Frozen OOD controller: it beat the hindsight-best fixed alpha across four shifts and three unseen projections; the failed all-endpoint gate identified strong updates as a boundary where alpha=0 is sufficient.
5. Domain-augmented controller: on three new projection seeds and eight final domains, including three unseen shift compositions, macro NLL improved by 0.1548 over hindsight-best fixed alpha and 0.0574 over endpoint control. Both source-group confidence intervals excluded zero. Independent reconstruction audits passed at every positive stage.

The actual Qwen3-4B final-checkpoint mechanism diagnostic adds a boundary result: three native TTT writes caused `<|im_end|>` to become the first-token argmax on 100% of 128 binding queries. Retaining only the last update reduced this to 4.69% and improved answer NLL by 5.4088 (source-group 95% CI [3.4846, 7.4589]) versus cumulative writing. However, no-write still had lower NLL by 0.0634 (95% CI [0.0245, 0.1021] in favor of no-write), so the experiment supports accumulation control but not content utility of the current writer.

## Mechanistic interpretation

- Low conflict or already strong current writes favor retaining old memory or the alpha=0 endpoint.
- Conflicting weak updates require more decay to suppress stale values.
- Unrelated historical facts penalize excessive global decay.
- Intermediate alpha is useful when these two forces coexist.
- Local component deletion remains stronger than a single global coefficient when the memory representation exposes separable components.

## Claims that are not supported

- The current Qwen3-4B native TTT writer has not shown reliable memory-only random fact binding.
- Restricting native writes removes the observed termination collapse, but has not yet made fast weights outperform the no-write condition.
- The controlled results do not establish improvements on ZsRE, MMLU, RULER, LongMemEval-S, or LoCoMo.
- Clean semantic conflict/duplicate ratios are available in the constructed environment; a real deployment must estimate them from activations or retrieved session state.
- The five-point label oracle is discrete; a continuous optimum was not exhaustively computed.

## Recommended manuscript positioning

Present the current result as a mechanistic and algorithmic validation of adaptive forgetting, with the negative Qwen writer experiments as evidence that memory writing and memory control are separate problems. The main method should explicitly separate a writer/readable-memory assumption from the forgetting controller. Avoid claiming an end-to-end language-model improvement until a readable TTT-compatible writer passes Correct-vs-Wrong/Empty content gates and the controller improves held-out natural tasks.

Key reports:

- `experiments/cbf_ttt/structured_forgetting_v1/REPORT.md`
- `experiments/cbf_ttt/lowrank_forgetting_v1/REPORT.md`
- `experiments/cbf_ttt/continuous_forgetting_v1/REPORT.md`
- `experiments/cbf_ttt/continuous_stress_v1/REPORT.md`
- `experiments/cbf_ttt/continuous_robust_v2/REPORT.md`
- `experiments/cbf_ttt/forgetting_only_validation_v1/REPORT.md`
- `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/native_accumulation/REPORT.md`
