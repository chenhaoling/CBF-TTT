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

After fixing the open-vocabulary readout on source-disjoint data, the single-update causal intervention also failed the content gate. Full-KV candidate and first-token accuracy were both 100%, but Correct versus Wrong-twin fast weights improved answer NLL by only 0.00216 (95% CI [-0.00064, 0.00493]); Zero and norm-matched Random weights had lower NLL than Correct. This isolates the current native writer, rather than the forgetting controller, as the next component requiring training changes.

Directly fine-tuning the final checkpoint writer with balanced eight-value counterfactual assignments did not recover content binding. On eight new source groups, Correct fast memory reached 10.94% versus 12.50% for Wrong and Zero; Wrong-minus-Correct answer NLL was -0.01074 (95% CI [-0.02051, -0.00293]). Correct did improve NLL over Zero, separating a generic populated-memory effect from value-specific binding. This rules out simple writer fine-tuning as the bridge to the native dense memory and motivates a readable low-rank key-value substrate for the forgetting policy.

The Qwen-grounded rank-one bridge passed that prerequisite: both independent Q and one-time V achieved 100% candidate and full-vocabulary first-token accuracy under Correct memory, versus 0% for Wrong and 12.5% for Zero. On V, Correct improved answer NLL over Zero by 2.3364 (95% CI [2.3300, 2.3435]). This demonstrates a readable additive session memory in the actual frozen Qwen embedding/readout geometry, while leaving multi-component interference and forgetting as the next causal test.

The first multi-component Qwen-derived experiment then exposed that missing condition. On development, the full learned forgetting policy reached 27.73% candidate accuracy, below fixed global reset at 32.20%; learned-minus-fixed-global accuracy was -4.48 points with source-group 95% CI [-6.35, -2.60]. The greedy oracle reached only 28.49%, far below the preregistered 90% readability requirement. The gate failed and test remained sealed. Thus the single-component bridge does not yet supply an interference-resistant multi-component memory; the next study must improve and validate representation geometry before retraining the controller.

A perfect-semantic-state geometry diagnostic confirmed that controller error was not responsible. The exact prior representation reached 32.65% development accuracy; 128-dimensional whitening improved the generic 16-component load sweep from 69.16% to 87.03% but event-world accuracy remained 31.62%. A deterministic random-code control reached 100% throughout. Exploratory slicing of the saved rows found 17.38% historical-query accuracy and 3.56% unknown accuracy for 256-dimensional whitening, motivating a preregistered key/value factorial and contextual-hidden-state representation study.

The preregistered factorial localized this failure to keys: random keys with mean-Qwen values reached 98.18% development accuracy, whereas mean-Qwen keys with random values reached 54.88%. Replacing mean-token identifiers with the frozen final-layer hidden state at the last identifier token reached 100% on all development seeds, event families and registered 1–32 component loads. Target-span mean pooling reached only 76.51%. This releases the exact `context_last`, 256-dimensional whitening and train-selected unknown calibration into the unchanged forgetting-controller protocol; it does not itself establish a forgetting gain.

With that frozen contextual substrate, the unchanged full forgetting controller passed all development checks and was evaluated once on the sealed test. It reached 89.05% test accuracy versus 79.01% for the strongest fixed policy, a gain of 10.04 points with source-group 95% CI [9.35, 10.82]; NLL improved by 0.4657 with CI [0.4437, 0.4908]. The minimum projection-seed accuracy was 88.80%. Reference-blind matched fixed local and global-only learned reached 76.50%, supporting the value of history-conditioned local/global decisions. However, the explicit physics rule and greedy oracle matched the learned controller's query accuracy, so this controlled result does not yet demonstrate an advantage over direct access to symbolic revoke/reset cues.

## Mechanistic interpretation

- Low conflict or already strong current writes favor retaining old memory or the alpha=0 endpoint.
- Conflicting weak updates require more decay to suppress stale values.
- Unrelated historical facts penalize excessive global decay.
- Intermediate alpha is useful when these two forces coexist.
- Local component deletion remains stronger than a single global coefficient when the memory representation exposes separable components.

## Claims that are not supported

- The current Qwen3-4B native TTT writer has not shown reliable memory-only random fact binding.
- The current centered random projection of Qwen mean-token keys and LM-head value directions is not readable under multi-component interference.
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
- `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/fastweight_content_v2/REPORT.md`
- `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/final_writer_binding_v1/REPORT.md`
- `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_lowrank_bridge_v1/REPORT.md`
- `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_lowrank_forgetting_v1/REPORT.md`
- `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_lowrank_geometry_v1/REPORT.md`
- `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_contextual_geometry_v1/REPORT.md`
- `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_contextual_forgetting_v2/REPORT.md`
