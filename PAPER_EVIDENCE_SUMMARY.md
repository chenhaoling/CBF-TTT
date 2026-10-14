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

The natural ZsRE scope study removes that symbolic-reference privilege for memory applicability. A decomposed gate freezes three-projection cosine candidate selection and learns only retrieve versus defer from frozen joint-Qwen evidence and score geometry. After a six-part development gate passed, the original 534-subject test was scored once. The learned gate reached 98.47% versus 88.58% for an ensemble similarity threshold, a 9.90-point gain with 67-session bootstrap 95% CI [8.57, 11.24]. Exact, rephrase and locality accuracy were 100.00%, 97.75% and 98.06%, with 98.46% minimum seed accuracy. This validates automatic natural update-scope evidence in structured memory, but does not yet show causal benefit from deleting accumulated memory.

The independent V19 confirmation directly established that causal benefit on a new controlled contextual-memory corpus. The registered locality-safe oracle reached 96.25% joint update-and-retention success versus 39.38% for the outer-fold fixed comparator, a 56.88-point gain with subject-cluster 95% CI [52.81, 61.25]. Exact complete-key deletion reached 86.56% and explained 82.97% of the oracle gap. A nested five-fold, three-seed learned forgetting policy reached 86.67%, only 0.10 points above the exact rule; its subject-cluster interval [-1.56, 1.88] crossed zero and only one of five inner selections met the registered constraints. This confirms selective forgetting as the causal mechanism while rejecting a learned-policy advantage on a corpus whose conflicts are already exposed by exact keys. The next preregistered study tests natural paraphrase conflicts before any semantic controller is trained.

The disjoint natural-paraphrase headroom study then supplied that missing case. On 320 sealed CounterFact/MQuAKE episodes, relation-aware deletion reached 92.19% joint success versus 36.56% for exact surface deletion, a 55.63-point gain with subject-cluster 95% CI [50.00, 61.25] and relation-cluster CI [49.17, 61.74]. Every fold was positive; CounterFact and MQuAKE gains were 56.86 and 38.10 points. Update, retention and locality were 95.31%, 99.22% and 100%. All registered gates and independent audits passed. This is an oracle relation-ID result: it establishes semantic forgetting headroom and releases learned residual training, but does not yet establish an inference-time semantic controller.

The learned semantic-residual development chain is informative but has not passed its release gate. On 63,782 relation- and subject-disjoint pairs, mean-pooled Qwen cosine reached only 54.10% balanced accuracy and 0.393 AUROC. Switching to the causal last-token state and applying the registered rank-16 contrastive projection raised these to 69.94% and 0.775. A nested five-fold, three-seed 64-unit utility/risk controller further reached 74.08% and 0.811. A symmetric frozen-Qwen pair cross-encoder materially improved the result to 83.56% balanced accuracy and 0.880 AUROC, with every fold above 76% and every seed above 83%. However, lexical-hard-negative specificity remained 72.89%, and no inner fold met the frozen 80% recall/95% specificity/3% risk-violation constraint. An independent row-level audit passed all checks. The result supports joint pair encoding as the main missing semantic signal while rejecting the current controller as deletion-safe. The registered stop on CounterFact/MQuAKE tuning is active; the 320 model outcomes and external benchmarks remain unused.

The source-disjoint FewRel study then tested conservative high-confidence deletion on new relations. Official FewRel was split into 48 training, eight development, eight internal-test and 16 official external-test relations, with subject entities and masked surfaces disjoint across roles. On 9,554 development pairs, the three-seed ensemble reached 70.76% balanced accuracy and 0.768 AUROC, improving substantially over the 55.59%/0.586 zero-shot margin but failing the full binary gate. At zero relation-bootstrap upper false-deletion bound, only 9/4,777 positives could be deleted, giving 0.19% recall versus the registered 25% floor. Random and lexical-hard-negative specificity were similarly weak, so threshold changes cannot repair the broad unseen-relation separation error. Independent audits passed; internal and external test features were never generated and the FewRel branch stopped as registered.

A separately preregistered DocRED relation-description space did not repair this
failure. After excluding every relation reserved by FewRel, the audited corpus
contained 51 representation-training, eight development and eight sealed
source-test relations, with 10,952/492/356 statement instances and no held-out
relation descriptions released. Frozen Qwen3-4B last-token extraction covered
11,444 train/development statements plus 51 training descriptions; all 11
feature checks and exact-batch reproduction passed. A three-seed rank-64
statement/description projection improved development balanced accuracy from
54.89% raw cosine to 63.37%, but reached only 0.6564 AUROC. Two held-out
relations were below chance (P31: 0.4666; P527: 0.4193), lexical-hard-negative
specificity was 46.91%, and neither the full nor selective safety gate was
feasible. An independent checkpoint reconstruction passed all 11 checks. The
source test and both FewRel test partitions therefore remain sealed. This
rejects relation-description metric learning as the missing bridge for safe
natural forgetting under the current frozen representation.

Returning to the forgetting-only mainline, a fresh Qwen-grounded continuous
cohort study independently controlled history, last-2 and last-1 retention on
the grid `{0, 0.5, 1}` while always admitting the current write. Across three
projection seeds, the continuous controller reached 88.35% development
accuracy versus 66.41% for the strongest fixed policy, a 21.94-point gain with
source-group 95% CI `[20.15, 23.80]`; NLL improved by 0.7271, CI
`[0.6822, 0.7752]`. It recovered 96.9% of the available dense-reader oracle
gain. The complete release gate nevertheless failed: historical accuracy was
65.49%, minimum family accuracy 71.59%, and retention MAE 0.1787. The oracle
itself reached only 89.06%/67.90% overall/historical accuracy, localizing the
dominant failure to the reader rather than the controller. Physics and exact
referenced-batch deletion reached 88.54%, so learned control did not beat the
visible symbolic rule. All 13 independent audit checks passed and test stayed
unencoded.

The preregistered causal reader diagnostic then isolated dense cross-key
interference. On the same development sessions and frozen embeddings, the
27-grid oracle improved from 89.06% with all active components to 99.33% with
top-1 key routing; historical accuracy rose from 67.90% to 98.00% and minimum
family accuracy from 71.82% to 97.34%. Top-2 reached 97.12%, while top-4/8/16
collapsed toward dense performance. The ineligible exact-key upper bound was
100%. Top-1 passed every registered check on all seeds and was selected by the
smallest-reader rule. An exact reproduction check matched every dense action,
query and NLL before comparing readers, and all 11 independent audit checks
passed. Under top-1 reading, physics reaches 92.90% while the oracle reaches
99.33%, creating a clean 6.44-point target for the separately frozen
fresh-corpus controller confirmation.

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
- The ZsRE result establishes structured retrieve/defer scope accuracy, not official generative knowledge-editing accuracy. Improvements on MMLU, RULER, LongMemEval-S, LoCoMo, and native dense TTT remain unestablished.
- Clean semantic conflict/duplicate ratios are available in the constructed environment; a real deployment must estimate them from activations or retrieved session state.
- The five-point label oracle is discrete; a continuous optimum was not exhaustively computed.
- The current symmetric pair natural-semantic controller does not meet its registered deletion-safety constraint, despite improving pair AUROC to 0.880. Its model-level benefit is therefore unestablished.

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
- `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/zsre_decomposed_scope_gate_v1/REPORT.md`
- `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/memory_interference_v19_final/REPORT.md`
- `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/natural_semantic_headroom_v1/REPORT.md`
