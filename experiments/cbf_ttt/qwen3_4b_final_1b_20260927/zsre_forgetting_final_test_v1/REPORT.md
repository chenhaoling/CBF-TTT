# ZsRE learned forgetting: one-time held-out test

## Protocol

The final run used the calibration-trained 13-feature residual forgetting head and the development-selected threshold 0.675. No controller weight, feature, candidate selector, or threshold changed after development. The runner loaded the 516-unit subject-disjoint test once after verifying the passing boundary summary and its independent audit.

The update rule remains forgetting-only: the current fact is always written, while the controller decides whether to remove the most related old session-memory item. The Qwen3-4B 1B-token checkpoint and frozen contextual/pair features are unchanged.

## Main result

| Metric | Learned forgetting | Cosine local | Paired gain | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| Query accuracy | 77.23% | 70.67% | **+6.59 pp** | **[5.43, 7.79]** |
| Action accuracy | 95.16% | 82.43% | **+12.73 pp** | **[10.27, 15.25]** |

The relative improvements are large and confidently positive across 516 paired subject units.

| Event/query family | Learned | Cosine local |
|---|---:|---:|
| Primary initial action | 100.00% | 99.61% |
| Same-subject hard retain action | 96.12% | 73.84% |
| Counterfactual update action | 89.34% | 73.84% |
| Current exact query | 98.51% | 88.44% |
| Current rephrase query | 50.65% | 45.00% |
| Historical query | 95.27% | 97.52% |

The unchanged absolute gate did not pass: overall action was below 96% and update action below 92%. This limits the claim to robust superiority over the cosine heuristic; it does not support near-oracle reliability. Rephrase readout remains the largest query bottleneck.

## Audit and resources

- Independent audit: passed every raw hash, identity, aggregation, interval, threshold, source, and 516-unit check.
- Code commit used to launch: `7505257`.
- Runtime: 82.47 seconds.
- Peak CUDA memory: 8.281 GiB allocated, 10.350 GiB reserved.
- Remote root: `/home/ctj/cbf_ttt_zsre_forgetting_final_test_v1`.

Raw test traces remain on the server. This directory archives the compact summary, resource record, manifest, audit, and terminal status.

## Claim boundary and next stage

The result establishes natural, subject-disjoint evidence that learned state-dependent forgetting improves a constructed Qwen-derived candidate memory. It does not establish that the original dense `ttt_conv/ttt_proj` writer stores readable factual content. Earlier Correct/Wrong/Zero/Random interventions showed that it does not at this checkpoint.

The next architecture stage therefore uses the already validated Qwen low-rank key-value fast memory rather than attaching the controller to a content-free dense delta. It must replace symbolic revoke/reset cues with automatic text/activation evidence, support natural multi-token values, and evaluate causal forgetting with fresh Qwen readout. Existing ZsRE test results are treated as observed; a new source-disjoint benchmark is required for the final architecture claim.
