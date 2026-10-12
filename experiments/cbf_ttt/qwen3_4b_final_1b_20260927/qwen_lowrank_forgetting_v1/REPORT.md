# Qwen-grounded multi-component forgetting v1

## Question

Does the forgetting policy that succeeded with synthetic random low-rank keys and values still work when the keys and values are derived from the final 1B-token Qwen3-4B checkpoint?

This experiment freezes the language model. Entity keys are centered means of Qwen input-embedding rows, value directions are centered means of Qwen LM-head rows, and both are mapped to 128 dimensions with three preregistered Gaussian projections. Only the small forgetting controllers are trained.

## Protocol

- Model: final Qwen3-4B checkpoint at step 81,381.
- Data: 128 train, 32 development, and 32 sealed test source groups; 4,608 episodes and 27,648 queries in total.
- Memory seeds: 2301, 2302, and 2303.
- Actions: retain, half decay, referenced local deletion, and global reset.
- Comparisons: four fixed actions, deterministic random, reference-blind learned, global-only learned, full learned, visible physics rule, and a greedy action oracle.
- Test release: score once only if the aggregate development gate passes.
- Runtime: 2026-10-12 15:53:39 to 15:55:36 Asia/Shanghai; controller training/evaluation time 102.67 seconds.

## Development results

| Policy | Mean accuracy | Minimum seed accuracy | Mean NLL |
|---|---:|---:|---:|
| Fixed retain | 15.60% | 15.26% | 7.2989 |
| Fixed half | 24.48% | 22.89% | 4.4405 |
| Fixed local | 17.18% | 17.12% | 6.5122 |
| **Fixed global** | **32.20%** | **31.77%** | 4.7087 |
| Random | 22.20% | 21.92% | 5.5654 |
| Reference-blind learned | 28.41% | 27.95% | 4.2556 |
| Global-only learned | 28.78% | 27.82% | 4.1408 |
| Full learned | 27.73% | 27.00% | 4.0998 |
| Physics rule | 23.59% | 23.39% | 5.0592 |
| Greedy oracle | 28.49% | 28.17% | 3.7166 |

The full learned controller was 4.48 accuracy points below the strongest fixed policy. The source-group bootstrap interval for learned minus fixed-global accuracy was `[-6.35, -2.60]` points. Its NLL was lower by 0.6089, with 95% CI `[0.4395, 0.7748]`, but the preregistered gate requires both reliable retrieval accuracy and a positive accuracy margin.

The oracle-readability check failed badly: 28.49% versus the required 90%. Therefore this run does not identify controller capacity or optimization as the main bottleneck. The current multi-component Qwen-derived representation is not a sufficiently readable associative memory under interference. The greedy oracle is also myopic, so its lower accuracy than fixed-global is not evidence that a globally optimal oracle is worse; it reinforces that one-step labels and final retrieval are unreliable in this representation.

## Gate and test status

- Passed: near-oracle proximity only.
- Failed: oracle readability, learned accuracy, all-seed accuracy, control margin, and confidence-bound margin.
- Development gate: **failed**.
- Test scored: **no**.
- Independent audit: **passed**; data hashes, split/seeds, embedding hashes, every development record, aggregate metrics, gate reconstruction, and the test lock were verified.

## Interpretation

The earlier 100% single-component Qwen bridge does not extend automatically to many simultaneous synthetic identifiers and multi-token codes. Shared prefixes and digit-token structure create correlated Qwen key/value directions; random projection preserves much of that correlation. With many outer-product components, cross-talk dominates the exact-key signal. Training the controller longer cannot repair a readout whose oracle ceiling is below the fixed baseline.

This is a useful boundary result. It preserves the earlier controlled claim that adaptive forgetting works when the memory substrate is readable, while showing that the present Qwen-derived substrate needs an interference-resistant transform before an end-to-end claim is justified.

## Next experiment

Run the preregistered geometry/readability diagnostic in `QWEN_LOWRANK_GEOMETRY_PLAN.md`. It will keep the test set sealed, compare the current transform against train/development-fitted whitening at 128 and 256 dimensions, quantify cross-talk as the number of active components grows, and use a random-code upper-bound control. Only a representation that passes the clean semantic-state readability gate will be allowed into forgetting-controller v2.

## Artifacts

- `summary.json`: full development aggregates and gate result.
- `training.json`: controller curves, label counts, embedding hashes, and raw-record hashes.
- `training_manifest.json`: frozen protocol and model paths.
- `execution_audit.json`: independent audit result.
- `manifest.json`: generated event-curriculum manifest.
- Large raw JSONL records and the controller checkpoint remain under `/home/ctj/cbf_ttt_qwen_lowrank_forgetting_v1` on `hku-gpu2`.

## Limitations

- The candidates are scored in a projected low-rank space rather than by full autoregressive generation.
- Keys and values are extracted from explicit dataset fields; no learned text parser is evaluated.
- Only one event-curriculum seed and three projection seeds are used.
- GPU peak memory was not captured by this runner; no peak value is claimed.
- No natural benchmark or test split result is available from this failed development stage.
