# Qwen low-rank memory geometry and readability plan

## Objective

Identify an interference-resistant Qwen-derived key/value representation before spending further compute on the forgetting controller. The experiment separates memory readability from forgetting decisions by constructing a perfect semantic active state at each development boundary.

The sealed test split from `qwen_lowrank_forgetting_v1` will not be loaded during representation diagnosis or representation selection.

## Fixed data and model

- Reuse the existing 128 train and 32 development source groups and their hashes.
- Keep the 32 test source groups sealed.
- Freeze the final Qwen3-4B 1B-token checkpoint.
- Fit every mean, projection, covariance transform, and hyperparameter using train plus development names only.
- Use representation seeds 2301, 2302, and 2303.

## Stage G1: representation geometry

Measure the following separately for entity keys and value vectors:

- effective rank and covariance condition number;
- mean, 95th percentile, 99th percentile, and maximum off-diagonal cosine similarity;
- exact-key signal versus cross-talk after 1, 2, 4, 8, 16, and 32 simultaneous components;
- candidate top-1 accuracy, mean reciprocal rank, and correct-versus-best-wrong margin.

Include a deterministic random-code representation as an upper-bound control. It cannot be selected as the Qwen-grounded method.

## Stage G2: frozen representation sweep

Evaluate these preregistered variants on train and development only:

1. `rp128`: current centered mean-token representation followed by a 128-dimensional Gaussian projection.
2. `rp128_whiten`: the same projected vectors with covariance whitening fitted on train plus development names.
3. `rp256_whiten`: the same method at 256 dimensions.

Whitening uses an eigendecomposition of the projected train/development covariance and a fixed regularizer chosen from `{1e-4, 1e-3, 1e-2}` by train readability. Development is used only for the final variant gate. Test vectors, if eventually released, must reuse the selected frozen mean, projection, eigenvectors, eigenvalues, and regularizer.

If all three Qwen variants fail, stop before controller training and add a second preregistered study of contextual Qwen hidden-state keys. Do not tune controller depth, epochs, or decision thresholds to compensate for unreadable memory.

## Perfect-state readability evaluation

At every decision boundary, reconstruct the exact active facts from semantic events and materialize one memory component for each currently valid entity-value pair. This removes any forgetting-policy error. Query the same fixed candidate set used by the forgetting experiment.

A representation passes only if all conditions hold on development:

- mean candidate top-1 accuracy at least 90%;
- minimum projection-seed accuracy at least 85%;
- every event family accuracy at least 85%;
- accuracy with up to 16 active components at least 85%;
- random-code control accuracy at least 95%, confirming the diagnostic itself is capable of reaching the ceiling.

Select the lowest-dimensional Qwen variant that passes every condition. If multiple 128-dimensional regularizers pass, select the largest regularizer within one accuracy point of the best train result.

## Stage G3: forgetting-controller v2

Freeze the selected representation and rerun the same four actions, controller architecture, seeds, fixed policies, ablations, and development gate as v1. No v1 test result exists, so the original sealed test split remains valid.

The v2 development gate remains:

- greedy oracle mean accuracy at least 90%;
- learned mean accuracy at least 88%;
- learned minimum seed accuracy at least 85%;
- learned within 3 points of oracle;
- learned at least 8 points above the stronger of the best fixed policy and global-only learned policy;
- source-group bootstrap lower bound for that gain above 5 points.

Only after all checks pass may the test split be loaded and scored once. The independent auditor must reconstruct the selected transform, perfect-state diagnostic, controller records, gate, hashes, and test lock/release.

## Stage G4: external validity after a positive test

If G3 passes test, move to natural-data experiments in this order:

1. ZsRE sequential updates for fact replacement and retention.
2. RULER/QA for long-context retrieval with distractor load.
3. LongMemEval-S and LoCoMo for multi-session retention and revocation.
4. MMLU as a non-regression check on base knowledge.

Report task quality, controller actions, memory size, update latency, peak GPU memory, and performance as a function of session length. Use at least three controller/model seeds for the main comparison.

## Interpretation rules

- G1/G2 success supports only a readable Qwen-derived memory substrate.
- G3 success supports learned forgetting under controlled multi-component interference.
- Natural-task gains are required for an end-to-end language-model claim.
- A failed readability gate is evidence against the current representation, not against adaptive forgetting in a readable memory.
