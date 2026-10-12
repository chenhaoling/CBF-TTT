# ZsRE contextual memory readout plan

## Objective

Establish whether the frozen Qwen contextual memory that passed the synthetic event curriculum can store and retrieve natural factual updates. This is a prerequisite for evaluating forgetting without privileged symbolic revoke/reset cues.

The source is the downloaded KnowEdit `ZsRE-test-all.json` with SHA256 `0e0214dda853a906ef02bef58c7ce2978af5a962e91fc4a10827e0125459905c`. Because this file is itself a public test resource, the experiment is reported as a subject-disjoint constructed benchmark rather than an official KnowEdit score.

## Data split

- Group all 1,301 rows by subject; there are 1,068 unique subjects.
- Deterministically select one row per subject by canonical-row hash.
- Sort subjects by `SHA256("20261201|" + subject)`.
- Use the first 267 subjects for calibration, the next 267 for development, and the remaining 534 for a sealed one-time test.
- Pack consecutive subjects into sessions of at most eight facts.
- No subject may appear in more than one split.

## Memory and queries

For each session, build two perfect semantic memories:

1. `original`: map each source prompt to its first ground-truth answer;
2. `updated`: map the same prompt to `target_new`.

For each fact and memory, evaluate:

- `exact`: query with the source prompt;
- `rephrase`: query with `rephrase_prompt`;
- `locality`: query each relation-specificity prompt and require `unknown`, because that relation was not written to session memory.

The candidate set contains all original and updated answers in that session plus `unknown`. This is candidate retrieval, not open-vocabulary generation or the official knowledge-editing metric.

## Representation

- Frozen final Qwen3-4B checkpoint.
- Final-layer hidden state at the last target token.
- Prefixes: `Memory key: ` for questions and `Memory value: ` for answers.
- Three new projection seeds: 5101, 5102 and 5103.
- 256-dimensional Gaussian projection followed by covariance whitening fitted on calibration names only.
- Choose regularizer from `{1e-4, 1e-3, 1e-2}` and unknown score from `{0.1, 0.2, 0.35, 0.5, 0.75}` using calibration only. Use the same largest-regularizer-within-one-point rule as the prior geometry experiment.

## Development gate

All conditions must pass before test is loaded:

- original exact accuracy at least 90%;
- updated exact accuracy at least 90%;
- original rephrase accuracy at least 70%;
- updated rephrase accuracy at least 70%;
- locality unknown accuracy at least 90%;
- minimum representation-seed overall accuracy at least 75%.

If the gate fails, leave test sealed and train a contrastive prompt/rephrase key alignment layer using calibration subjects only. Do not lower these thresholds.

## Follow-up after a pass

Construct natural sequential sessions with true revisions, unrelated insertions and same-subject locality facts. Infer reference candidates from contextual key similarity and train a controller using only activation-derived features. Compare against fixed retention, a calibrated similarity-threshold rule, the frozen synthetic controller, a reference-blind controller and a counterfactual oracle.

## Claim boundary

A positive result supports natural prompt/rephrase readability of the low-rank substrate. It does not establish official ZsRE editing performance, native dense TTT writing, open-vocabulary generation or a learned forgetting advantage.
