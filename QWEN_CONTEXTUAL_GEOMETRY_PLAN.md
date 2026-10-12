# Qwen contextual key/value geometry plan

## Motivation

The registered mean-token geometry study failed its development gate. Global whitening improved the 16-component cross-group load sweep from 69.16% to 87.03%, but perfect-state event accuracy stayed near 32%. Known-value accuracy was about 38%, unknown accuracy about 2%, and historical queries were the weakest. The random-code control remained 100%.

These observations suggest that globally fitted whitening cannot separate the deliberately similar identifiers inside one source world, such as three `unit_<same namespace>_<index>` keys. This follow-up is frozen before any contextual representation result is observed.

## Fixed scope

- Reuse the same final Qwen3-4B checkpoint and the same 128 train / 32 development groups.
- Keep the existing 32 test groups sealed and do not load them.
- Keep memory construction, perfect semantic states, candidate sets, metrics, loads and seeds unchanged.
- Freeze every Qwen parameter; this stage trains no forgetting controller.

## C0: key/value factorial diagnosis

Using the previously selected 256-dimensional whitened mean-token representation, evaluate:

1. Qwen keys with Qwen values;
2. Qwen keys with deterministic random values;
3. deterministic random keys with Qwen values;
4. deterministic random keys with random values.

The two hybrids are diagnostics and cannot be selected as the method. They identify whether within-world interference is primarily caused by entity keys, value codes, or both.

## C1: contextual Qwen representations

Encode each identifier using the frozen final Qwen backbone:

- entity prompt: `Entity identifier: {entity}`;
- value prompt: `Assigned value code: {value}`.

Compare two preregistered pooling rules:

1. `context_last`: hidden state at the final identifier token;
2. `context_mean`: mean hidden state over identifier tokens only.

For both rules, use the final transformer layer, then a seed-specific 256-dimensional Gaussian projection and covariance whitening. Fit means, projections and covariance transforms on train plus development names, as in the prior registered geometry study.

The whitening regularizer is selected from `{1e-4, 1e-3, 1e-2}` using train readability. Because the prior result exposed a systematic unknown-detection failure, select `unknown_raw` from `{0.1, 0.2, 0.35, 0.5, 0.75}` using train only. For each regularizer, select the threshold with highest train accuracy, breaking ties by proximity to 0.35. Then select the largest regularizer whose optimized train accuracy is within one point of the best regularizer.

## Development gate

Apply the unchanged readability conditions to each contextual variant:

- mean candidate top-1 accuracy at least 90%;
- minimum representation-seed accuracy at least 85%;
- every event family accuracy at least 85%;
- 16-component load-sweep accuracy at least 85%;
- random-code control at least 95%.

Select `context_last` if it passes; otherwise select `context_mean` if it passes. If neither passes, stop before controller training and design a supervised contrastive key encoder using train groups only.

## C2: controller release rule

Only a passing contextual representation may enter forgetting-controller v2. The selected prompt, pooling, projection, whitening regularizer and unknown threshold are frozen before controller training. The original v1 test split remains sealed until the unchanged v2 controller development gate passes.

## Claims

- A positive C1 result supports a readable frozen-Qwen contextual memory substrate.
- The factorial result localizes the source of interference but is not itself a deployable method.
- No forgetting or natural-task claim follows until the subsequent controller and external evaluations pass.
