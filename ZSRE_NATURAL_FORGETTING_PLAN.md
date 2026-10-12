# Natural ZsRE sequential forgetting plan

## Motivation

The decomposed scope gate can decide whether a query should retrieve a structured update, but it has not yet controlled accumulated memory. This experiment turns previously excluded duplicate ZsRE rows into natural-text update chains and tests whether automatic scope evidence can drive local forgetting.

## Construction

- Verify the original `ZsRE-test-all.json` hash.
- Preserve the canonical benchmark's subject ownership: calibration subjects remain calibration, development subjects remain development, and test subjects remain sealed test.
- Within each subject, group rows by normalized original ground-truth set. Rows in one group describe the same pre-edit fact.
- Keep one deterministic row per distinct normalized `target_new` and retain only groups with at least two distinct targets.
- Put the canonical row first when it belongs to the chain; order remaining versions by source-row hash. This is a deterministic counterfactual update order, not a claim about real timestamps.
- Pack up to eight subject/fact chains into a session. Interleave versions by round: each chain's first version introduces an independent fact (`retain`); later versions supersede that chain's active version (`local forget`).

Expected chain counts from schema-only audit are 26/31/52 for calibration/development/test, with 30/42/68 local transitions after distinct-target deduplication. Exact counts and file hashes are recorded by the builder.

## Policies

1. `fixed_retain`: append every update.
2. `fixed_global`: clear all memory before every update.
3. `fixed_top`: always delete the current top cosine candidate before appending.
4. `cosine_local`: delete the top candidate only when the frozen ensemble cosine exceeds the calibration-selected 0.3 threshold.
5. `learned_local`: use the frozen decomposed scope gate; delete only its selected top candidate when retrieve probability exceeds its frozen 0.4 threshold.
6. `oracle_local`: delete the active record with the same hidden chain identity, which policies never observe.

All policies use the same frozen decomposed scope gate for query readout. The gate checkpoint, Qwen model, transforms, thresholds and three gate seeds are inherited without retraining.

## Queries and metrics

After every update, query every active chain with:

- its current exact prompt and rephrase;
- every previously observed exact prompt and rephrase, all of which must resolve to the newest value;
- the current version's two locality prompts, which must defer.

Report overall/current/historical/locality accuracy, NLL, action accuracy, stale-read rate, memory size, and update/query counts. Pairwise policy intervals bootstrap fact chains rather than queries or packed sessions.

## Stage A: calibration-only feasibility pilot

Run all calibration chains only. Test real frozen-Qwen pair extraction, runtime, peak memory, oracle readability and policy separation. Stage A may diagnose and repair implementation errors, but may not load development or test. Before Stage B, freeze the development gates in this file from calibration evidence and commit them.

## Stage B release rule

After the calibration pilot is archived, run development once under frozen gates. Load test once only if every development check passes. Test remains sealed during Stage A and all code repair.

## Claim boundary

This is a deterministic natural-text counterfactual sequence made from ZsRE edits, not an official timestamped ZsRE task. A positive result would show causal benefit from deleting stale structured-memory records. It would not establish native dense `ttt_proj`/`ttt_conv` forgetting or official generative knowledge-editing accuracy.
