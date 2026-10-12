# Full ZsRE counterfactual forgetting benchmark

## Motivation

The 26-subject natural duplicate pilot established a 3.64-point cosine-to-oracle gap and a low-dimensional residual gate improved action accuracy from 91.46% to 95.12%. Its paired confidence interval remained inconclusive because only a few subjects exposed cascading errors. The next calibration stage expands the same causal mechanism to the complete canonical ZsRE subject split.

## Construction

For every canonical ZsRE fact, build one three-event sequence:

1. write the source `ground_truth` under the original prompt;
2. write the independent same-subject relation-specificity fact and retain the first fact;
3. present the source rephrase with `target_new` and locally replace only the first fact.

No answer is generated. Original, counterfactual, rephrase and locality fields all come from the downloaded ZsRE record. Four subject pairs are interleaved per session, producing 267/267/534 subject-disjoint calibration/development/test units. Development and test are transformed mechanically but remain unscored during calibration.

## Calibration gate

Repeat the residual cosine-top controller with nested leave-one-session-out on all 267 calibration subjects. Freeze the controller only if both paired query and action gains over cosine have bootstrap 95% lower bounds above zero, hard-retain and local-update action accuracies exceed 90%, and no development or test record has been loaded by the evaluator.
