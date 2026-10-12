# Full ZsRE counterfactual forgetting benchmark

## Motivation

The 26-subject natural duplicate pilot established a 3.64-point cosine-to-oracle gap and a low-dimensional residual gate improved action accuracy from 91.46% to 95.12%. Its paired confidence interval remained inconclusive because only a few subjects exposed cascading errors. The next calibration stage expands the same causal mechanism to the complete canonical ZsRE subject split.

## Construction

For every canonical ZsRE fact, build one three-event sequence:

1. write the source `ground_truth` under the original prompt;
2. write the independent same-subject relation-specificity fact and retain the first fact;
3. present the source rephrase with `target_new` and locally replace only the first fact.

No answer is generated. Original, counterfactual, rephrase and locality fields all come from the downloaded ZsRE record. Records whose normalized source and target values are identical are explicitly excluded and hashed in the manifest (9/11/18 records), leaving 258/256/516 subject-disjoint calibration/development/test units. Four subject pairs are interleaved per session. Development and test are transformed mechanically but remain unscored during calibration.

## Calibration gate

Repeat the residual cosine-top controller with nested leave-one-session-out on all 258 valid calibration subjects. Freeze the controller only if both paired query and action gains over cosine have bootstrap 95% lower bounds above zero, hard-retain and local-update action accuracies exceed 90%, and no development or test record has been loaded by the evaluator.

The first 258-unit run made query gain significant (+0.91 points, CI `[0.13,1.77]`) while action gain remained borderline (+1.68 points, CI `[-0.13,3.75]`). Its single four-subject inner tune session produced noisy thresholds. The repair fixes the consistently selected 200-epoch fit and chooses each outer fold's threshold through five-fold grouped inner out-of-fold causal replay over all 64 available training sessions. The threshold objective is causal query accuracy, then action accuracy; the untouched outer session remains the only reported fold result.

The first grouped implementation was stopped after eight minutes before producing any score because it redundantly refit the same convex head for three readout seeds and replayed every query at every threshold. The efficient protocol trains one head per inner fold, gathers subject-grouped out-of-fold action logits once, and selects the threshold by action accuracy. It then reuses that action controller with all three frozen readout seeds. The outer evaluation, feature set, threshold grid and 200-epoch fit are unchanged.
