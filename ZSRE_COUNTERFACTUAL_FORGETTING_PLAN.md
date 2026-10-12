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

The efficient calibration passed both frozen statistical gates: query gain +1.03 points (95% CI `[0.34,1.76]`) and action gain +2.33 points (CI `[0.65,4.13]`). The controller threshold is frozen at 0.4. Formal development requires confident positive query and action gains, at least 96% overall action accuracy, and primary-initial/hard-retain/local-update floors of 98/95/92%. Test is loaded exactly once only if all six checks pass.

Formal development at commit `d7757ae` improved query accuracy by 3.51 points (95% CI `[2.27,4.82]`) and action accuracy by 7.03 points (CI `[4.56,9.77]`) over cosine-local, but reached only 91.41% overall action, 87.89% hard-retain, and 86.72% primary-update accuracy. The absolute gate failed and test remained unopened; the independent audit verified this. The next registered stage is development error attribution followed by a single threshold-only diagnostic. If threshold recalibration cannot meet all absolute floors, controller V2 will use separate conflict and retention confidence terms with relation-balanced subject-grouped training. V2 is selected using calibration plus development, then frozen before a one-time test. The write gate is fixed at one throughout; the learned variable is the retention of old session fast memory.
