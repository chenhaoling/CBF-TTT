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

The threshold diagnostic causally replays the frozen head on development at thresholds 0.1 through 0.9 in increments of 0.1. It reports all query/action metrics and paired intervals at every threshold, ranks thresholds by action accuracy, worst hard-retain/update accuracy, then query accuracy, and independently audits every raw replay. This diagnostic is allowed to use development after the formal gate failed; it must not load test. A threshold passes only under the original six checks. If none passes, threshold-only recalibration is rejected before implementing V2.

The audited coarse sweep found no passing point, but threshold 0.7 reached 96.61% overall action, 98.05% hard-retain, and 91.80% primary-update accuracy: one additional correct update among 256 would satisfy the remaining 92% floor. Before rejecting scalar calibration, perform the registered sparse boundary sweep at 0.625, 0.650, 0.675, 0.725, 0.750, and 0.775. These points are fixed from the adjacent coarse interval and exclude previously scored 0.6/0.7/0.8. If any passes all original checks, select the passing point by action accuracy, worst difficult-event accuracy, then query accuracy and freeze it for test. Otherwise proceed to joint candidate/retain V2.

The independently audited boundary sweep found two passing values, 0.625 and 0.675. The registered ranking selected 0.675: query accuracy 77.91%, action accuracy 96.74%, hard-retain 98.05%, and primary-update 92.19%. Paired gains over cosine-local were +6.27 query points (95% CI `[4.70,7.95]`) and +12.37 action points (CI `[8.98,15.89]`). Freeze the original calibration-trained head and threshold 0.675. The next and only permitted test operation is a one-time 516-unit causal replay; no controller refit, feature change, or further threshold selection may occur before it.

The one-time test completed at the frozen threshold and passed independent audit. Learned forgetting reached 77.23% query and 95.16% action accuracy versus cosine-local 70.67/82.43%. Paired gains were +6.59 query points (95% CI `[5.43,7.79]`) and +12.73 action points (CI `[10.27,15.25]`). Hard-retain reached 96.12%, while primary-update reached 89.34%; the unchanged absolute gate therefore failed on overall action and update floors. The supported claim is statistically robust superiority over cosine-local on natural subject-disjoint sequences, not near-oracle reliability or native dense-writer efficacy. Test is now observed and may not be used for further method selection.
