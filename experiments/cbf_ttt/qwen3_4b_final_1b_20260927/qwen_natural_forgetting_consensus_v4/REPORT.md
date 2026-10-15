# Cross-fit on-policy consensus forgetting V4

## Result

V4 evaluated 17 registered learned policies on five-fold held-out calibration
trajectories. Every session used benefit/risk heads trained without that
session, and every policy maintained its own memory state. The independent
audit passed all nine checks. No candidate met the complete safety and
stability release, so the 129-session test remained unopened.

Retain-all reached 82.21%; the privileged source-action rule reached 94.28%.
The best candidate by useful accuracy was engineered plus no-schedule
intersection `x_n_b50_r200`: 86.22%, a 4.00-point gain whose paired
session-bootstrap 95% CI was [2.15, 5.85] points. It failed three checks: the
gain was below five points, it added 178 collateral errors, and outer fold 2
lost 4.33 points.

The most conservative intersection `x_n_b50_r050` added 142 errors but reached
only 81.73%, with CI crossing zero. Standalone no-schedule thresholds and seed
voting were unstable under policy-induced states; several lost 5--14 points
overall and up to 50 points in individual folds. No candidate was selected.

## Interpretation

Cross-view agreement and stricter thresholds cannot repair the distribution
shift by themselves. The original heads learned isolated retain-prefix
benefit/risk, whereas repeated actions create memory states outside that
training distribution. The registered failure branch is now active: build
labels from policy-induced calibration states and train a dedicated safe-
delete head before any test release.

The result is a scientific calibration stop rather than an execution failure.
All 5,144 query rows, state transitions, features, checkpoint probabilities,
actions, generations, hashes, aggregates, search results and bootstraps were
independently reconstructed. Development, test and external benchmarks were
not loaded.
