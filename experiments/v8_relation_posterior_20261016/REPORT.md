# V8 relation-posterior controller

## Decision

V8 is an audited positive development result but failed its preregistered
release gate.  The posterior branch is closed and no held-out source is
opened.

## Result

| Domain | V7 gain | V8 gain | V8 oracle-gap closure |
|---|---:|---:|---:|
| TempLAMA development | +2.569 pp | **+4.214 pp** | **97.78%** |
| LAMA/ParaRel development | +0.859 pp | **+1.189 pp** | **61.71%** |

Every one of the three seeds improved both domains, added zero unrelated
errors, and had nonnegative semantic-overwrite and stale-repetition gains.
LAMA benefit average precision increased from 0.2460 to 0.3381 (+0.0921).

The run failed two prospective effect conditions: LAMA gain was below 1.5
points, and its improvement over V7 was +0.330 point rather than at least
+0.5.  The feature and training auditors reconstructed all features,
checkpoints, predictions, actions and gates and passed.

## Representation and resources

The frozen final Qwen3-4B checkpoint encoded 224 unique relation-slot texts.
Three train-only linear heads covered 25 relation IDs from 190 unambiguous
deduplicated texts; four ambiguous texts were excluded from head fitting.
Feature extraction peaked at 7.61/7.64 GiB allocated/reserved.  The 74-feature
controller trained in 29.26 seconds.

## Interpretation

Train-only posterior compatibility substantially improves the policy and
nearly closes TempLAMA headroom, but the 25-class posterior compresses away
detail needed for the LAMA effect-size target.  The registered next experiment
uses the frozen candidate/incoming vectors directly with counterfactual
benefit ranking and keeps the risk network isolated to the original 66
features.
