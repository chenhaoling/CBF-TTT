# Grouped OOF Barrier-Q forgetting controller V1

## Motivation and released calibration evidence

The audited formal resource contains 774 events from 65 calibration sessions.
Its safe oracle improves future candidate accuracy from 80.8691% to 83.6438%
(gain 2.7747 points; session-bootstrap 95% CI [2.5423, 3.0115]) with no
additional unrelated error.  Always applying `scope_top1_a0` reaches 81.5647%
but adds 855 unrelated errors.  A non-deployable rule using the source event
type reaches 82.0791% and adds four errors.  Thus the learnable problem is not
whether local deletion can help; it is whether observable state can identify a
high-utility action while controlling interference.

This protocol trains and selects only on calibration with grouped
out-of-fold (OOF) predictions.  It does not load natural development, test or
an external benchmark.

## Observable features

Reconstruct the retain-all memory immediately before every calibration event.
Use only variables available at inference:

- the frozen contextual/pair feature for the incoming prompt against the
  frozen key scorer's top memory record;
- frozen cosine top score, top-two margin, score mean/standard deviation and
  softmax entropy;
- frozen V5 retrieve probability;
- memory size, normalized event position, top-record recency and a
  has-memory indicator;
- case-folded token Jaccard, containment and length ratio between the incoming
  prompt and the top memory key.

For empty memory use a registered zero representation plus the has-memory
indicator.  Do not expose source event kind/action, chain IDs, target values,
which affected record shares a chain, future outcomes or oracle labels as
features.  Frozen feature extraction may see all calibration text because it
does not fit a parameter or use an outcome; normalization is fit separately
inside every training fold.

## Counterfactual targets

For every state and each of the 11 unique actions derive three targets from
the audited outcomes:

1. `gain`: action future candidate accuracy minus retain-all accuracy;
2. `nll_benefit`: retain-all mean answer NLL minus action mean answer NLL;
3. `risk`: whether the action adds at least one unrelated candidate error
   relative to retain-all.

Retain-all has zero gain and zero risk by construction.  Training on all
11 outcomes turns 774 decisions into 8,514 action-supervision tuples and does
not discard rare safe-oracle classes.

## Models and policy

The primary `barrier_q` model shares a state encoder and emits per-action gain,
NLL-benefit and risk logits.  Optimize Huber loss for the two continuous heads
and weighted binary cross entropy for risk.  At inference:

1. discard actions whose predicted risk exceeds the selected barrier
   probability;
2. discard non-retain actions below the selected minimum predicted gain;
3. find the largest predicted gain among remaining actions;
4. among actions within the selected gain-tie tolerance, choose the largest
   predicted NLL benefit, then the lower known immediate memory mass and fixed
   action order;
5. fall back to retain-all when no intervention remains.

Train two ablations on the identical features and folds:

- `gain_q`: the same value model without the risk barrier;
- `direct_classification`: predict the safe-oracle action with class-balanced
  cross entropy and a retain confidence fallback.

Report fixed retain, every fixed action, frozen-scope delete-if-retrieve,
rate-matched random forgetting, the non-deployable source-action rule and both
counterfactual oracles.  The source-action rule is an analysis upper control,
not a deployable baseline.

## Nested grouped protocol

Assign sessions to five deterministic outer folds by SHA256 of
`barrier-q-v1|session_id`.  For each outer fold, use only the other sessions in
four deterministic inner folds to choose hyperparameters and policy
thresholds.  No label or outcome from an outer session may affect its model,
normalization or threshold.

Registered model grid:

- hidden width `{0, 32, 64}` where zero is linear;
- learning rate `{0.001, 0.005}`;
- risk-loss weight `{1, 3}`;
- 100 full-batch epochs with AdamW and zero weight decay;
- training seeds `{9401, 9402, 9403}`.

Registered policy grid:

- barrier probability `{0.05, 0.10, 0.20, 0.30}`;
- minimum predicted gain `{0, 0.005, 0.01, 0.02}`;
- gain-tie tolerance `{0, 0.005, 0.01}`.

On inner OOF predictions, first retain configurations whose unrelated-error
count is no greater than the non-deployable source-action rule on the same
rows and whose candidate accuracy exceeds retain-all by at least 0.5 points.
Among them maximize candidate accuracy, then minimize unrelated errors, model
width and distance from the conservative thresholds `(0.10, 0.01, 0.005)`.
If none is feasible, select minimum unrelated errors, then maximum accuracy;
that outer fold is marked infeasible.  Train the selected configuration on all
outer-training sessions and emit predictions only for the held-out sessions.

After OOF evaluation, choose the final all-calibration configuration by the
median outer selections, train one checkpoint per seed and freeze them.  This
checkpoint is not a development result.

## OOF metrics and release gate

Use the stored action outcomes for exact offline policy evaluation.  Aggregate
candidate accuracy, NLL, unrelated errors, action distribution, safe-oracle
regret and fraction of oracle headroom captured.  Use 20,000 deterministic
session bootstrap draws for paired accuracy gains.

Release the controller to one-time natural-development replay only if all
checks pass for the three-seed mean and every seed has positive direction:

- Barrier-Q candidate accuracy exceeds retain-all by at least 0.75 points;
- Barrier-Q versus retain-all session-bootstrap 95% lower bound is above zero;
- Barrier-Q exceeds the strongest fixed deployable action by at least 0.25
  points, with paired 95% lower bound above zero;
- Barrier-Q captures at least 30% of safe-oracle headroom;
- Barrier-Q adds no more unrelated errors than the source-action analysis
  control;
- Barrier-Q has fewer unrelated errors than `gain_q` while losing no more than
  0.5 accuracy points to it;
- at least four of five outer folds satisfy the registered inner feasibility
  constraints;
- all three seeds improve over retain-all and agree on the sign of the
  fixed-baseline comparison;
- independent audit verifies folds, features, targets, OOF exclusivity,
  outcomes, bootstraps, checkpoints and held-out seals.

Passing this calibration gate authorizes a separately preregistered one-time
online development rollout.  Failure keeps development sealed and triggers
calibration-only diagnosis or on-policy data aggregation.
