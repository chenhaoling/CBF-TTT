# Cross-fit on-policy consensus forgetting V4 preregistration

## Motivation and held-out boundary

Factor Gate V2 passed isolated-intervention calibration and V3 produced a
large on-policy development gain: 93.32% versus 84.82% retain-all, paired 95%
CI [7.48, 9.41] points.  V3 nevertheless stopped because it added 113
collateral errors, while the privileged source-action rule added none.  Every
added error was an active record physically deleted by the policy.  The
registered no-schedule ablation reached 93.54% with 27 added errors, made no
wrong-top local deletion, and motivates a safety-focused candidate family.

V3 development may be used to define V4, but it may not select a V4 threshold
or count as confirmation.  V4 selection uses only cross-fit calibration
trajectories.  The separate source test file, containing 129 sessions and 516
subjects, remains unopened until the complete calibration release passes.
External benchmarks remain unopened.

## Cross-fit controller reconstruction

Rebuild one Factor Gate V2 checkpoint for every outer fold and each of seeds
9401, 9402 and 9403, separately for the registered `engineered` and
`no_schedule` views.  For fold `k`, fit only on sessions whose frozen V2 fold
is not `k`, with the configuration and thresholds selected by that fold's
nested inner search.  Normalization is fit on the same outer-training rows.

The reconstruction must reproduce every saved V2 OOF benefit/risk probability
and action before any online calibration trajectory is scored.  Save model
states, means, standard deviations, thresholds, training session IDs, source
hashes and the exact maximum reproduction error.  An independent audit must
verify exclusion, folds, checkpoints, probabilities and hashes.

## Registered on-policy candidates

Each calibration session uses only its held-out-fold checkpoints.  Every arm
maintains independent memory.  At each event it computes the two view
ensembles on its current state, decides, deletes at most the inferred scope
top, appends the incoming record and continues from that state.

Controls are retain-all, always scope-top and the privileged source-action
rule.  Registered learned candidates are:

- frozen engineered V2: `E(b=.8,r=.1)`;
- frozen no-schedule V2: `N(b=.5,r=.2)`;
- the 3x3 no-schedule grid
  `b in {.5,.7,.8}`, `r in {.05,.1,.2}`;
- cross-view intersection `E(b=.8,r=.1) AND N(b,r)` for
  `N(.5,.05)`, `N(.5,.1)`, `N(.5,.2)`, `N(.7,.1)` and `N(.8,.1)`;
- no-schedule majority and unanimity, where each seed votes using
  `N(b=.5,r=.2)` and at least two or all three votes are required.

This is a finite family fixed before online calibration outcomes.  It may not
be enlarged after seeing calibration.  Candidate features remain inference
visible; event kind, source action, chain ID, answer and future queries are
never controller inputs.

## Online calibration scoring and selection

Replay all 65 calibration sessions and score every exact/rephrased active
query at every registered checkpoint with the frozen V5 reader and Qwen3-4B
value bridge.  Record state transitions, probabilities, active-record
deletions, stale/wrong/defer reads, candidate NLL, final fixed-length
generation, runtime and memory.  Use paired retain-all accounting for added,
recovered and collateral errors.

Use 20,000 whole-session bootstrap draws with seed 202610261.  A candidate is
safe-feasible only if:

- it improves candidate accuracy over retain-all by at least five points;
- the paired 95% gain lower bound is above zero;
- it adds at most four collateral errors over the complete calibration
  trajectory, matching the previously registered source-action safety budget;
- every outer fold has non-negative candidate-accuracy gain;
- delete rate is between 5% and 50%;
- final generation exact is no lower than retain-all by more than 0.5 points.

Among feasible candidates select maximum candidate accuracy, then fewer
collateral errors, higher recovered-to-added ratio, lower delete rate and
lexicographic candidate name.  If no candidate is feasible, keep test sealed
and train a new risk head from counterfactual labels collected on policy-
induced calibration states.  Do not relax the gate.

## One-time test confirmation

If calibration and its independent audit pass, freeze the selected rule and
all final all-calibration V2 model weights.  Only then hash and materialize the
129-session natural test corpus from the pre-existing source `test.jsonl`.
Candidate construction is deterministic and model free, following the V1
natural corpus exactly.  No test outcome may select a parameter.

Evaluate the selected policy, retain-all, always scope-top, source-action rule,
the frozen engineered and no-schedule V2 policies, and benefit-only.  The
primary test passes only if:

- learned minus retain-all accuracy is at least five points and its paired
  session-bootstrap 95% lower bound is above zero;
- learned accuracy exceeds both frozen V2 view baselines;
- collateral added-error rate is at most 0.1% and no greater than twice the
  source-action count plus two;
- recovered errors are at least ten times added errors;
- final generation exact is no lower than retain-all;
- exact and rephrase accuracy both exceed retain-all;
- partial- and full-width session gains are both positive;
- independent audits verify corpus construction, controller isolation, every
  trajectory, score, aggregate, bootstrap, hash and held-out seal.

A test pass supports risk-aware learned selective forgetting in the frozen
Qwen-derived readable-memory substrate.  Native dense `ttt_proj`/`ttt_conv`
deletion and external long-memory generalization remain separate claims.
