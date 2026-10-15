# Cross-fit safe-delete DAgger V6 preregistration

## Motivation and boundary

V5 learned a lifecycle safe-delete target from 2,213 V4 policy-induced states.
Its audited five-fold OOF result reached 98.90% precision and 81.49% recall,
with six false-positive deletes.  It failed only the preregistered absolute
false-positive budget of four.  All six false-positive rows came from three
hard-retain events in which the incoming and retrieved prompts expressed the
same relation for different subjects.

V5 preregistered one DAgger iteration on failure.  V6 executes that branch
without changing features, model family, search grid, selection criteria or
release gate.  It uses only calibration.  V3 development, the 129-session
source test and external benchmarks remain sealed.

## Cross-fit behavior rollout

For each of the 65 calibration sessions, use its frozen V5 outer-fold
ensemble.  That ensemble was trained without the session.  Starting from an
empty memory, compute the registered ten `no_schedule` features, delete the
reader top only when the V5 ensemble probability meets that fold's frozen
threshold, append the incoming record and continue from the induced state.

Record the ordered `(chain_id, version)` memory, inferred top, ten features,
three-seed mean probability, frozen threshold, action and physical
safe-delete label at every event.  Source action and chain identity construct
the label and audit state transitions but are never model inputs.  Two GPU
shards must use the same frozen Qwen3-4B checkpoint, reader, transforms and
scope artifact as V4.

## Merge and repeated grouped training

Merge the 774 DAgger states with the 2,213 audited V5 corpus rows and
deduplicate by the existing state ID, which hashes session, event step and
ordered memory records.  Overlapping rows must agree exactly on features,
reader top and physical label.  Preserve session folds.

Repeat V5's complete nested grouped protocol unchanged: hidden size 0/16/32,
learning rate 0.001/0.005, weight decay 0/0.01, thresholds
0.50/0.70/0.80/0.90/0.95/0.98/0.99, three-seed ensembles, 300 epochs,
positive-class weighting and four-fold inner OOF selection inside every outer
fold.  No threshold or feature may be added after observing V6 outcomes.

## Release gate and next branch

The combined outer OOF result passes only with precision at least 98%, recall
at least 50%, no more than four false-positive deletes, at least 95% precision
in every active outer fold, delete rate between 5% and 50%, exact fold
isolation, independent audit success and all held-out sources sealed.

If V6 passes, run a new cross-fit on-policy outcome replay with retain-all,
source action, V4 best and V6.  Test remains sealed until that replay gains at
least five accuracy points with positive paired CI lower bound, at most four
collateral errors, nonnegative gains in all folds and preserved generation.

If V6 fails, do not repeat DAgger or tune a threshold.  Preregister an
entity-key V7 head motivated by the already localized same-relation,
different-subject failure.  The new feature must be computed from prompt text
at inference; source subject fields, chain IDs and event actions remain
forbidden inputs.
