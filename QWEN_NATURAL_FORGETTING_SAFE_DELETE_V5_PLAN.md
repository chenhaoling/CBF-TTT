# On-policy lifecycle safe-delete head V5 preregistration

## Question and held-out boundary

V4 showed that thresholding or intersecting isolated-intervention utility
heads does not control errors under policy-induced memory states.  Its best
useful candidate improved calibration accuracy by 4.00 points but added 178
collateral errors and was negative in one outer fold.  V5 therefore asks a
different, mechanistic question: can inference-visible state features predict
whether deleting the currently retrieved memory record is consistent with the
fact lifecycle?

V5 uses only the 65 V4 calibration sessions.  It must not load V3 development,
the 129-session source test, or any external benchmark.  Model and threshold
selection are grouped by session and nested inside the existing five outer
folds.  Passing offline cross-fitting authorizes a new cross-fit on-policy
calibration replay; it does not authorize test by itself.

## Policy-induced corpus and physical target

Use every registered V4 behavior arm, including controls.  Reconstruct each
arm's ordered memory before every write.  Track records by `(chain_id,
version)`, where version is the zero-based occurrence of that chain in the
session event stream.  Deduplicate rows by `(session_id, event_step, ordered
memory records)`.  Duplicates must agree on features, inferred top and label.

The privileged training target is one exactly when all conditions hold:

1. the source event is a local lifecycle update;
2. memory is nonempty and the reader supplies a top record; and
3. the top record has the same chain as the incoming update.

Otherwise the target is zero.  This target says whether the only executable
operation, deleting the inferred top record, is physically valid.  Event
kind, source action, chain ID, version, answer, future query outcome and arm
name are metadata or labels only and may never enter the model input.

Use the frozen ten-dimensional `no_schedule` feature view from V2/V3:
`has_memory`, `top_cosine`, `cosine_margin`, `cosine_mean`, `cosine_std`,
`cosine_entropy`, `v5_probability`, `lexical_jaccard`,
`lexical_containment`, and `length_ratio`.  Exclude `memory_size`,
`event_position`, and `top_recency` because V3 localized schedule-dependent
false deletion.  No Qwen forward pass is required to build this corpus: all
features were frozen and recorded before V4 actions.

## Nested grouped training

Preserve `fold_for(session_id)` from Barrier-Q V1.  For each outer fold, use
only the other four folds for selection and fitting.  Evaluate the Cartesian
product of:

- linear, hidden-16 and hidden-32 binary heads;
- AdamW learning rate 0.001 or 0.005;
- weight decay 0 or 0.01;
- probability threshold 0.50, 0.70, 0.80, 0.90, 0.95, 0.98 or 0.99;
- seeds 9401, 9402 and 9403, averaged as an ensemble.

Fit for 300 full-batch epochs with training-fold normalization and
positive-class weighting.  Select configurations exclusively from four-fold
inner OOF predictions.  A selection is inner-feasible only if precision is at
least 98%, recall at least 50%, false-positive deletes at most four, and every
inner fold with predicted deletes has precision at least 95%.  Among feasible
rows prefer higher recall, fewer false positives, higher precision, the
simpler head and higher threshold.  If none is feasible, use a deterministic
safety-first fallback for diagnosis; the outer release gate remains binding.

Save the deduplicated corpus, exact feature and label tensors, per-fold model
states and normalization, inner searches, every OOF probability/action,
session lists, source hashes and runtime.  An independent auditor must replay
memory state construction and labels, verify feature isolation, fold
exclusion, checkpoint predictions, metrics and hashes.

## Offline release gate

The combined outer OOF policy passes only if all checks hold:

- precision is at least 98%;
- recall is at least 50%;
- at most four false-positive deletes occur over all 2,213 states;
- every outer fold with predicted deletes has precision at least 95%;
- delete rate is between 5% and 50%;
- every row is predicted by models trained without its session;
- V3 development, source test and external benchmarks remain unloaded; and
- the independent execution audit passes.

If this gate fails, do not search extra thresholds and do not open test.  Run
one registered DAgger iteration instead: collect states induced by the frozen
cross-fit V5 policy, merge and deduplicate them with this corpus, and repeat
the same grouped protocol.

## Authorized next stage and final confirmation

If the offline gate and audit pass, replay the frozen fold-specific V5 policy
on the 65 calibration sessions.  Compare retain-all, fixed deletion, source
action, V2 Factor Gate, V4 best registered intersection and V5.  The on-policy
release requires at least a five-point accuracy gain over retain-all, paired
session-bootstrap 95% lower bound above zero, at most four collateral errors,
nonnegative gain in all five outer folds, 5--50% deletion rate and no final
generation degradation beyond 0.5 points.

Only after that result and its audit pass may an all-calibration model be
frozen and the existing 129-session test be opened once.  The claim remains
selective forgetting in the readable session-memory substrate; native dense
`ttt_proj`/`ttt_conv` deletion and external generalization are separate
experiments.
