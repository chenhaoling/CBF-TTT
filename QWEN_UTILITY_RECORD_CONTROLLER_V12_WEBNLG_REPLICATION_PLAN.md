# V12 frozen-ERM WebNLG external replication

## Registered hypothesis

V11 provided positive relation-clustered evidence on eight unseen FewRel
relations, but its release failed because relation-group DRO did not improve
benefit AP over equal-source ERM.  A frozen diagnostic showed that both ERM
and DRO gained about 3.2 accuracy points, while removing the constructed
FewRel corpus reduced gain to 0.005 point.  The supported hypothesis is
therefore natural counterfactual corpus coverage, not robust loss.

V12 freezes the three V11 `erm_equal` checkpoints, their normalization,
checkpoint epochs, thresholds, risk calibration and V6 fact-key projection.
No controller is fit or selected.  It asks whether that policy transfers to a
new RDF-to-text source with unseen surface realizations.  Current writes remain
fixed at `g=1`; only historical session memory is controlled.

## New source and acquisition boundary

Use the English WebNLG enriched corpus from the public
`ThiagoCF05/webnlg` repository, version 1.6.  The acquisition script clones the
repository into a new directory and records repository commit, relative file
paths, byte counts and SHA256 hashes before any XML element is parsed.  The
exact commit and file inventory will be appended to this plan after acquisition
and before the first schema/capacity inspection.

FewRel internal/external tests, DialogRE development/test, SemEval test,
LongMemEval-S and LoCoMo remain sealed.  No WebNLG row, relation, lexicalization
or model output may alter the V11 checkpoints or deployment configuration.

## Model-free feasibility stage

Inspect English WebNLG training entries only.  A candidate source fact must:

1. contain exactly one modified RDF triple `(subject, relation, value)`;
2. have a human lexicalization whose text contains unambiguous subject and
   value mentions, or an official aligned template with one subject and value
   slot;
3. produce a nonempty natural `[X]/[Y]` template;
4. have distinct normalized subject and value strings; and
5. be unique by source entry, normalized statement and entity pair.

Report capacity per relation without sampling.  A relation is eligible only
with at least 28 rows, 20 subjects, 20 values and seven distinct natural
templates.  At least eight relations must be eligible.  Failure stops V12
before corpus construction or model execution and motivates a new source; the
thresholds or single-triple rule cannot be relaxed after seeing capacity.

## Conditional corpus and one-time evaluation

If feasibility passes, select exactly eight relations by deterministic SHA256
rank under seed `202610201`.  Build 14 groups per relation, two per lifecycle
mechanism, using the unchanged V10/V11 natural-template construction.  Require
112 groups, 224 twins, 672 sparse boundaries, 3,360 future queries, globally
unique material rows/entity pairs and no overlap with any prior corpus.

Label the unchanged `{0, 0.5, 1}` counterfactual grid on the final 1B-token
Qwen3-4B checkpoint and extract the unchanged 66 inference-safe features.
Apply the three frozen V11 ERM checkpoints once.  Baselines are retain-all,
global clear, last-1/last-2, exact statement, explicit lifecycle physics,
deletion-matched random controls and the safe fact-key oracle.

## One-time replication gate

V12 passes only if an independent audit verifies:

1. every frozen ERM seed improves over lifecycle physics;
2. mean gain is at least 1.5 accuracy points;
3. source-group and relation-cluster bootstrap 95% lower bounds exceed zero;
4. every seed adds zero unrelated-query errors;
5. mean safe-oracle gap closure is at least 50%;
6. overwrite, stale repetition and reset/revoke gains are nonnegative;
7. all source, corpus, label, feature, checkpoint and result hashes reconstruct;
   and
8. every previously sealed source remains unopened.

A pass supports a cross-source claim for the simple corpus-trained forgetting
controller and authorizes separately registered LongMemEval-S/LoCoMo
evaluation.  A failure is reported once without adapting to WebNLG.
