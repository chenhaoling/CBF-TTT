# V10 three-source failure diagnostic

## Scope

V10 passed every integrity and safety audit but failed its preregistered effect
gate.  FewRel removal reduced LAMA gain by 0.165 point, just below the fixed
0.2-point condition, while the primary policy remained 0.043 point below V8
and lost 1.38 TempLAMA points.  This diagnostic is descriptive and cannot
release V10 or authorize another search on the opened development domains.

## Frozen analyses

Using only the six saved V10 checkpoints and already opened development rows:

1. reconstruct benefit/risk AUROC, average precision, Brier score and ECE for
   both arms, every seed and both domains;
2. compare primary versus FewRel-removal probabilities and final fact-key
   projected actions, including changed-boundary count, favorable/unfavorable
   outcome changes and mechanism slices;
3. measure standardized centroid distances among the three training sources
   and assign development examples to their nearest training centroid; and
4. report target prevalence by source and mechanism.

No checkpoint, threshold, feature, source weight or policy configuration is
fit or selected.  TempLAMA test, LAMA confirmation, LongMemEval-S and LoCoMo
remain sealed.

## Interpretation

- Little probability/action change indicates weak causal uptake of FewRel.
- Better ranking with few action changes indicates policy discretization.
- Many harmful action changes indicate supervision mismatch.
- Poor ranking plus strong source separation indicates representation shift.

The result determines whether the constructed-corpus branch closes as a small
positive result or whether a new study must use a new validation protocol.  It
cannot justify another iteration on the same two development outcomes.
