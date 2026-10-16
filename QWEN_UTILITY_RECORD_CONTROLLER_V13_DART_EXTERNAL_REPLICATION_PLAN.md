# V13 frozen-ERM DART external replication

## Scientific question and frozen method

V11 showed that the natural counterfactual corpus, rather than relation-group
DRO, produced the selective-forgetting gain. Equal-source ERM improved over
lifecycle physics by 3.239 accuracy points with a relation-cluster 95% interval
of `[2.406, 4.137]` points and zero new unrelated-query errors. Removing the
FewRel corpus reduced the gain to approximately zero. V12 did not test the
method because WebNLG lacked enough disjoint factual pairs.

V13 therefore asks one question: do the three frozen V11 `erm_equal`
controllers transfer without retraining to a second natural structured-to-text
source? Their checkpoints, feature normalization, selected epochs, utility
thresholds, risk calibration and V6 fact-key safety projection remain frozen.
Current writes remain fixed at `g=1`; the learned decision controls only
forgetting of historical session memory represented by the audited Qwen
residual-memory substrate.

This experiment does not claim successful fact binding by the original dense
`ttt_proj`/`ttt_conv` writer. Its favorable single-record binding gate already
failed and it remains a negative baseline.

## Source registration and contamination boundary

The first candidate is the official DART structured-data-to-text corpus. Only
its training partition may be inspected. Records whose provenance is WebNLG,
FewRel, TempLAMA, LAMA or any previously used source are excluded. Before any
JSON record is parsed, acquisition must record the repository/data revision,
download URLs, relative paths, byte counts and SHA256 hashes.

The following sources remain sealed during V13: the eight FewRel internal-test
relations, sixteen FewRel external-test relations, DialogRE development/test,
SemEval test, LongMemEval-S and LoCoMo. No row, label, feature or result from a
sealed source may influence DART feasibility, corpus construction or the V13
decision.

## Stage A: exact model-free feasibility

A source item is eligible only if it provides a natural human reference and an
unambiguous aligned `(subject, relation, value)` fact. The normalized subject
and value must both be recoverable in the reference or in an official aligned
template. Source rows, normalized statements and entity pairs must be unique.

The feasibility program must run the complete deterministic material allocator,
not a coarse row-count proxy. It must demonstrate that exactly eight relations
can each support fourteen groups while satisfying all V10/V11 constraints:

- two groups for each of the seven lifecycle mechanisms;
- globally disjoint source rows and entity pairs;
- distinct target, four version, same-relation, pattern and stable-background
  donors required by the existing constructor;
- at least seven distinct natural templates per target relation; and
- zero overlap with every prior formal corpus.

The allocator must emit the proposed relation inventory and all donor hashes,
then an independent auditor must reconstruct the allocation. If eight
relations cannot be allocated exactly, V13 stops before model loading. Group
count, uniqueness, alignment and natural-reference rules cannot be weakened.

## Stage B: conditional one-time frozen evaluation

If Stage A passes, construct exactly 112 groups, 224 twin episodes, 672 sparse
boundaries and 3,360 future queries. Label the unchanged
`alpha in {0, 0.5, 1}` grid with the final 1B-token Qwen3-4B checkpoint and
extract the unchanged 66 inference-safe features. Apply the three frozen V11
ERM checkpoints once; V13 performs no training, checkpoint selection or
threshold search.

Report retain-all, global clear, last-1, last-2, lifecycle physics,
deletion-matched random, exact-statement and safe fact-key oracle baselines.
In addition to aggregate utility, report every relation and lifecycle mechanism,
the controller deletion rate, unrelated-query errors, safe-oracle closure,
latency and peak allocated/reserved GPU memory.

## Primary registered gate

V13 passes only if independent reconstruction verifies all of the following:

1. every frozen ERM seed improves over lifecycle physics;
2. the three-seed mean improvement is at least 1.5 accuracy points;
3. both source-group and relation-cluster bootstrap 95% lower bounds exceed
   zero;
4. at least six of eight relations have nonnegative mean gain;
5. every seed adds zero unrelated-query errors;
6. mean safe-oracle gap closure is at least 50%;
7. overwrite, stale-repetition and reset/revoke gains are nonnegative; and
8. all source, corpus, label, feature, checkpoint and result hashes reconstruct
   while all reserved sources remain sealed.

Failure is reported once without adapting to DART. It distinguishes source
specificity of the learned policy from the V12 capacity failure.

## Conditional confirmation sequence

A V13 pass freezes the complete ERM method and authorizes a separately
registered, one-time evaluation on the eight reserved FewRel internal-test
relations. Passing that internal test authorizes one final evaluation on the
sixteen external-test relations. No controller refit or threshold change is
allowed between these releases.

Only after both releases pass should the method enter long-context end-to-end
evaluation on RULER/QA, LongMemEval-S and LoCoMo. That phase compares no
forgetting, global decay, last-1/last-2, learned forgetting and safe-oracle
forgetting under matched writes and compute, and reports task quality, session
interference, latency and memory overhead.

## Decision table

| Outcome | Interpretation | Next action |
|---|---|---|
| DART capacity fails | Candidate source is too small or insufficiently aligned | Archive the model-free stop; register another natural source |
| Capacity passes, frozen ERM fails | V11 policy is source-specific | Stop the cross-source claim; analyze frozen errors without tuning |
| Frozen ERM passes | Selective forgetting transfers across natural sources | Open the reserved 8-relation internal test once |
| Both reserved releases pass | Evidence supports a general forgetting controller | Run end-to-end long-memory benchmarks and full efficiency ablations |

