# V14 frozen-ERM REBEL external replication

## Registered question

V11 supports natural-counterfactual corpus coverage and rejects the proposed
relation-group DRO improvement. V12 WebNLG and V13 DART both stopped before
model execution because their eligible relation capacities were insufficient.
V14 retains the complete frozen V11 equal-source ERM method and asks whether it
transfers, without retraining, to natural Wikipedia sentences aligned with
repeated Wikidata relations in the official REBEL corpus.

The three V11 `erm_equal` checkpoints, feature normalization, epochs, utility
and risk thresholds, calibration and V6 fact-key safety projection are frozen.
Current writes remain fixed at `g=1`; only historical session-memory forgetting
is controlled. V14 cannot change the native `ttt_proj`/`ttt_conv` writer result,
which remains a failed content-binding baseline.

## Source and evidence boundary

Use the official `Babelscape/rebel-dataset` release. Its contexts are natural
English Wikipedia sentences; subject-relation-object annotations were produced
automatically from Wikipedia/Wikidata and filtered by NLI. This makes V14 a
large-scale natural-text, silver-relation replication. The automatic-label
limitation must be reported and cannot be described as human fact annotation.

Before any example or Parquet row is parsed, acquisition records the resolved
Hugging Face dataset revision, downloaded relative paths, byte counts and
SHA256 hashes. Only the official training split may be downloaded and parsed.
Validation/test splits, the eight FewRel internal-test relations, sixteen FewRel
external-test relations, DialogRE development/test, SemEval test,
LongMemEval-S and LoCoMo remain sealed.

## Frozen model-free filter and allocation

Use seed `202610221`. Accept a training example only when the official
linearization contains exactly one complete triplet with nonempty subject,
object and relation, and the natural context contains exactly one
case-insensitive literal occurrence of both subject and object. Reconstruct one
`[X]/[Y]` template without semantic or model-based matching. Remove duplicate
`(relation, subject, object)` facts and every normalized statement/template
overlap with prior formal corpora.

Candidate target relations require at least 140 distinct entity pairs and seven
distinct natural templates. Rank them by SHA256 under the frozen seed and take
the first eight; do not search alternative subsets after seeing allocation.
Relations with at least five pairs and four templates may be background donors,
excluding the eight selected targets.

Run the complete deterministic V10/V11 material allocator before loading a
model. It must allocate eight relations × fourteen groups while preserving
globally unique source rows/entity pairs and all target, version,
same-relation, pattern, stable-background and anchor donors. A second process
must reconstruct the source filter, ranked relation set, allocation and hashes.
Failure stops V14 without reducing scale, reusing pairs or relaxing alignment.

## Conditional frozen-controller evaluation

If capacity passes, construct 112 groups, 224 twin episodes, 672 sparse
boundaries and 3,360 future queries. Label the unchanged
`alpha in {0, 0.5, 1}` grid with the final 1B-token Qwen3-4B checkpoint, extract
the unchanged 66 inference-safe features and apply the three frozen V11 ERM
checkpoints once. No V14 label can train a controller, select an epoch or tune a
threshold.

Compare retain-all, global clear, last-1, last-2, lifecycle physics,
deletion-matched random, exact-statement and safe fact-key oracle baselines.
Report aggregate, per-relation and per-mechanism quality, deletion rate,
unrelated-query errors, oracle closure, latency and peak GPU memory.

## Primary one-time gate

V14 passes only if all audits verify:

1. every frozen ERM seed improves over lifecycle physics;
2. mean improvement is at least 1.5 accuracy points;
3. source-group and relation-cluster bootstrap 95% lower bounds exceed zero;
4. at least six of eight relations have nonnegative mean gain;
5. every seed adds zero unrelated-query errors;
6. mean safe-oracle gap closure is at least 50%;
7. overwrite, stale-repetition and reset/revoke gains are nonnegative; and
8. all source, corpus, label, feature, checkpoint and result hashes reconstruct
   while reserved sources remain sealed.

A capacity pass followed by a policy failure is reported once without adapting
to REBEL. A policy pass freezes the complete method and authorizes a separately
registered one-time evaluation on the eight reserved FewRel internal-test
relations, followed conditionally by the sixteen external-test relations. Only
after both releases pass may RULER/QA, LongMemEval-S and LoCoMo be opened for
end-to-end long-memory evaluation.
