# V6 original-source LAMA/ParaRel confirmation protocol

## Freeze point and claim

This protocol is frozen after the independently audited V6 TempLAMA test
confirmation passed and before any episode from the seven held-out
LAMA/ParaRel confirmation relations is generated.  The TempLAMA test result
SHA256 is
`58ea580a19cbb585366483acf6ad1ba22016de837955fa0b8183d91d44a833df`.
No result from the original-source confirmation may change a controller
checkpoint, normalization statistic, threshold, feature, candidate rule,
physical rule, fact-key rule, prompt, episode size, endpoint, or gate.

The prospective hypothesis is that the unchanged TempLAMA-trained V5
residual controller plus the V6 semantic fact-key projection transfers back
to unseen LAMA/T-REx facts and unseen ParaRel relations while preserving
unrelated session records.  A pass supports cross-source selective forgetting
in the addressable Qwen residual memory.  It does not establish content
binding in the original dense `ttt_proj`/`ttt_conv` writer.

## Sealed source and corpus

Use only the seven confirmation relations fixed by the original
LAMA/ParaRel source inspection.  Their source rows have remained ungenerated
and unscored through V1--V6.  Reuse the registered V1 episode grammar:

- exactly 64 source groups, allocated as evenly as possible across the seven
  confirmation relations;
- two counterfactual twins per group;
- balanced lengths 8, 16, 32 and 64;
- three sparse decision boundaries per episode;
- five future queries per boundary; and
- the unchanged seven lifecycle mechanisms.

All input subjects remain unique across groups.  The builder must exclude the
previously removed cross-partition subjects and must not load a model output.
It records the pinned LAMA, ParaRel and source-inspection hashes.  An
independent exact reconstruction audit is required before model scoring.

## Frozen labels, features, and policy

Use the final 1B-token Qwen3-4B checkpoint and the audited addressable residual
memory.  Counterfactual labels enumerate the same full `{0, 0.5, 1}` action
grid and append every current write with `g=1`.  The safe label objective first
forbids a loss on unrelated queries, then maximizes future correctness,
minimizes answer NLL and retained memory mass.  Labeling runs once on two GPUs
and records every boundary's latency and CUDA peak memory.

Extract the unchanged 66-dimensional inference-safe V1/V2 feature schema.
The three frozen V5 checkpoints are seeds `26101651`, `26101652`, and
`26101653`; their stored means, standard deviations, selected epochs,
utility thresholds, risk penalties, and risk thresholds are immutable.
The raw subject-only action is reported as an ablation.  V6 accepts its one
possible residual deletion only when candidate and incoming records share the
immutable `(normalized subject, relation id)` key; otherwise it returns the
explicit lifecycle-physics action.  Relation IDs and subjects are used only
by this deterministic safety projection and never enter the neural tensor.

## Comparators and endpoints

Report retain-all, global clear, last-1, last-2, exact-string deletion,
exact-subject conflict, frozen cosine thresholds 0.98/0.99/0.995, explicit
lifecycle physics, all three raw subject-only controllers, all three V6
fact-key controllers, their majority policy, deletion-matched random controls,
and the safe single fact-key oracle.

The primary endpoint is pooled future-query accuracy.  Safety is the number of
new unrelated-query errors relative to retain-all.  Also report update,
retention and anchor accuracy, deletion precision/fraction, per-mechanism and
per-relation gains, answer NLL, latency, peak memory, and closure of the
physics-to-oracle gap.  Use 20,000 source-group bootstrap draws and a separate
relation-cluster bootstrap.  Every statistic is recomputed by an independent
result auditor.

## One-time confirmation gate

The original-source confirmation passes only if all conditions hold:

1. every frozen V6 seed improves over explicit lifecycle physics;
2. the arithmetic mean gain is at least two accuracy points;
3. both source-group and relation-cluster 95% gain interval lower bounds are
   above zero;
4. every seed adds zero unrelated-query errors;
5. mean closure of the safe fact-key oracle gap is at least 30%;
6. every mechanism has nonnegative mean gain;
7. all construction, label, feature, checkpoint, result, resource, hash and
   seal audits pass.

A pass freezes V6 for external LongMemEval-S, LoCoMo, RULER/RULER-QA and MMLU
evaluation.  A failure is reported without threshold repair or a second use of
these confirmation relations.  Method iteration after failure must use a new
training/development source and a new independent confirmation source.

## Model-free feasibility amendment

The first builder invocation stopped before writing a corpus or loading a
model: after the registered cross-partition subject filter, confirmation
relation P530 retained only seven facts from four subjects.  It cannot supply
globally unique target and background subjects for a 64-group corpus.  This is
a source-capacity failure, not an outcome, feature, label, or model result.

Before retrying construction, freeze the following deterministic eligibility
rule.  A confirmation relation must retain at least 128 unique normalized
subjects after the original cross-partition removal.  The threshold is twice
the registered group count and conservatively supports both target and
same-relation records while stable backgrounds remain group-unique.  Apply it
once to all seven registered confirmation relations.  This excludes P530 and
retains P937, P27, P136, P407, P106 and P495.  Allocate the unchanged 64
groups as evenly as possible across these six relations.  Require at least
four eligible relations, keep every other protocol element and gate unchanged,
and record all retained/excluded subject counts in the manifest and audit.

