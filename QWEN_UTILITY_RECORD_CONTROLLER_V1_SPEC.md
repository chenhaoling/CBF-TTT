# Qwen utility record controller V1: frozen implementation specification

## Freeze boundary

This specification is frozen after the independently audited resource pilot
and formal model-free corpus audit, while the formal Qwen counterfactual label
job is still running and before its summary is read. It instantiates the
controller section of `QWEN_UTILITY_SUPERVISED_SEMANTIC_RECORD_GATE_V1_PLAN.md`.
Formal-label outcomes, development relations and confirmation relations cannot
change the feature set, target definitions, architecture, loss, checkpoint
rule or finite threshold grid below.

The final 1B-token Qwen3-4B checkpoint remains frozen. The controller decides
only whether each addressable historical session record is kept or deleted.
The incoming write is always admitted (`g=1`). The deployable primary action
space is binary because the sealed resource pilot found uniquely useful
half-strength labels on only 2.08% of boundaries. The complete ternary labels
remain a registered soft-decay ablation.

## Binary teacher and risk target

For each boundary, reconstruct every endpoint action whose coefficients are in
`{0,1}`, plus global clear. Apply the same rowwise collateral barrier used by
the full oracle: an eligible action must have unrelated-chain correctness no
lower than retain-all. Select the binary teacher lexicographically by:

1. future correct count;
2. negative total correct-answer NLL;
3. negative retained memory mass;
4. larger retained coefficient sum;
5. stable action name.

For each candidate record, the keep/delete target is the corresponding teacher
coefficient. The risk target is one when deleting only that record from
retain-all reduces unrelated-chain correctness; otherwise it is zero. The
utility margin is the teacher score difference from the best eligible action
that flips that record, and is used only as a bounded sample weight. Future
queries, outcomes, relation IDs, answers and teacher actions never enter an
inference feature.

## Inference-safe per-record features

All text features come from the frozen plain-Qwen checkpoint. Mean-pooled
hidden vectors are L2-normalized. For each historical candidate and incoming
record, compute the following fixed scalar features:

1. exact normalized-subject equality;
2. exact normalized-value equality;
3. subject-equal/value-different conflict;
4. exact normalized-statement equality;
5. cosine of relation-masked statements (`[SUBJECT]`, `[VALUE]`);
6. cosine of full statements;
7. cosine after masking only the subject;
8. cosine after masking only the value;
9. candidate relation-cosine minus the best competing candidate cosine;
10. candidate relation-cosine rank divided by candidate count;
11. length-normalized frozen-Qwen `yes` minus `no` log-likelihood for the
    fixed prompt `Do these two session facts update the same property?`;
12. record age divided by 63 and clipped to one;
13. recency rank divided by current memory size;
14. number of earlier exact-statement repetitions divided by current memory
    size;
15. number of earlier same-subject records divided by current memory size;
16. number of earlier same-subject/same-value records divided by current
    memory size;
17. current memory size divided by 64 and clipped to one;
18. previous observable query accesses to the normalized subject divided by
    three and clipped to one;
19. selector membership;
20. current record coefficient;
21. explicit reset indicator;
22. explicit revoke-target indicator.

The pair prompt contains only the two statements and requests the literal
answer `yes` or `no`; no generated rationale is used. Pair log likelihood is
computed under the frozen slow model without memory residuals. Relation IDs,
source IDs, split names, mechanism names and audit semantic keys are prohibited
from the feature tensor.

For each scalar feature, append the mean and maximum across the candidates at
that boundary. This produces a permutation-equivariant record tensor. Missing
competitors or accesses map to zero. Feature normalization statistics are fit
on controller-training relations only and are hash-locked.

## Controller and physical projection

A shared MLP maps each augmented record row through widths 128 and 64 with
GELU and dropout 0.1. It has a two-class keep/delete utility head and one
collateral-risk logit. The same weights are applied to every record; no
position-specific head is allowed.

The physical projection is fixed:

- the incoming record is retained;
- an exact normalized-statement duplicate is idempotent and removes the older
  duplicate;
- an explicit reset clears historical records;
- an explicit revoke deletes only the referenced historical record;
- otherwise a candidate can be deleted only when its delete probability meets
  the selected utility threshold and its risk upper bound is below the selected
  conformal risk level;
- rejected or uncertain deletions retain the record.

Projection is applied after the neural logits. A no-projection controller and
the explicit-physics-only policy are registered comparisons.

## Training and selection

Relation IDs may be used only by the offline split auditor. Hash the 17 formal
training relations into five folds. For seeds 2610161, 2610162 and 2610163,
rotate a relation-disjoint inner validation fold, choose only among 25, 50 and
100 epochs by joint utility accuracy subject to zero extra collateral errors,
then retrain on all formal training relations for the selected epoch count.

Use AdamW with learning rate `1e-3`, weight decay `1e-4`, batch size 256 record
labels and gradient norm one. Optimize weighted utility cross entropy plus
unit-weight binary risk cross entropy. Clip the utility-margin sample weight to
`[1,3]`. Class weights are inverse-frequency values computed on the current
training fold. No outcome-derived resampling is allowed.

Only the relation-disjoint development partition can select deployment values
from utility thresholds `{0.4,0.5,0.6}`, risk penalties `{0,0.5,1,2}` and
conformal risk levels `{0.01,0.03,0.05}`. The checkpoint, feature extractor and
grid are immutable after any development result. The sealed confirmation
partition is opened once only if all development gates in the parent protocol
pass.

## Required audits

Before training, an independent auditor must prove:

- exact reconstruction of every safe binary teacher and per-record risk
  target from raw action outcomes;
- one-to-one coverage of all formal boundary candidates;
- absence of future outcomes, answers, relation IDs and audit identifiers from
  tensors;
- relation-disjoint folds and train-only normalization statistics;
- frozen checkpoint, tokenizer and source hashes;
- finite features and deterministic reconstruction;
- development, confirmation and external benchmark seals.

Training artifacts must retain per-seed checkpoints, fold assignments, class
weights, loss curves, gradient norms, selection decisions, hashes, runtime and
peak memory. A failure of any audit stops before development generation.
