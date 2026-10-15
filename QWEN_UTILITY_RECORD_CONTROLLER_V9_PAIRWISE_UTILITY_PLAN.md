# V9 direct pairwise counterfactual-utility controller

## Registered cause and hypothesis

V8's audited relation-posterior representation improved the TempLAMA policy to
+4.214 points and the LAMA/ParaRel policy to +1.189 points, with zero
collateral in every seed/domain.  It closed 97.78% and 61.71% of safe
fact-key-oracle headroom and raised LAMA benefit average precision from 0.2460
to 0.3381.  It nevertheless failed the preregistered 1.5-point LAMA effect
gate and improved LAMA by only 0.330 point over V7 instead of the required
0.5.  The posterior branch is closed.

The registered interpretation is that compressing a candidate/incoming pair
through 25 train-relation posterior classes discards utility-relevant detail.
V9 learns a compact pair interaction directly from the audited frozen-Qwen
relation-slot vectors under downstream counterfactual benefit supervision.
It also adds a within-boundary ranking loss, matching the deployed action that
selects at most one extra deletion.  The risk branch receives only the locked
66 inference-safe scalar features, so pair representation cannot degrade the
already strong collateral detector.

## Immutable evidence boundary

Use exactly the audited V7/V8 train and development sources and the V8 frozen
embedding artifact:

- 4,392 TempLAMA and 4,684 LAMA/ParaRel training residual candidates;
- 2,204 TempLAMA and 588 LAMA/ParaRel development candidates;
- 224 ordered normalized relation-slot embeddings produced by the final
  1B-token Qwen3-4B checkpoint with native TTT disabled; and
- the original V7 66-dimensional features, counterfactual benefit/risk labels
  and V6 exact fact-key safety projection.

Hash-lock the V8 embedding and feature audits.  Do not load V8 development
outcomes as input features.  TempLAMA test, the original-source confirmation,
LongMemEval-S, LoCoMo and every earlier stopped semantic source remain sealed.

## Fixed architecture

For candidate vector `c` and incoming vector `i`, apply one shared trainable
linear projection `P: 2560 -> 128` without bias.  Construct

```text
[P(c), P(i), abs(P(c)-P(i)), P(c)*P(i)]
```

and concatenate it with a 64-dimensional encoding of the normalized base
66-vector.  A 128-unit GELU/dropout fusion layer produces the benefit logit.
The risk logit is produced by an independent 128/64 GELU/dropout network that
sees only the normalized base 66-vector.  Relation IDs, posterior classes,
counterfactual outcomes and source IDs are absent at inference.

Train with the same seeds, AdamW settings, batch scale, dropout, clipping and
checkpoint epochs as V7/V8.  The objective is weighted benefit BCE plus
weighted risk BCE plus `0.25` times a within-boundary pairwise logistic loss.
For every training boundary containing both classes, average
`softplus(-(positive benefit logit - negative benefit logit))` over all
positive/negative pairs.  Batches preserve complete boundaries.  Every
example contributes once per epoch; the two domains remain naturally near
equal in residual-candidate mass.

## Fixed ablations

Run all three seeds for the primary pair-plus-ranking model.  For causal
attribution, run these development-only ablations with the same seeds and
selection grid:

1. pair representation with ranking coefficient zero; and
2. base 66-vector with the ranking loss but no pair representation.

Ablations cannot be selected as the released V9 model and cannot change the
primary hyperparameters.  Report their benefit ranking and policy gains.

## Policy selection and release gate

Use V7's unchanged epochs, utility/risk grid, minimum two-domain conformal
risk threshold, dual-domain zero-collateral feasibility, worst-domain
selection order and V6 fact-key projection.  Report AUROC, average precision,
Brier, ECE, within-boundary pairwise accuracy, policy gain and mechanism
breakdowns.

The primary V9 model passes only if an independent audit verifies all of:

1. every seed improves lifecycle physics in both domains;
2. three-seed mean gain is at least 1.5 points in each domain;
3. TempLAMA mean gain remains at least 3.5 points;
4. every seed/domain adds zero unrelated-query errors;
5. mean safe fact-key-oracle closure is at least 50% in each domain;
6. semantic-overwrite and stale-repetition gains are nonnegative in each
   domain;
7. LAMA mean gain exceeds locked V8 by at least 0.3 point;
8. all embedding, pairing, checkpoint, result and held-out-seal audits pass.

A pass freezes V9 and authorizes a separately preregistered LoCoMo natural
session experiment.  A failure closes the learned pair-controller branch; it
does not permit another threshold search on these development sets.
