# V8 relation-posterior utility controller

## Registered cause and hypothesis

The audited V7 failure diagnostic rejected both threshold selection and
negative transfer as the principal cause of the LAMA development miss.  The
joint controller's LAMA ceiling remained +0.859 points / 44.59% oracle-gap
closure; LAMA-only fitting was worse at +0.495 points / 25.68%.  Its benefit
head had only 0.788--0.819 AUROC and 0.209--0.293 average precision, while the
risk head had 0.973--0.986 AUROC.  A source centroid score reached 0.836
AUROC, with the largest shifts in scalar semantic-cosine features.

V8 tests one prospective change: replace part of the source-sensitive scalar
similarity bottleneck with a supervised pairwise relation-posterior
representation.  The frozen Qwen encoder maps candidate and incoming facts to
relation-slot vectors; train-only relation supervision shapes three linear
heads; agreement between the two posterior distributions becomes an
inference-safe compatibility feature.  Counterfactual benefit/risk labels
still train the final forgetting controller.  This is not a generic relation
classifier claim and does not change the current-write gate, which remains
fixed at `g=1`.

## Immutable evidence boundary

Use exactly the V7 audited sources:

- TempLAMA train: 512 groups, 4,392 residual candidates;
- LAMA/ParaRel train: 512 groups, 4,684 residual candidates;
- TempLAMA development: 256 groups, 2,204 candidates; and
- LAMA/ParaRel development: 64 groups, 588 candidates.

The final 1B-token Qwen3-4B checkpoint is a frozen feature extractor with
native TTT disabled.  TempLAMA test, the original-source confirmation,
LongMemEval-S, LoCoMo and every earlier stopped semantic benchmark are
prohibited inputs.  Relation IDs and counterfactual outcomes from development
may be used only for the registered evaluation metrics, never for fitting the
relation heads or controller.

## Frozen relation representation

For every record render only:

```text
Represent the relation slot only: <statement with [SUBJECT] and [VALUE]>
```

Mean-pool and L2-normalize the final hidden states of the frozen Qwen model.
Deduplicate training mentions by `(rendered text, relation id)`.  If one
rendered text maps to multiple relation IDs, exclude it from relation-head
training and record the ambiguity; it remains legal at inference.

Fit three train-only linear relation heads with seeds
`26101651/26101652/26101653`, combined-train normalization, square-root
inverse-frequency class weights, ten full-batch epochs, AdamW learning rate
0.01 and weight decay 0.05.  The union of training relation IDs defines the
output vocabulary.  A development relation need not occur in that vocabulary:
posterior compatibility compares distributions and never exposes a relation
ID to the deployed controller.

For each candidate/incoming pair append these eight features to the locked V7
66-vector:

1. ensemble-mean posterior dot product;
2. ensemble standard deviation of posterior dot product;
3. ensemble-mean logit cosine;
4. ensemble standard deviation of logit cosine;
5. fraction of heads whose top relation class agrees;
6. mean incoming maximum posterior probability;
7. posterior-dot margin over the best competing candidate; and
8. posterior-dot rank fraction within the current memory boundary.

The result is a 74-dimensional vector.  Save the raw normalized embeddings,
relation-head states, ordered identities and augmented feature rows so an
independent auditor can reconstruct all eight values.  Record extraction time,
token lengths and peak allocated/reserved memory per GPU shard.

The first extraction stopped on its output-capacity assertion before feature
audit or utility-controller training.  The registered 4,392/4,684 and
2,204/588 counts are residual candidates after the exact-subject filter,
whereas the source feature shards intentionally retain every pre-filter
candidate.  The corrected extraction support is 3,072 boundaries / 9,216
candidates in each training domain, 1,536 / 4,608 in TempLAMA development and
384 / 1,152 in LAMA development.  This capacity correction changes no row,
feature, target, filter, hyperparameter or release gate.  The stopped output
is retained and the audited retry uses a new directory.

## Controller and selection

Retain V7's 128/64 MLP, benefit/risk heads, optimizer, batch size, dropout,
gradient clipping, seeds, checkpoint epochs, utility thresholds, risk
penalties and conformal levels.  Normalize the 74 features on the combined
training data, use combined class weights, apply the V6 exact fact-key safety
projection, and select only zero-collateral configurations by the same
worst-domain ordering used by V7.  No relation-posterior threshold is added.

Report the locked V7 model as the primary ablation.  Also report record-level
benefit/risk AUROC, average precision, Brier score and ECE for V8 so the causal
chain from representation to policy gain is visible.

## Development release gate

V8 passes only if all conditions hold after an independent audit:

1. every seed improves lifecycle physics in both domains;
2. three-seed mean gain is at least 1.5 points in each domain;
3. every seed/domain adds zero unrelated-query errors;
4. mean safe fact-key-oracle gap closure is at least 50% in each domain;
5. semantic-overwrite and stale-repetition gains are nonnegative in each
   domain;
6. mean LAMA gain exceeds locked V7 by at least 0.5 point;
7. mean LAMA benefit average precision exceeds locked V7 by at least 0.03;
8. every data, embedding, relation-head, feature, checkpoint and result audit
   passes; and
9. all held-out sources remain sealed.

A pass freezes all V8 artifacts and authorizes a separately preregistered
natural-session LoCoMo experiment.  Failure stops this posterior-feature
branch.  It may motivate a direct pairwise counterfactual-utility projection,
but may not weaken the V8 gate or tune on any opened confirmation.
