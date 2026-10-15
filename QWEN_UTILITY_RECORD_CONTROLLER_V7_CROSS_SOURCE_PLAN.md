# V7 domain-balanced cross-source residual controller

## Evidence boundary and hypothesis

V6 passed official TempLAMA test with a 4.151-point gain and zero collateral,
then produced a statistically positive 1.536-point gain on the one-time
LAMA/ParaRel original-source confirmation.  That confirmation failed only its
preregistered two-point effect-size gate; the safe fact-key oracle headroom was
2.344 points and V6 closed 65.56% of it.  The 64 confirmation groups are now
closed and cannot train, select, calibrate or repair V7.

The prospective hypothesis is that the unchanged residual-controller
architecture can reduce the domain gap when trained jointly on the two
already audited training sources, while the deterministic V6 fact-key safety
projection preserves zero collateral in both already observed development
domains.  V7 is a development stage.  A pass may authorize a separately
registered LoCoMo confirmation; it does not reopen TempLAMA test,
LAMA/ParaRel confirmation, LongMemEval-S, or any earlier held-out result.

## Fixed data boundary

Training uses only:

1. the 512-group TempLAMA official-train corpus used by V5: 4,392 exact-subject
   residual candidates, 685 benefit positives and 201 risk positives; and
2. the 512-group LAMA/T-REx plus ParaRel training-relation corpus used by V1:
   4,684 candidates, 747 benefit positives and 160 risk positives.

The domains contribute 48.39% and 51.61% of the combined 9,076 examples, so
every example is used once per epoch without replication or source weights.
No domain identifier enters the numeric feature tensor.

Selection uses only the already observed, relation-disjoint development sets:

- TempLAMA validation: 256 groups, 2,204 candidates; and
- LAMA/ParaRel development: 64 groups, 588 candidates.

Every source manifest, label file, feature file and audit must be hash locked.
The LAMA label format's relation stratum is reconstructed only from its paired
audited feature metadata and never enters the controller tensor.

## Frozen model family and search

Retain V5's exact 66-dimensional inference-safe tensor, 128/64 MLP,
benefit/risk heads, optimizer, batch size, dropout, gradient clipping, three
seeds, epochs `{25,50,100}`, utility thresholds `{0.4,0.5,0.6,0.7,0.8}`, risk
penalties `{0,0.5,1,2}` and conformal levels `{0.01,0.03,0.05}`.  Normalize on
the combined training examples and use combined-training class weights.

For each epoch and seed, compute a conformal risk threshold independently in
each development domain and deploy the smaller threshold.  For every finite
configuration, first apply the learned one-delete residual policy and then the
unchanged V6 `(normalized subject, relation id)` projection.  A configuration
is feasible only when it adds zero unrelated-query errors in both domains.

Among feasible configurations select lexicographically by:

1. larger worst-domain fact-key-oracle gap closure;
2. larger mean closure across the two domains;
3. larger worst-domain accuracy gain over lifecycle physics;
4. larger mean gain;
5. higher mean residual-deletion precision;
6. lower mean deletion fraction; and
7. the existing simpler-model deterministic tie breaks.

The three selected checkpoints remain separate; no seed is chosen after
development.

## Development gate

V7 passes only if all checks hold:

1. each domain's safe single fact-key oracle headroom is at least 1.5 points;
2. every seed improves over lifecycle physics in each domain;
3. the three-seed mean gain is at least 1.5 points in each domain;
4. every seed adds zero unrelated errors in each domain;
5. mean fact-key-oracle gap closure is at least 50% in each domain;
6. semantic-overwrite and stale-repetition mean gains are nonnegative in each
   domain; and
7. all data, label, feature, checkpoint, result, hash and seal audits pass.

A pass authorizes only a new LoCoMo protocol with a natural-text fact-key
adapter and independent conversation-level uncertainty.  A failure stops this
unchanged 66-feature/MLP cross-source iteration.  No V7 result may be evaluated
on the already opened TempLAMA test or LAMA/ParaRel confirmation.

