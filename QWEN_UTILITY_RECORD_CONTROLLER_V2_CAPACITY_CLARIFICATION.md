# V2 TempLAMA capacity and background-bank clarification

## Freeze boundary

This clarification is frozen after the model-free TempLAMA source inspection
and the train-only ternary resource pilot passed, but before constructing or
scoring any validation episode.  It does not change the frozen controller,
thresholds, 256-group development size, mechanisms, queries, release gates or
test/confirmation seals.

## Relation allocation

The parent protocol requires exactly 256 validation groups and representation
of every eligible relation.  Equal allocation is impossible after deterministic
decontamination because validation relation capacities range from 16 to 79
eligible trajectories.  Allocate a minimum of eight target groups to each of
the nine relations, then distribute the remaining 184 groups by Hamilton
apportionment in proportion to each relation's remaining capacity.  Ties use
relation ID.  This produces the fixed quotas:

| Relation | Eligible | Development groups |
|---|---:|---:|
| P102 | 45 | 31 |
| P108 | 79 | 52 |
| P127 | 22 | 17 |
| P286 | 42 | 29 |
| P39 | 44 | 30 |
| P488 | 36 | 25 |
| P54 | 54 | 36 |
| P6 | 33 | 23 |
| P69 | 16 | 13 |

Sampling inside each relation remains SHA256 deterministic.  Every target
subject appears in exactly one source group.  Primary micro accuracy is
reported together with equal-weight macro-relation gain and relation-cluster
uncertainty, as already required by the parent protocol.

## Fixed background bank

The validation partition has 532 decontaminated subjects, which cannot provide
seven globally unique input subjects for each of 256 groups.  Target subjects
are reserved and never used as distractors.  Same-relation and protected
background records are selected from the remaining validation-only subjects;
subjects are distinct inside a group but may recur as fixed background across
groups.  Selection depends only on source hashes, relation and group index.

Bootstrap clusters remain target source groups.  A sensitivity analysis must
also cluster jointly by target group and reused-background identity, and a
background-held-out subset must exclude every group sharing a background with
another selected group.  The development gate requires the primary
subject-group interval; the two sensitivity intervals are reported and cannot
be used to choose a method or threshold.

## Pilot decision

The audited train-only pilot reached 91.18% safe-oracle accuracy versus 83.87%
retain-all, a 7.31-point gain with source-group 95% interval
`[4.98, 9.93]` points and zero added unrelated errors.  Half strength was
uniquely NLL-optimal on 2.08% of boundaries and changed no prediction.  The
registered formal action space is therefore all binary endpoint tuples plus
global clear.  The projected two-GPU label time for 256 groups is 480 seconds
and projected output is 0.012 GiB; these values change no sample or gate.
