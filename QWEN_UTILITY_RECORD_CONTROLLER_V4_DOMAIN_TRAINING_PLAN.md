# Qwen utility record controller V4: TempLAMA domain training

## Freeze point

This protocol is frozen after the audited V3 train-only locality calibration
failed on TempLAMA validation and before constructing a larger TempLAMA train
corpus or observing any new model output.  TempLAMA test and the original
LAMA/ParaRel confirmation remain sealed.

V3 reduced validation collateral errors from 28--87 to 4--6 per seed but also
reduced mean gain over explicit lifecycle physics from 1.81 to 1.07 points.
This shows that conservative projection alone does not supply a domain-robust
utility/risk representation.  V4 trains the same small shared-record
architecture on official TempLAMA train subjects and treats the existing
addressable-memory outcome model as a fixed label generator.

## Formal train corpus

Use exactly 512 unique target subjects from the decontaminated official
TempLAMA train split.  Allocate a minimum of 16 groups per relation and
Hamilton-apportion the remainder by eligible trajectory capacity:

| Relation | Eligible | Train groups |
|---|---:|---:|
| P102 | 134 | 75 |
| P108 | 159 | 87 |
| P127 | 47 | 31 |
| P286 | 104 | 60 |
| P39 | 112 | 64 |
| P488 | 79 | 47 |
| P54 | 124 | 70 |
| P6 | 92 | 54 |
| P69 | 33 | 24 |

Sampling is deterministic by a new frozen seed.  Target subjects are unique.
All backgrounds come from remaining train subjects, are distinct inside a
group and may recur across groups.  Construct two twins at lengths 8/16/32/64,
the seven registered mechanisms, three sparse boundaries and five future
queries per boundary.  Real source transitions supply every target update.
Validation/test rows and any earlier validation model outcome are forbidden
construction inputs except that the passed V3 result audit must prove the
branch transition and sealed-test state.

## Labels and features

Enumerate all binary endpoint actions plus global clear with incoming write
fixed to one.  The safe Oracle cannot reduce unrelated or anchor correctness
relative to retain-all.  Use episode-scoped caches and record time, cache use
and peak memory.  An independent auditor reconstructs every action, label,
aggregate and source-group interval.

Extract the unchanged 66-dimensional inference-safe V1 tensor: 22 record
features plus candidate mean and max context.  Timestamps, source partition,
relation ID, mechanism, answers, future outcomes and teacher actions are not
numeric inputs.

## Domain controller training

Use the existing shared MLP (66 -> 128 -> 64, utility softmax and collateral
risk sigmoid) and loss definition.  Train seeds `26101641`, `26101642` and
`26101643` with AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size
256, dropout 0.1 and gradient norm 1.0.  Candidate epochs are 25, 50 and 100.

For checkpoint selection, SHA256-split target source groups within every
relation into 80% inner train and 20% inner validation.  Twins and all
boundaries remain in one group.  A checkpoint is eligible only with zero risky
single-record deletions after its empirically zero-risk projection.  Select by
boundary action accuracy, record action accuracy, risk recall and earlier
epoch.  Retrain the selected epoch on all 512 groups with normalization and
class weights fit only on formal train.

## Validation selection and honest evaluation

Hash the 256 validation target groups into fixed, relation-stratified 128-group
calibration and 128-group evaluation halves.  All twins remain together.
Calibration enumerates utility thresholds `{0.4,0.5,0.6}`, risk penalties
`{0,0.5,1,2}`, conformal levels `{0.01,0.03,0.05}` and the ten V3 locality
rules.  Select each seed only among zero-collateral calibration configurations
using accuracy, deletion precision, lower deletion fraction and the registered
tie order.  Evaluation is scored once after selection.

Report the V2 baselines, majority action, safe binary Oracle, 20 matched-random
draws, query roles, mechanisms, relations, group/relation uncertainty and
background diagnostics.  The evaluation half releases TempLAMA test only if:

1. safe Oracle headroom over the strongest zero-collateral baseline is at least
   three points;
2. every seed gains at least one point and mean gain is at least two points;
3. pooled target-group 95% lower bound is above zero;
4. every seed adds zero unrelated or anchor errors;
5. mean Oracle-gap closure is at least 40%;
6. overwrite and stale-repetition mean gains are nonnegative;
7. macro-relation gain is positive, at least eight relations are nonnegative,
   and the worst relation loses at most two points;
8. every upstream and result audit passes.

The official test is then scored once with unchanged checkpoints, normalization
and selected calibration settings.  A failed V4 evaluation keeps it sealed.

## Claim boundary

A test pass supports domain-trained selective forgetting for addressable
session memory on unseen subjects and temporal trajectories.  V2 remains the
reported failed zero-shot transfer result.  V4 does not make a claim about
content binding in the original dense `ttt_conv`/`ttt_proj` writer.
