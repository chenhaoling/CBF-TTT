# V9 direct pairwise utility controller

## Decision

V9 failed its preregistered development release gate.  All integrity and
safety audits passed, but the learned pair branch is closed.

## Main result

| Arm | TempLAMA gain | LAMA gain | LAMA pairwise accuracy |
|---|---:|---:|---:|
| pair + rank (primary) | +2.606 pp | **+1.189 pp** | **90.91%** |
| pair, no rank | +2.595 pp | +1.155 pp | 90.91% |
| base 66 + rank | +2.480 pp | +1.137 pp | 82.83% |

The primary model closed 60.47% of TempLAMA and 61.71% of LAMA safe
fact-key-oracle headroom.  Every arm/seed/domain added zero unrelated errors.
The independent auditor reconstructed all nine checkpoints and metrics and
numerically verified that risk logits were invariant to pair tensors.

## Interpretation

Direct pair vectors and the boundary ranking loss improve record ranking, but
LAMA policy gain exactly matches V8 and TempLAMA regresses from V8's +4.214
points.  The result rules out another small pair head or ranking loss as the
main missing component.  Training coverage is the next registered cause: the
two existing sources produced only 190 unambiguous deduplicated relation-slot
texts, while the difficult evaluation is relation and surface-form shifted.

V10 therefore constructs a new train-only natural counterfactual corpus from
official FewRel training relations.  It returns to the locked 66-feature V7
controller so the experiment isolates data diversity from architecture.
