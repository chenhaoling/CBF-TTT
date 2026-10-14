# Dynamic record forgetting: audited resource pilot

## Decision

The corrected two-RTX-5090 pilot passed all nine independent audit checks and
selected **512 formal training source groups**, the largest preregistered
choice. The critical-path wall time was 26.73 seconds for 56 pilot groups and
three projection seeds. Linear two-GPU extrapolation is 61.09, 122.19 and
244.38 seconds for 128, 256 and 512 groups. Overall peak allocated/reserved GPU
memory was 7.912/8.221 GiB, below the 24-GiB gate.

The first execution produced identical scientific labels but reset CUDA peak
counters at each label and omitted the Qwen extraction peak. It is excluded
from formal size selection. The corrected retry preserved extraction, label
and overall peaks separately; its non-resource fields match the excluded run
exactly.

## Pilot design

- Corpus seed `202610156`.
- Seven lifecycle families: retain, recent revoke, distant revoke, reset,
  exact duplicate, overwritten-record no-op and mixed two-event.
- Exactly eight source groups per family, two anchor twins per group and three
  sparse decision boundaries.
- Session lengths 8, 12 and 16.
- Projection seeds 2301, 2302 and 2303.
- Dynamic selection of at most three old records.
- Full `{0,0.5,1}^3` grid under the registered global-clear and physical masks.
- Accuracy-first lexicographic oracle, followed by NLL, removed mass and fixed
  action order.

The run constructed 1,008 independently hashed label rows, 144 per family.
Valid-action counts were 3/9/27 for 18/552/438 rows. It observed 17 distinct
actions. Retain-all accounted for 654 rows and global reset for 144; the
remaining 210 rows require record-specific state changes, including interior
coefficients.

## Measured resources

| Quantity | Result |
|---|---:|
| Mean time per label | 0.02076 s |
| P95 time per label | 0.04471 s |
| Mean time per valid action | 0.001192 s |
| Label peak allocated/reserved | 0.0141 / 0.0273 GiB |
| Qwen extraction peak allocated/reserved | 7.9116 / 8.2207 GiB |
| Overall peak allocated/reserved | 7.9116 / 8.2207 GiB |
| Maximum prompt length | 1,730 tokens |
| Maximum answer length | 24 tokens |
| Maximum active records | 15 |

## Integrity

The independent auditor reloaded the causal corpus, verified all shard and raw
row hashes, reproduced labels across the two executions after removing only
timing/memory fields, recomputed counts and resource statistics, reapplied the
formal-size rule and confirmed that no formal result root existed. All nine
checks passed. Remote unit tests passed 6/6 and runner exit code was zero.

The resource result measures the controlled proxy stage. It does not yet show
that the dynamic controller learns the labels or improves held-out accuracy.
The next stage is a 512-group train/development run; sealed matched test data
must remain absent until every development gate passes.
