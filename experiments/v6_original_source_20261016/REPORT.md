# V6 LAMA/ParaRel original-source confirmation

## Question

This one-time experiment tests whether the unchanged TempLAMA-trained V5
residual controller and V6 `(normalized subject, relation id)` safety
projection transfer to the original LAMA/T-REx plus ParaRel source.  The
checkpoint, normalization, thresholds, physical policy, fact-key projection,
corpus size, endpoints and gates were frozen before the held-out episodes were
generated.

## Data and integrity

- 64 source groups, 128 twins, 384 decision boundaries and 1,920 immediate
  registered queries (3,840 scored label queries across twins/actions).
- Six held-out relations: P937, P27, P136, P407, P106 and P495, with 10--11
  target groups each.
- P530 was excluded before model execution by the model-free capacity
  amendment: only four unique subjects remained after the original
  cross-partition filter, below the frozen 128-subject requirement.
- 448 input source facts and subjects are group-unique.
- Episode SHA256:
  `bb393e5ad144a8d3b33d969162d8b1411c70007333b0070e250d15907de3bcf4`.
- Data, label, feature and result audits passed.  The result auditor exactly
  reran every policy and reproduced the complete JSON.

## Main result

| Policy | Accuracy | Gain over physics | Added unrelated errors | Oracle-gap closure |
|---|---:|---:|---:|---:|
| Explicit lifecycle physics | 81.615% | -- | 0 | 0% |
| V6 seed 26101651 | 83.203% | +1.589 points | 0 | 67.78% |
| V6 seed 26101652 | 82.891% | +1.276 points | 0 | 54.44% |
| V6 seed 26101653 | 83.359% | +1.745 points | 0 | 74.44% |
| Safe single fact-key oracle | 83.958% | +2.344 points | 0 | 100% |

The three-seed mean gain is **+1.536 points**.  The source-group bootstrap 95%
CI is **[+0.677, +2.552] points**, and the relation-cluster bootstrap 95% CI is
**[+0.202, +3.126] points**.  Mean oracle-gap closure is **65.56%**.

All six relation gains and all seven mechanism gains are nonnegative.  Gains
concentrate in mixed lifecycle (+4.568 points), semantic overwrite (+3.111)
and stale repetition (+2.901).  The subject-only ablation adds 9/15/15
unrelated errors for the three seeds; V6 adds zero for every seed.

## Registered decision

The independent result audit passed, but the preregistered confirmation gate
did **not** pass.  Seven of eight checks passed.  The sole failure is the
required mean gain of at least two points: observed gain was 1.536 points.
This threshold is not changed after the outcome.

The result therefore supports a statistically positive, zero-collateral
cross-source effect and the safety value of the fact-key projection, while it
does not release the frozen V6 branch as a fully confirmed method.  Iteration
must use new training/development evidence and a new independent confirmation
source; these 64 groups cannot tune V6.

## Efficiency

- Label construction: 77.96 s two-GPU critical path, 7.70 GiB peak reserved
  memory; per-boundary mean/median/p95/max were 0.394/0.040/1.420/1.746 s.
- Feature extraction: 12.23 s two-GPU critical path, 7.63 GiB peak reserved
  memory.
- Label safe-oracle gain over retain-all: +11.146 points with group-bootstrap
  CI [+7.969, +14.635], confirming substantial actionable headroom.

## Artifact locations

Compact evidence is stored in this directory.  Full immutable artifacts remain
on `hku-gpu2`:

- data: `/home/ctj/cbf_ttt_utility_record_controller_v6_original_confirmation_data_20261016`
- labels: `/home/ctj/cbf_ttt_utility_record_controller_v6_original_confirmation_labels_20261016`
- features: `/home/ctj/cbf_ttt_utility_record_controller_v6_original_confirmation_features_20261016`
- failed interface-only evaluation root:
  `/home/ctj/cbf_ttt_utility_record_controller_v6_original_confirmation_evaluation_20261016`
- final evaluation:
  `/home/ctj/cbf_ttt_utility_record_controller_v6_original_confirmation_evaluation_20261016_retry1`

The failed evaluation root contains no result and stopped because V1-format
labels do not duplicate the audit-only relation stratum.  The retry only
reconstructs this reporting field from immutable episode metadata; labels,
features, actions and gates are unchanged.
