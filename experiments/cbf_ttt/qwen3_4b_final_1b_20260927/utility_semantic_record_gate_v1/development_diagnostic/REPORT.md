# Utility record controller V1 development diagnostic

## Result

The diagnostic completed on the untouched 32-group evaluation half and passed
all nine independent audit checks.  The strongest safe deterministic baseline,
selected on the disjoint 32-group calibration half, was explicit lifecycle
physics.  Its evaluation accuracy was 81.823% with zero added unrelated errors.

| Controller seed | Frozen setting | Accuracy | Gain over physics | Group bootstrap 95% CI | Oracle-gap closure | Added errors |
|---:|---|---:|---:|---:|---:|---:|
| 2610161 | u=0.6, p=2.0, c=0.05 | 84.635% | +2.812pp | [1.094, 4.844]pp | 61.4% | 0 |
| 2610162 | u=0.6, p=2.0, c=0.05 | 85.052% | +3.229pp | [1.302, 5.521]pp | 70.5% | 0 |
| 2610163 | u=0.6, p=2.0, c=0.01 | 84.948% | +3.125pp | [1.250, 5.365]pp | 68.2% | 0 |

Retain-all reached 78.750%.  The full safe oracle reached 86.406%, so the
controllers remained 1.35--1.77 points below it.  The learned gain localized
mainly to semantic overwrite (13.89--18.33 points over physics) and stale
repetition (5.95 points); reset/revoke behavior was unchanged by design.  Every
seed preserved the physics baseline's 89.236% unrelated accuracy and added no
errors.

## Interpretation

This is positive evidence for the hybrid architecture.  It rules out the
explanation that the formal counterfactual teachers merely encode a rule already
captured by exact lifecycle operations: on held-out relations, learned semantic
decisions add about three points over the strongest zero-collateral rule and all
cluster intervals exclude zero.

It does not release the V1 confirmation.  The preregistered full-development
safe-oracle gain was 9.375 points, below the frozen ten-point headroom gate.
That gate remains false regardless of controller performance.  Confirmation
and external benchmarks were not loaded.  The next stage is the prospectively
registered cross-source V2 replication on official TempLAMA before either the
TempLAMA test or the original LAMA/ParaRel confirmation can be opened.

## Reproducibility

- Remote output root:
  `/home/ctj/cbf_ttt_utility_record_controller_v1_development_diagnostic_20261016`
- Remote commit: `89787a3`
- Result SHA256:
  `5d8f7b543f00a4f0144296a6a95bb7da5086bbfe8a0e51d741d0a7b27e6f1f2a`
- Runner exit code: 0
- Independent audit: passed
- Local artifacts: `result.json` and `development_diagnostic_audit.json`

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
bash scripts/run_qwen_utility_record_controller_v1_development_diagnostic.sh
```
