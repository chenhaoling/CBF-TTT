# Native Layerwise Fast-Weight Admission V3

## Decision

V3 built 624 source-disjoint groups and locked confirmation/train/development/
sealed-test partitions of 48/384/96/96.  The corpus passed all 11 independent
data checks before model scoring.  The 48-group fresh confirmation reproduced
layerwise headroom but **failed the frozen EOS safety gate**.  Therefore V3 did
not score train/development/test and did not train a controller.

## Fresh confirmation

| Quantity | Result |
|---|---:|
| Groups / boundaries | 48 / 144 |
| Singleton-headroom fraction | 90.97% |
| reject-all J | 7.55667 |
| admit-all J | 11.95417 |
| accuracy-safe oracle J | 7.46663 |
| Oracle gain over reject | 0.09005 |
| Group bootstrap 95% CI | [0.07071, 0.11229] |
| Oracle protected-accuracy delta | +1.157 pp |
| Oracle protected-NLL delta | -0.02103 |
| Oracle EOS delta | **+1.250 pp** |
| Allowed EOS delta | <= +1.000 pp |

The execution and independent artifact audits passed.  Feature width was
1,854; test was unopened.  Each label scored reject, full admit and seven
singleton branches.  Mean time was 14.31--14.37 seconds per label; peak CUDA
allocated/reserved memory was 10.76/11.47 GiB.

## Post-stop diagnostic

The diagnostic did not alter the V3 decision.  Adding a rowwise requirement
that singleton EOS may not exceed reject changed 7/144 oracle actions.  The
result retained gain 0.08059 with bootstrap interval `[0.06412, 0.09991]`,
improved protected accuracy by 0.694 pp, reduced protected NLL by 0.02576 and
reduced EOS by 0.278 pp.

This is design evidence for a new EOS-risk head.  It is not a prospective V3
confirmation.  V4 was therefore frozen before any of the untouched train/dev
groups were scored.

## Claim boundary

V3 independently supports native fast-weight layer selectivity and rejects the
whole-update scalar gate.  It does not validate a deployable learned controller
because its registered safety gate failed.  Sealed-test release count is zero.

Raw semantic corpus and action rows remain on `hku-gpu2`.  This directory keeps
only manifests, summaries, audits, resource sidecars and the failure diagnostic.
