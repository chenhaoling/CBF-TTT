# Native Layerwise Fast-Weight Admission V2

## Decision

V2 is an oracle/headroom diagnostic on the 48 already observed V1 groups.  It
passed its frozen diagnostic gate but is development evidence only.  It does
not establish a learned-gate or held-out effectiveness claim.

The whole seven-layer native update remained harmful:

| Action | Mean J |
|---|---:|
| reject all | 7.56404 |
| admit all | 11.99533 |
| safe singleton oracle | 7.47812 |

The safe singleton oracle improved J by 0.08592 over reject-all; the paired
48-source-group bootstrap 95% interval was `[0.06418, 0.11321]`.  Singleton
headroom appeared at 132/144 boundaries (91.67%).  All frozen safety,
composition and independent-audit checks passed.

Only layer 24 had positive mean fixed-layer benefit (+0.00407).  Other fixed
singletons were negative on average, while the rowwise oracle selected several
layers.  This supports state-dependent layer selection rather than a globally
useful layer or whole-update gate.

## Claim boundary

V2 shows that useful and harmful components can coexist inside a native
seven-layer write.  Because the same V1 groups informed the diagnostic, V2
cannot be cited as an independent replication.  The frozen next step was a
source-disjoint confirmation before controller training.

## Artifacts

The directory retains smoke/pilot summaries and independent audits.  Raw action
rows remain hash-locked on `hku-gpu2` at
`/home/ctj/cbf_ttt_native_layerwise_write_gate_v2_20261016`.
