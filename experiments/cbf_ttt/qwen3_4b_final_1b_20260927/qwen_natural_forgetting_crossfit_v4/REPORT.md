# Factor Gate V4 cross-fit reconstruction

V4 reconstructed one engineered and one no-schedule controller for each of the
five outer V2 folds and all three seeds. Each checkpoint was trained only on
the other four folds with that fold's preregistered nested-search
configuration, normalization and thresholds.

The reconstructed models reproduced all 4,644 saved combinations of 774 rows,
two feature views and three seeds with maximum benefit/risk probability error
`0.0` and zero action mismatch. All four release checks and all six independent
audit checks passed. Development, test and external benchmarks were not
loaded.

This artifact allows every V4 online calibration session to use a controller
that was never trained on that session. It is an isolation prerequisite, not a
new accuracy result. The binary checkpoint remains on the remote experiment
server and is excluded from Git by the repository-wide `*.pt` rule; hashes and
complete machine-readable audit evidence are archived here.
