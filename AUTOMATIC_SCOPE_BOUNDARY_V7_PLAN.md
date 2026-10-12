# Automatic scope boundary V7 preregistration

## Motivation and scope

V6 improved text-only candidate top-1 to 93.25%, but failed its 95% gate. Every
inner fold selected the maximum registered entity/relation weight of 8, and
inner accuracy was still monotonically increasing at that boundary. This is
evidence that the registered grid truncated calibration, rather than evidence
for another representation change.

V7 performs exactly one boundary-extension diagnostic. It changes no text,
features, Qwen checkpoint, V5 relation classifier, folds, IDF construction,
metric, or gate. The only candidates are `{8, 12, 16, 24, 32, 48, 64}`. Each
outer weight is selected on the next disjoint inner fold exactly as in V6.

## Gate and stop rule

The unchanged V6 gate requires top-1 ≥95%, MRR ≥97%, performance within three
points of oracle-segmented V5, at least ten points gain over V4, and a positive
paired-bootstrap lower bound.

If V7 fails, weight-grid expansion stops permanently and the automatic entity
representation must change. If it passes, the selected text-only resolver is
frozen for causal forgetting; further CounterFact scope tuning is prohibited.

