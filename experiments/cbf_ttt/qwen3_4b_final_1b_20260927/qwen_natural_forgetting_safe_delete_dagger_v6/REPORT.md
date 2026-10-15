# Cross-fit safe-delete DAgger V6

## Result

V6 rolled each of the 65 calibration sessions with its fold-specific V5
controller. The collection contained all 774 events and passed all nine
collection-audit checks, including exact probability/action replay and
confirmation that each behavior controller excluded its own session.

Only 19 rollout states were absent from the 2,213-row V5 corpus. After exact
deduplication, the V6 corpus contained 2,232 states. Repeating the unchanged
nested protocol produced 551 true-positive, six false-positive and 127
false-negative deletes: 98.92% precision, 81.27% recall and 24.96% deletion.
The training audit passed all nine checks with zero prediction/action mismatch.

V6 failed only the registered maximum of four false positives. The exact six
errors and the two affected folds were unchanged from V5. DAgger therefore
rules out missing on-policy state coverage as the explanation. The failure is
representational: the ten features cannot distinguish a shared relation from
a shared fact key when two subjects use the same prompt template.

## Next registered branch

The V6 plan permits no second DAgger pass or post-hoc threshold adjustment.
The next experiment adds a deterministic entity anchor derived only from the
incoming and retrieved prompt text. It evaluates both a soft augmented head
and a physical barrier requiring observable entity-key containment. Dataset
subject fields and chain identities remain excluded at inference. Test and
external benchmarks remained sealed throughout V6.
