# On-policy lifecycle safe-delete head V5

## Result

V5 reconstructed all 15,480 decisions from the 20 V4 behavior arms and
deduplicated them into 2,213 distinct policy-induced states from 65 calibration
sessions. There were 659 physically safe top deletions and 1,554 unsafe
states. Every outer prediction used a three-seed model ensemble trained
without that session. The independent audit passed all nine checks with zero
probability or action mismatch.

The OOF controller made 537 true-positive and six false-positive deletions,
for 98.90% precision, 81.49% recall and a 24.54% delete rate. Fold precision
was 100%, 100%, 100%, 96.26% and 97.33%. It passed seven of the eight
registered scientific checks but exceeded the absolute false-positive budget
of four by two. The offline gate therefore stopped and did not authorize a
V5 online outcome replay or source test.

## Failure localization and next branch

All six false-positive rows reduce to three hard-retain events, with two
distinct memory states per event. In each case the incoming auxiliary fact and
the retrieved auxiliary record express the same relation for different
subjects. Whole-prompt semantic and lexical scores are consequently very
high: the controller confuses relation agreement with fact-key agreement.

The preregistered next branch is one DAgger iteration. It will collect states
created by the frozen fold-specific V5 policy, merge and deduplicate them with
the V5 corpus, and repeat the same grouped training and safety gate. No
threshold is changed post hoc. If DAgger does not solve the localized failure,
the next registered model family must add an inference-visible subject/entity
key feature; chain IDs and source actions remain forbidden at inference.

## Execution note

The first runner attempt stopped before training because the selected remote
environment does not install pytest. The runner was changed to the standard
library unittest module. Training then completed in 111.64 seconds. Its first
audit invocation used a file path and could not resolve the repository package;
the audit was rerun once through the equivalent module entry point, without
changing any model, search, threshold, prediction, metric or hash. The
recovered audit passed, and both operational failures are retained in the
experiment records.
