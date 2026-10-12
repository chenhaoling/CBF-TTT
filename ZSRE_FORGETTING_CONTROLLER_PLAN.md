# ZsRE update-conflict forgetting controller plan

## Trigger

Hard-retain A2 created the preregistered cosine-to-oracle gap (3.64 points), but the frozen query-scope gate was 2.46 points below cosine and its paired confidence interval crossed zero. Query relevance therefore does not transfer directly to update conflict.

## Calibration-only controller

Each update is a variable-size decision over the active memory: delete candidate 0–7 or retain all. Training examples use oracle memory states. A local event labels exactly the predecessor from the same update chain; primary-initial and same-subject auxiliary events label retain. Chain ids and action labels are used only to construct targets and never enter controller features.

Candidate features concatenate the frozen Qwen joint key/update representation, three frozen bidirectional cosine similarities and deterministic lexical similarities. The controller is a regularized linear candidate scorer with a learned retain logit.

## Nested leave-one-session-out protocol

There are seven calibration sessions. For each outer held-out session:

1. choose one of the remaining sessions deterministically for inner epoch selection;
2. train on the other five sessions;
3. select among 10/25/50/100/200 epochs using inner accuracy then NLL;
4. refit on all six outer-training sessions for the selected epoch;
5. replay the held-out session causally so early errors affect later memory.

Repeat with three controller seeds and pair them with the three frozen readout seeds. After LOSO evaluation, train deployable heads on all calibration sessions using the median selected epoch. Development and test remain sealed.

## Decision gate

Compare against the preregistered cosine-local baseline by paired primary/auxiliary subject unit. Enter development only if both memory-query accuracy and action accuracy have bootstrap 95% lower bounds above zero. Otherwise diagnose calibration errors and revise the feature or objective without loading development or test.

## V1 outcome and V2 repair

The full 2,571-dimensional candidate controller overfit 75 candidate-bearing events: nested-LOSO query/action accuracy was 54.64/54.88%, versus 74.02/91.46% for cosine. Low-dimensional lexical diagnostics reached at most 76.00% action accuracy alone. V2 therefore freezes cosine candidate selection and trains only a 13-dimensional binary delete/retain gate from three cosine scores and general lexical/relation similarities. Epoch and probability threshold selection remain nested inside each outer fold. This residual design can only change whether the strongest cosine candidate is deleted; it cannot replace the stable candidate ordering.

V2 raised query/action accuracy to 76.45/95.12%, with +2.75/+4.17 point mean gains over cosine, but the paired confidence intervals had lower bounds 0.00/-1.92 points. V3 augments only the binary gate training with the 267-subject contextual calibration split: each canonical fact supplies one exact-to-rephrase positive and two exact-to-locality negatives. Every outer fold excludes all auxiliary examples sharing a subject with the held-out natural session; the inner tune session's subjects are also excluded during epoch and threshold selection. The natural causal sessions remain the sole outer evaluation source.
