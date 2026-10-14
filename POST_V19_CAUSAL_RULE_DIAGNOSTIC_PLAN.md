# V19 Causal-Rule Explanation Diagnostic

## Freeze point

This diagnostic was specified while both V19 Qwen shards were still running
and before any V19 arm outcome was read or aggregated. It cannot release a
paper claim by itself. It determines whether a learned controller on the V19
corpus would be scientifically meaningful.

## Question

Does a deterministic write-time rule that deletes historical components with
the same explicit key as the newest conflicting update already explain the
locality-safe oracle gain?

## Frozen policies

- outer-fold-selected fixed policy, copied from the audited V19 summary;
- retain all;
- global reset;
- last one;
- last two;
- delete the newest historical component with the current explicit key;
- delete the oldest historical component with the current explicit key;
- delete every historical component with the current explicit key.

All rule decisions use only session content available after the current update
arrives. They do not inspect queries, candidate scores, answers, outcome labels
or any future event. The current update is always retained.

## Metrics and decision rule

For every policy, report joint success, generation joint success, update
accuracy, retained-key accuracy, locality accuracy and joint answer NLL. For
the whole-key rule, report subject and template clustered 95% bootstrap
intervals for joint gain over the outer-fold fixed comparator and its fraction
of the registered 3%-budget oracle gap closed.

The explicit-key explanation is considered sufficient when all hold:

1. locality accuracy is at least 97%;
2. subject and template bootstrap lower bounds for joint gain are above zero;
3. every fold has non-negative joint gain;
4. it closes at least 90% of the positive oracle-vs-fixed joint-success gap.

If sufficient, V19 is evidence for selective forgetting headroom but not for a
learned decision policy. Controller training on V19 is skipped and the next
confirmatory study must remove exact surface identity through frozen semantic
aliases. If insufficient and V19 itself passes, the next stage may train a
nested out-of-fold causal policy on V19.
