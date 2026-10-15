# Natural counterfactual forgetting-utility pilot V1

## Scientific question

The frozen V5 natural-memory reader passed its one-time subject-disjoint
development release.  This pilot asks the next, separate question: does a
causal intervention on the *session memory state* create enough safe future
utility to supervise a forgetting controller?  The Qwen weights, native
writer, contextual transforms, V5 reader and value bridge are frozen.

This is a calibration-only resource and headroom experiment.  It may decide
whether to construct the full controller-training corpus; it may not release a
development, test or external-benchmark result.

## Sealed inputs

- natural causal-memory manifest SHA256:
  `ceb2c85ae2a01fd6de7ef08d68004e58e43b8ae424708bea4fa313344d1a7ba7`;
- V5 reader checkpoint SHA256:
  `3a0ba5a83ece4850dd3824fc2277dc2e0dfdb04902e361dbff764e1b630abadc`;
- contextual transforms SHA256:
  `6f6a696005185d80d6b8f468fa6442146dc8d9683e232e7a591676ba9515c378`;
- decomposed scope checkpoint SHA256:
  `0ed4dab8b77fd0fb91fdd49720c783261ab2fd5c64cd705ebe36346d99615619`;
- Qwen3-4B 1B-token checkpoint at native step 81,381;
- scale-8 natural value residual and V5 threshold 0.30.

## Pilot sampling

Use exactly 16 of the 65 calibration sessions.  Include the unique short
session when present, then rank all remaining sessions by SHA256 of
`natural-forgetting-utility-pilot-v1|session_id` and take the first 15.
Development and test data remain unopened.

Within each selected session label every `primary_update`, because those are
the registered conflict boundaries, plus one hash-selected
`primary_initial` and one hash-selected `hard_retain` anchor.  This gives a
sparse mixture of positive and negative forgetting opportunities without
using model outcomes to select a decision point.

## Counterfactual action grid

Apply one intervention immediately before the selected event, then append the
incoming record.  All later events append records without another
intervention.  Reconstruct the pre-intervention state with retain-all so each
label estimates the marginal effect of one action.

Use five scopes:

1. `global`: every existing record;
2. `last1`: the most recent record;
3. `last2`: the two most recent records;
4. `scope_top1`: the frozen key scorer's top record for the incoming prompt,
   independent of the retrieve/defer gate;
5. `hybrid`: the union of `scope_top1` and `last1`.

Cross each scope with retention strength `alpha in {0, 0.5, 1}`.  Alpha zero
deletes the selected records; alpha 0.5 halves their value-residual strength;
alpha one leaves them unchanged.  Since all alpha-one states are identical,
execute them once as `retain_all` and record the five registered alpha-one
aliases.  Thus the full 15-point grid has 11 unique executions.

The intervention affects session memory only.  It never erases or changes the
pretrained Qwen parameters.

## Future-utility label

After an action, replay every registered checkpoint from the intervention to
the end of the session.  At each checkpoint evaluate every active exact and
rephrased query with the frozen V5 reader and the selected record's retained
value strength.  Cache identical query/selected-value/strength computations;
cache reuse is part of the intended construction implementation.

For each action record future candidate correctness, answer NLL, related and
unrelated correctness, stale/wrong/defer outcomes, and mean memory mass.  An
action is safe only when it creates no additional candidate error on chains
unrelated to the incoming event compared with `retain_all`.

Choose the safe oracle label lexicographically among safe actions by:

1. maximum total future candidate correctness;
2. minimum total future answer NLL;
3. minimum mean memory mass;
4. fixed action order.

Also report the unconstrained oracle.  This control-barrier rule prevents a
controller target from trading an unrelated fact for a gain on the updated
fact.  Candidate accuracy is the primary endpoint; NLL and memory mass break
ties rather than introducing unregistered scalar weights.

## Resource accounting

For every label record elapsed seconds, Qwen cache hits/misses, absolute peak
allocated GPU memory and peak reserved GPU memory.  Report median, p90 and
maximum label time plus the two-GPU critical path.  Keep raw per-query action
outcomes so aggregates and labels can be audited.

## Preregistered pilot decision

`scale_to_formal` passes only when all checks hold:

- at least 90 counterfactual labels are produced;
- every label contains all 11 unique actions and all 15 registered grid
  points through the alpha-one aliases;
- at least 10% of safe-oracle labels choose a non-retain action;
- safe-oracle future candidate accuracy exceeds retain-all by at least one
  percentage point;
- the safe oracle introduces zero aggregate unrelated candidate errors beyond
  retain-all;
- at least three canonical actions are selected;
- median construction time is no more than 120 seconds per label;
- peak allocated memory is no more than 28 GiB on each RTX 5090.

In addition, report a 20,000-draw session bootstrap confidence interval for
safe-oracle minus retain-all candidate accuracy.  A lower bound above zero is
the stronger `pilot_evidence_ready` result, but it is not required to enlarge
the calibration label set because this 16-session pilot is intentionally
underpowered.

Passing `scale_to_formal` authorizes construction on all 65 calibration
sessions and controller training with grouped out-of-fold evaluation.  A
failure stops scaling and requires diagnosis of action coverage, reader
behavior or scenario composition before changing the controller.
