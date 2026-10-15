# Preregistered Qwen natural residual scope gate V4

## Motivation and fixed boundary

The audited V2 linear transfer gate significantly improved end-to-end natural
memory but missed its correct-top recall gate by 0.97 points.  V3 tested a
joint linear key/defer replacement and stopped before Qwen outcome scoring: no
configuration retained both 92% no-match specificity and 98% exact selection.
Its best safe configuration reduced exact/rephrase selection to 90.59/56.18%,
showing that the frozen cosine ranker must be preserved.

V4 therefore changes only the nonlinear calibration of the frozen rank-one
record.  It cannot change the selected key, writer, value, lifecycle,
candidates, prompts or Qwen checkpoint.  Natural development, test and all
external benchmarks stay sealed until every calibration gate passes.

## Immutable examples and features

Use the exact V2 calibration feature artifact with SHA256
`1d3f89ee7db51208b1d1934d7229a6b1c5d3e302cc2ebc6a67b8c6c09535a140`:
5,144 natural queries plus 646 deterministic in-fold no-match controls across
five session folds.  The natural target is retrieve only when the frozen
cosine top is the active record.  Use the same three equally weighted strata:
correct natural top, wrong natural top and no-match.

The old seed-7301 decomposed-gate linear logit is frozen.  Natural V1 rows,
states and Qwen scores and the audited V2 OOF rows remain immutable.

## Registered residual model and search

Add a residual MLP to the frozen linear logit:

`logit_v4 = logit_frozen + Linear2(GELU(Linear1(feature)))`.

Use no dropout.  Initialize `Linear1` with seed 202610201 and Kaiming uniform;
initialize `Linear2` weights and bias to zero so the initial model exactly
equals the frozen gate.  Train deterministic full-batch weighted BCE plus the
mean squared residual-parameter norm.

Search only:

- hidden width: `16, 32`;
- residual penalty: `0.01, 0.1, 1.0`;
- learning rate: `0.001, 0.005`;
- epochs: `25, 50`;
- probability threshold: `0.10, 0.15, ..., 0.90`.

Emit only five-fold session OOF decisions.  Among configurations satisfying
no-match specificity at least 92% and natural wrong-top specificity at least
85%, maximize correct-top recall, then no-match specificity, wrong-top
specificity and three-stratum balanced accuracy.  Break ties by the smaller
hidden width, larger penalty, lower learning rate, fewer epochs and threshold
closest to 0.5.  Then train one final all-calibration checkpoint.

## Exact Qwen outcome scoring

For each OOF natural query, retrieve uses the unchanged frozen cosine top.
Correct-top retrieval reuses the V1 `oracle_oracle` score; defer reuses V1
`zero`; every retrieved wrong top is explicitly rescored with Qwen.  Use scale
8, fresh KV, the registered 12 candidates, fixed-correct-length final
generation and 20,000 subject-group bootstrap draws.  Compare against audited
V2 OOF outcomes.

## Calibration release gate

All checks must pass:

- OOF correct-top recall at least 92%;
- OOF natural wrong-top specificity at least 85%;
- OOF no-match specificity at least 92%;
- three-stratum balanced accuracy at least 92%;
- candidate accuracy at least 93%;
- gap to `oracle_oracle` at most 5.5 points;
- final candidate accuracy at least 91%;
- final generation exact at least 74%;
- candidate-accuracy gain over V2 has a subject-group bootstrap 95% lower
  bound above zero.

Passing freezes the final residual gate and releases exactly one natural-
development run.  Development must retain at least 90% no-match specificity,
90% candidate accuracy, an oracle gap no greater than seven points and a
positive confidence bound over V1 `oracle_auto`.  Failure keeps development
sealed.  No observed result may be used to relax these thresholds.
