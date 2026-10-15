# Preregistered Qwen natural scope-transfer repair V2

## Motivation and frozen evidence

The audited V1 calibration run contains 5,144 natural-memory queries from 65
sessions.  Oracle lifecycle plus oracle selection reaches 98.31% candidate
accuracy, while the same lifecycle with the frozen decomposed gate reaches
87.05%.  A score-free reconstruction made after V1 terminated shows that the
frozen ranker puts the active record first on 96.09% of queries, but the gate
retrieves on only 74.30%.  Of the 1,322 deferred queries, 1,121 have the correct
record at rank one.  The failure is concentrated in rephrased auxiliary facts.

This protocol repairs that calibration-to-natural-memory gate transfer.  It is
an enabling experiment, not the paper's forgetting contribution.  The natural
development split, counterfactual test split and all external benchmarks stay
sealed until this protocol has selected and frozen one reader.

## Immutable inputs

- the audited natural-memory V1 calibration rows and lifecycle snapshots;
- the V1 natural-memory corpus and its calibration/development split hashes;
- Qwen3-4B checkpoint at step 81,381;
- contextual transforms SHA256
  `6f6a696005185d80d6b8f468fa6442146dc8d9683e232e7a591676ba9515c378`;
- decomposed-scope checkpoint SHA256
  `0ed4dab8b77fd0fb91fdd49720c783261ab2fd5c64cd705ebe36346d99615619`;
- reader seed 7301 and natural-value scale 8.

No writer, value residual, candidate set, forgetting action, prompt, split or
Qwen parameter may change in this experiment.

## Calibration examples

At every registered V1 oracle-lifecycle snapshot, reconstruct the rank-one
record and the exact frozen decomposed-gate feature: selected pair feature plus
the four standardized ensemble-geometry features.  The target is one only when
the rank-one record is the active chain and active version.  Thus a memory-
relevant query whose rank one is wrong teaches the gate to defer rather than
amplify the wrong value.

Add two deterministic no-match controls per session checkpoint, one exact and
one rephrased query drawn from a different session in the same cross-validation
fold.  Donor and memory sessions are selected by SHA256 order and have disjoint
subjects and targets.  Their target is defer.  Donors are constructed within a
fold, so no donor text crosses the train/validation boundary.

Use five deterministic session folds.  For each held-out fold, train only on
the other four folds and emit out-of-fold decisions for every natural query and
no-match control.  Fold membership, examples and features are written with
hashes before training.

## Model and registered search

Use a single linear binary gate with the same feature dimension as the frozen
decomposed gate.  Initialize from its seed-7301 weights.  Optimize weighted BCE
plus squared distance from the frozen weights.  Search only:

- anchor weight: `0.01, 0.1, 1.0, 10.0`;
- learning rate: `0.001, 0.005`;
- epochs: `25, 50, 100`;
- decision threshold: `0.10, 0.15, ..., 0.90`.

Select lexicographically among configurations satisfying both out-of-fold
specificity constraints: maximize natural correct-top recall, then no-match
specificity, then natural wrong-top specificity, then use the larger anchor,
lower learning rate, fewer epochs and threshold closest to 0.5.  If no
configuration satisfies the constraints, stop without development.

Train one final gate on all calibration examples with the selected
hyperparameters.  It is the only reader allowed on natural development.

## Calibration gates

All checks must pass:

- out-of-fold correct-top recall at least 90%;
- out-of-fold no-match specificity at least 92%;
- out-of-fold natural wrong-top specificity at least 85%;
- out-of-fold balanced accuracy at least 91%;
- exact Qwen candidate accuracy for oracle lifecycle plus the out-of-fold V2
  reader at least 90%;
- its gap to `oracle_oracle` at most six percentage points;
- final candidate accuracy at least 88%;
- final fixed-length generation exact match at least 70%;
- the subject-group bootstrap 95% lower bound for candidate-accuracy gain over
  the frozen V1 `oracle_auto` reader is above zero.

Top-correct retrieval reuses the already scored `oracle_oracle` residual and
defer reuses the already scored `zero` residual.  Any out-of-fold decision that
retrieves a wrong top record must be scored explicitly with Qwen; it cannot be
imputed as an error.  Use 20,000 deterministic subject-group bootstrap draws.

## Release and failure rules

Passing calibration freezes the final gate checkpoint and releases exactly one
natural-development run.  Development must retain no-match specificity of at
least 90%, candidate accuracy of at least 88%, an oracle gap of at most eight
points, and a positive bootstrap lower bound over V1 `oracle_auto`.

If the OOF specificity constraints fail, improve the gate using calibration
only.  If specificity passes but end-to-end candidate accuracy fails, repair
rank-one selection using calibration only.  Do not change the forgetting
controller until this reader is frozen.  Natural test and external benchmarks
remain sealed regardless of a calibration failure.
