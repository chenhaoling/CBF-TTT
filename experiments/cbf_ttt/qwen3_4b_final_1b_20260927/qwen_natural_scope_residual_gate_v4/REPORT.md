# Qwen natural residual scope gate V4

V4 preserved the frozen cosine rank-one record and added a zero-initialized
two-layer residual MLP to the original decomposed-gate logit.  It used the
immutable V2 examples and features, five-fold session OOF prediction and the
registered no-match/wrong-top constraints.

The selected width-32 residual used penalty 1.0, learning rate 0.005, 50 epochs
and threshold 0.25.  It reached 95.88% correct-top recall, 86.01% wrong-top
specificity and 93.81% no-match specificity.  End-to-end candidate accuracy
was 94.58%, rephrase accuracy 90.55%, final candidate accuracy 93.31% and final
generation exact 77.62%.  It improved over V2 by 3.02 points with subject-group
bootstrap 95% CI `[2.07, 4.03]`.

The calibration release nevertheless stopped because three-stratum balanced
accuracy was 91.899%, narrowly below its frozen 92% gate.  Inspection of the
already produced search revealed that the selector constrained two specificity
metrics before maximizing recall but did not constrain the balanced-accuracy
metric later used for release.  Eleven existing records satisfied every
decision gate.  V5 preregistered a constrained selection correction without
adding a model, feature, search value or relaxed threshold.
