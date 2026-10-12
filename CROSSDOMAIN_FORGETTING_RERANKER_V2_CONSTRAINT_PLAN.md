# V2 constraint-calibration diagnostic

V2 is observed development evidence. Its threshold objective maximized the
smaller of hard-retain and update accuracy and selected 0.20–0.25, causing
irreversible false deletions and causal rollout collapse. The already saved
inner-OOF grids show that four of five folds contain thresholds satisfying the
registered hard-retain (85%) and update (75%) floors simultaneously.

Without changing any ranker or gate weight, select each fold threshold by:

1. among points with hard-retain at least 85% and update at least 75%, maximize
   overall action, then query-independent update accuracy, then proximity to 0.5;
2. if none is feasible, minimize the sum of the two threshold deficits, then
   maximize overall action and proximity to 0.5.

This frozen rule selects outer-fold thresholds 0.50, 0.45, 0.40, 0.45, and 0.40
from inner OOF only. Re-run the complete causal CounterFact trajectories and the
unchanged development gate. CounterFact remains observed development data; this
diagnostic cannot become a final held-out result. If the gate passes, freeze the
constraint-calibrated recipe for LongMemEval-S. If it fails because outer causal
metrics diverge from inner oracle metrics, proceed to on-policy state collection
and causal threshold selection. LongMemEval-S and LoCoMo remain unscored.

