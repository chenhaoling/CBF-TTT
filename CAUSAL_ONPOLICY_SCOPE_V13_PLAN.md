# Causal on-policy scope V13 preregistration

## Evidence

V12 significantly improved query accuracy over fixed retain by 1.46 points
(`95% CI [0.99,1.96]`) but reached only 93.50% hard-retain and 80.00% update
accuracy. Of 157 update errors, 121 were false retains and 36 were wrong
deletions. The V11 candidate resolver remains frozen at approximately 97%
top-1; V13 changes only the retain/local decision.

## Method

For each outer fold:

1. freeze the V11 token-span resolver, V5 relation heads, Qwen checkpoint, and
   old query readout;
2. reconstruct the frozen V12 outer-fold controller and roll it out on the
   three inner-training folds;
3. collect the states actually induced by that controller; a local label is
   positive only when the conflicting chain is still present, otherwise the
   safest available label is retain;
4. expose relation-posterior geometry in addition to V12's ten physical
   features: query/key confidence and entropy, argmax agreement, posterior
   cosine, and Jensen-Shannon divergence;
5. train a rank-16 nonlinear gate on the union of oracle and policy-induced
   states, using three seeds;
6. select epoch and threshold through causal replay of the untouched inner fold,
   then refit on all non-outer states and replay the outer fold once.

The registered `oracle_mlp` ablation uses the same nonlinear gate and augmented
features but no on-policy states. It distinguishes representation capacity from
data aggregation. No candidate score, subject-span head, relation classifier,
or query readout is retrained.

## Gate

The primary on-policy arm passes only if:

- static candidate top-1 remains at least 96%;
- overall causal action accuracy is at least 95%;
- hard-retain is at least 95%;
- primary-update is at least 90%;
- query accuracy is within 0.5 points of oracle local;
- query accuracy is not materially worse than V12: the paired-bootstrap lower
  bound versus V12 must exceed -0.2 points.

V13 is the final CounterFact gate-architecture iteration. A failure stops
CounterFact controller tuning and retains V12's significant causal result with
its limitation. A pass freezes the causal recipe for external-domain
validation. LongMemEval-S and LoCoMo remain sealed until this decision.
