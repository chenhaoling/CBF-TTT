# CounterFact zero-shot forgetting transfer

## Registered question

Test whether the ZsRE-trained forgetting controller transfers without retraining,
threshold tuning, or feature selection to official ROME CounterFact prompts. The
frozen controller always writes the current fact and decides only whether to
delete an old session-memory item.

The official 21,919-row file produced 785 unique same-subject units with two
different relations, packed into 197 sessions and 2,355 update decisions. The
construction used no model score. The controller, 13 input features, Qwen3-4B
1B-token checkpoint, readout seeds, and ZsRE development-selected threshold
0.675 were frozen before CounterFact scoring.

## Audited result

| Policy | Query accuracy | Action accuracy | Initial retain | Hard retain | Counterfactual update |
|---|---:|---:|---:|---:|---:|
| Learned, frozen zero-shot | 48.12% | 67.77% | 93.89% | 78.73% | 30.70% |
| Cosine local | 43.45% | 56.26% | 85.22% | 52.99% | 30.57% |
| Fixed retain | **49.44%** | 66.67% | 100.00% | 100.00% | 0.00% |
| Fixed global | 16.97% | 8.37% | 25.10% | 0.00% | 0.00% |
| Fixed top deletion | 16.97% | 8.37% | 25.10% | 0.00% | 0.00% |
| Oracle local deletion | 51.93% | 100.00% | 100.00% | 100.00% | 100.00% |

Learned forgetting improved over the registered cosine baseline by **4.57 query
points**, with subject-paired 95% CI **[3.90, 5.24]**, and **11.51 action
points**, CI **[9.51, 13.50]**. Both relative checks passed. The registered
transfer gate failed because overall action, hard retain, and update accuracy
were below their 90%, 90%, and 85% floors. Fixed retain also exceeded learned
query accuracy by 1.32 points, so this run does not establish useful zero-shot
forgetting on CounterFact.

The independent audit recomputed all six policies, aggregates, paired intervals,
source and raw hashes, zero-shot invariants, and the 785-unit count. Every audit
check passed. Runtime was 216.05 seconds on one RTX 5090, with 9.026/12.580 GiB
peak allocated/reserved memory.

## Failure attribution

The hash-verified post-hoc diagnostic did not fit or rescore a method. The frozen
delete probability retains useful cross-domain signal (local-versus-retain ROC
AUC 0.759), but its best fixed-trajectory binary threshold reaches only 72.61%
and therefore cannot close the reliability gap. This grid is noncausal because
it does not replay the memory state and is diagnostic only.

At the frozen threshold, the controller attempted deletion on 62.55% of update
events, but the cosine-selected candidate was the correct old chain on only
49.08% of those attempts. Hard-retain deletion was also triggered on 21.27% of
events. The main bottleneck is thus the combination of domain-shifted conflict
calibration and unreliable candidate identity, rather than the scalar threshold
alone.

## Supported conclusion and next experiment

The result supports a narrow claim: the ZsRE-trained state signal transfers
better than a cosine threshold, with statistically robust gains. It rejects the
stronger claim that the complete frozen policy transfers reliably.

CounterFact is now observed and becomes development-only evidence. The next
method must separate (1) pairwise old-memory candidate ranking from (2) the
retain/delete decision, preserve forgetting-only updates, and fuse the learned
pair score with the strong cosine prior instead of replacing it. Method selection
may use ZsRE plus observed CounterFact, while the final claim requires a new
source-disjoint temporal-memory benchmark such as LoCoMo or LongMemEval-S.

