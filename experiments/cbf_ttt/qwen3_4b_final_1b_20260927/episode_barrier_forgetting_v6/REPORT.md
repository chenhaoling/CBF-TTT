# Episode-barrier forgetting V6

## Status

Completed and independently audited. The pooled OOF prerequisite failed, so
subject-disjoint development was not loaded or scored. Per the registered stop
rule, gate loss and threshold work on this constructed ZsRE corpus ends here.
LongMemEval-S remained locked and LoCoMo remained sealed.

## OOF result

| Metric | V5 BCE | V6 barrier |
|---|---:|---:|
| Action accuracy | 99.27% | 99.56% |
| Strict pair state | 83.90% | 83.43% |
| Stale deletion | 83.99% | 83.52% |
| Current retention | 99.81% | 99.81% |
| Non-target retention | 99.70% | 99.98% |
| Hard-retain action accuracy | 99.86% | 100.00% |

V6 pair state by length was 85.02/83.52/82.40/82.77%. The fixed-margin
episode objective increased already-high safety but reduced update recall by
about 0.47 points. It did not pass the unchanged 85% stale/pair prerequisite.

## Supported conclusion

The failure is now well localized. Scope/value factorization reliably locates
the stale session, and nested calibration almost eliminates false deletion.
The remaining gap is conservative action recall under cross-subject shift.
Neither independent weighted BCE nor an update-versus-max-retain barrier closes
that gap under the fixed protocol. Further threshold, margin, loss-weight, or
epoch tuning on the same constructed corpus would be post hoc and is stopped.

The next experiment must introduce an independent natural long-session
training/development source while preserving LoCoMo as confirmation. The paper
can already support the factorized candidate mechanism and quantify the
safety/recall frontier, but it cannot yet claim externally validated end-to-end
long-session forgetting.

## Reproducibility

- 1,068 validation episodes / 42,720 actions;
- all validation subjects excluded from corresponding train and calibration
  folds;
- development files absent;
- independent audit: all eleven checks passed;
- summary SHA-256: `e46655d631440f0c0fc21452309e8bc9c198c9400043e811069f817b439d380f`;
- full OOF trace remains hash-covered at
  `/home/ctj/cbf_ttt_episode_barrier_forgetting_v6` on `hku-gpu2`.
