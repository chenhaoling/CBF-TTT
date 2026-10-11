# Counterfactually Reassigned Episodic Binding Result

## Goal and protocol

This matched experiment tested whether fixed query to value mappings let the writer optimize through slow parameter memorization instead of reading the current session memory. For every packed context, four equal length code values were cyclically reassigned among the same four entities. The fixed arm saw only shift 0 on all four rounds; the episodic arm saw shifts 0, 1, 2, and 3 exactly once. Both arms used identical 32/8 train/development worlds, initialization, order seed, 1,024 question exposures, 256 updates, listwise plus 0.25 LM loss, and cap 0.3.

Evaluation scored shifts 0 and 2 under Correct, next-shift Counterfactual, cross-world Wrong, Empty, and Full-KV. It reported four candidate and 16 candidate retrieval. Test remained sealed.

The two GPU run completed at commit `1a152a1`. Eighteen focused server tests and the independent audit passed. Frozen backbone, FP32 writer/optimizer state, data hashes, balanced assignments, targets, evaluation rows, and test sealing were verified.

## Training result

| Arm | First 32 loss | Last 32 loss | First 32 accuracy | Last 32 | Mean update | Peak allocated / reserved |
|---|---:|---:|---:|---:|---:|---:|
| Fixed assignment | 1.5526 | 1.3163 | 23.44% | 36.72% | 0.694 s | 18.821 / 19.053 GiB |
| Episodic assignment | 1.5387 | 1.4316 | 25.00% | 26.56% | 0.680 s | 18.821 / 19.053 GiB |

The fixed arm ended below the four way random loss `log(4)=1.3863`; episodic did not. With matched compute and initialization, the 10.16 point final window gap shows that a repeated fixed assignment materially inflated apparent optimization. This is evidence for a slow parameter shortcut, not yet a proof of the exact internal memorization mechanism.

## Final development result

| Arm / memory | Context4 top 1 | Global16 top 1 | Digit NLL |
|---|---:|---:|---:|
| Fixed / Correct | 22.66% | 3.12% | 2.6283 |
| Fixed / Counterfactual | 23.44% | 3.91% | 2.6372 |
| Fixed / Wrong | 24.22% | 7.81% | 2.6269 |
| Fixed / Empty | 26.56% | 5.47% | 2.7108 |
| Episodic / Correct | 25.00% | 7.03% | 2.6136 |
| Episodic / Counterfactual | 22.66% | 4.69% | 2.6303 |
| Episodic / Wrong | 22.66% | 10.16% | 2.6815 |
| Episodic / Empty | 26.56% | 5.47% | 2.7108 |

For episodic, Counterfactual minus Correct true digit NLL was `+0.0073`, bootstrap 95% CI `[-0.0033, 0.0188]`; the 2.34 point context4 top 1 advantage was far below the 25 point gate. Wrong minus Correct was `+0.0601`, CI `[-0.1263, 0.2585]`, and Empty minus Correct was `+0.0943`, CI `[-0.0057, 0.1933]`. Complete code generation remained 0. Full-KV development code accuracy was 95.31%.

On the train probe, episodic Correct/Counterfactual context4 top 1 was 25.00/24.22% and their NLL difference was effectively zero. The arm therefore failed even to fit causal assignment dependence, rather than only failing held out generalization.

The fixed arm produced a statistically positive but tiny development Counterfactual NLL gap `+0.0099`, CI `[0.0048, 0.0156]`, while Counterfactual top 1 exceeded Correct. This mismatch demonstrates why the registered gate requires both likelihood confidence and ranking separation.

## Conclusion and next step

The registered episodic gate failed. The result supports a narrow but useful mechanism conclusion: fixed entity to value mappings overstate writer learning, while balanced reassignment across optimizer steps removes most of that fit. Reassignment alone did not produce usable current-memory binding.

The next registered experiment pairs opposite assignments inside each optimizer step. Both memories score the same candidates, and a symmetric swap loss directly requires each target to be preferred under its own memory. This cancels the query prior within one gradient calculation and reuses the same forward scores. If that also fails, the next change should be the memory representation or readout rather than more training on fixed assignments.

## Files and compatibility

- `tasks/train_cbf_episodic_binding.py`: token preserving rotations, balanced schedule, training, causal evaluation, and gate.
- `scripts/run_cbf_episodic_binding.sh`: focused tests, two GPU smoke/formal run, summary, and audit.
- `scripts/audit_cbf_episodic_binding.py`: independent reconstruction and artifact verification.
- `tests/test_cbf_episodic_binding.py`: construction, schedule, and gate tests.
- `summary.json`, `execution_audit.json`, `RUN_STATUS.json`, and `git_commit.txt`: compact downloaded results. Raw rows, checkpoints, optimizer state, and logs remain at `/home/ctj/cbf_ttt_episodic_binding_v1`.

The first start was intentionally stopped before any optimizer update after detecting an overly tight BF16 cap audit tolerance; its preflight directory is preserved. The formal restart changed only the artifact tolerance from 0.3001 to 0.305, while the actual cap, data, objective, budget, and efficacy gate remained unchanged.

All functionality is opt in. Baseline model definitions, pretraining, dataset loaders, and default memory behavior are unchanged. The experiment uses eight development worlds, one writer seed, and four cyclic permutations; a positive paper claim would require a fresh permutation family and seed.
