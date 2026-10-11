# Candidate-Ranking Localization Experiment

## 1. Goal

The preceding dynamic binding experiment reduced digit NLL but generated no complete random code from fresh-KV memory. This experiment tested a narrower explanation: perhaps the correct value was already latent in the writer and free generation alone failed. Fixed candidate ranking removes answer termination and open-vocabulary decoding from the decision.

No writer was trained, no checkpoint was selected, and no forgetting label, controller, or test score was produced.

## 2. Fixed protocol

The experiment reused the audited Qwen3-4B `joint_ce` and `joint_binding` checkpoints at 0, 2,048, and 4,096 question exposures. It scored the fixed first eight training worlds as a probe and all 32 development worlds. Correct, twin, Wrong, and Empty memories used fresh attention KV.

Each query received deterministic nested candidates:

- `pair2`: correct and twin values for anchor questions;
- `world5`: the five unique values in the twin world;
- `global16`: `world5`, fixed Wrong-context values, and deterministic same-split negatives.

Primary ranking used teacher-forced digit NLL. The preregistered primary result was final-checkpoint `joint_binding` on development `global16`. Strong binding required Correct top-1 at least 0.50 plus large Correct-versus-Wrong/twin separation and a positive bootstrap lower bound. Weak binding required Correct top-1 at least 0.25 plus a positive NLL gain. Otherwise the interpretation was `no_localized_binding`.

The previous audited Full-KV development code accuracy, 0.9570, was reused as the readability control. Its source summary SHA-256 is `5e5f74e3f44eee886ee4ea6ef5d3efa5854fdffcc1dd3b5239692f93d18fefff`.

## 3. Execution and integrity

The two-GPU run completed on `hku-gpu2` in about 7 minutes 10 seconds.

- Launch-focused tests: 11/11 passed.
- Independent execution audit: passed.
- Every arm/checkpoint contained 320 expected rows: 64 train-probe and 256 development rows.
- The exact batched CBF scorer matched the established individual CBF readout with maximum error 0.0.
- Source checkpoint hashes were unchanged before and after scoring.
- The two arms had identical checkpoint-0 scores.
- Test data was not tokenized or scored.
- Peak CUDA allocated/reserved memory was 10.1881/10.6641 GiB per arm.

Two implementation failures occurred before the formal result and were handled without changing the protocol. The first invocation used a direct script path and failed before model loading because the `tasks` package was not on Python's module path. The runner was changed to `python -m`. The second attempt stopped at the mandatory bridge before writing ranking rows because an approximate functional scorer differed from the established CBF path by about 0.05 NLL, above the fixed `2e-5` tolerance. The tolerance was not relaxed; the evaluator was changed to the exact batched CBF read path. Both failure directories remain on the server for provenance.

## 4. Primary result

### Final `joint_binding` development result

| Candidate set | Chance | Memory | Top-1 | MRR | Mean rank | Best-negative margin | True digit NLL |
|---|---:|---|---:|---:|---:|---:|---:|
| pair2 | 0.5000 | Correct | 0.4688 | 0.7344 | 1.5313 | 0.0042 | 2.5526 |
| pair2 | 0.5000 | twin | 0.5313 | 0.7656 | 1.4688 | -0.0042 | 2.5568 |
| pair2 | 0.5000 | Wrong | 0.5156 | 0.7578 | 1.4844 | -0.0025 | 2.6364 |
| pair2 | 0.5000 | Empty | 0.5000 | 0.7500 | 1.5000 | 0.0000 | 2.6925 |
| world5 | 0.2000 | Correct | 0.2031 | 0.4558 | 3.0391 | -0.3306 | 2.5769 |
| world5 | 0.2000 | twin | 0.2070 | 0.4593 | 3.0117 | -0.3316 | 2.5779 |
| world5 | 0.2000 | Wrong | 0.2227 | 0.4693 | 2.9688 | -0.3221 | 2.6170 |
| world5 | 0.2000 | Empty | 0.1875 | 0.4517 | 2.9883 | -0.1815 | 2.6951 |
| global16 | 0.0625 | Correct | **0.0625** | 0.2108 | 8.1992 | -0.5417 | 2.5769 |
| global16 | 0.0625 | twin | 0.0625 | 0.2114 | 8.1719 | -0.5408 | 2.5779 |
| global16 | 0.0625 | Wrong | 0.0625 | 0.2104 | 8.6328 | -0.5916 | 2.6170 |
| global16 | 0.0625 | Empty | 0.0742 | 0.2167 | 8.3672 | -0.3373 | 2.6951 |

For `global16`, Correct-minus-control comparisons were:

| Control | True digit-NLL gain | 95% source-world bootstrap CI | Top-1 advantage | 95% CI |
|---|---:|---:|---:|---:|
| twin | 0.0011 | [-0.0012, 0.0031] | 0.0000 | [-0.0156, 0.0156] |
| Wrong | 0.0401 | [-0.0892, 0.1669] | 0.0000 | [-0.0430, 0.0430] |
| Empty | 0.1183 | [0.0217, 0.1975] | -0.0117 | [-0.0664, 0.0430] |

The Correct memory improves the likelihood of its true value relative to Empty, but it does not rank that value above explicit alternatives. Correct and twin/Wrong all obtain exactly chance-level `global16` top-1. The Empty condition is numerically higher in top-1. This is evidence for a generic populated-memory or answer-format effect, not entity-specific value retrieval.

## 5. Checkpoint trajectory and CE control

Development `global16` Correct-memory top-1:

| Exposure | joint CE | joint binding |
|---:|---:|---:|
| 0 | 0.0742 | 0.0742 |
| 2,048 | 0.0859 | 0.0938 |
| 4,096 | 0.0820 | 0.0625 |

The binding midpoint produced a small Correct-versus-Wrong top-1 advantage of 0.0430 with bootstrap CI [0.0039, 0.0820], but Correct top-1 remained only 0.0938, below the fixed weak-binding threshold of 0.25. The signal also disappeared at the fixed endpoint. This midpoint is a diagnostic observation, not a selected result.

At the final CE endpoint, Correct top-1 was 0.5000/0.2266/0.0820 for pair2/world5/global16. Its Correct-versus-Wrong global16 digit-NLL gain was -0.0031 with a confidence interval crossing zero. Explicit digit weighting and single-anchor contrast therefore improved NLL calibration without creating stable identity ranking.

## 6. Conclusion and decision

The fixed interpretation is **`no_localized_binding`**. Both the strong and weak preregistered rules failed.

The result rules out the useful version of the “correct value is present but free generation hides it” explanation for this writer. The dominant failure occurs before open-ended decoding: the current memory representation/readout does not reliably bind an entity query to its session-specific value. More examples, more training steps, or a forgetting controller would not repair that missing prerequisite by themselves. The endpoint becoming worse than the midpoint also argues against simply extending the same training schedule.

This conclusion does not invalidate earlier evidence about an effective forgetting policy under a setting where fast memory already carries usable content. It says that the current dynamic random-value writer is not an adequate substrate on which to evaluate that policy.

## 7. Code and artifact mapping

- `tasks/evaluate_cbf_candidate_ranking.py`: deterministic candidate construction, exact batched CBF scoring, bridge checks, row output, metrics, and bootstrap aggregation.
- `scripts/audit_cbf_candidate_ranking.py`: independent reconstruction of candidates, ranks, summaries, hashes, initialization equality, and test sealing.
- `scripts/run_cbf_candidate_ranking.sh`: focused tests, two-GPU arm execution, summary, audit, and terminal state.
- `tests/test_cbf_candidate_ranking.py`: candidate nesting, ranking, margins, aggregation, interpretation, and exact batch-read behavior.
- `cbf_ttt/runtime.py`: permits batched fresh-KV reads only when collection is disabled; memory collection remains batch-one and the default baseline path is unchanged.
- `CANDIDATE_RANKING_PLAN.md`: frozen protocol and fixed interpretation thresholds.

This directory contains compact manifests, summaries, audit output, status, and hashes. Raw per-query ranking JSONL and failure traces remain on `hku-gpu2` under `/home/ctj/cbf_ttt_candidate_ranking_v1` and the two suffixed failure directories. The downloaded transport archive is ignored by Git.

The transport archive SHA-256 is `09587b9ae71f0ccf98b872f7a23ef33177b7106fb326c2dd14dfcf2809955399`.

## 8. Reproduction

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_candidate_ranking_v1 \
SOURCE=/home/ctj/cbf_ttt_dynamic_binding_v1/formal \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
bash scripts/run_cbf_candidate_ranking.sh
```

Independent audit:

```bash
/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
  -m scripts.audit_cbf_candidate_ranking \
  --root /home/ctj/cbf_ttt_candidate_ranking_v1
```

## 9. Baseline compatibility

The experiment is opt-in and does not change baseline configs, datasets, training entry points, optimizer behavior, or saved writers. The runtime extension preserves batch-one behavior and rejects batched memory collection. It only allows batch scoring on the read-only `cbf_collect=false` path. No dependency was added.

## 10. Next experiment

The next experiment should change the binding mechanism before returning to forgetting-policy training:

1. Keep joint-context training, the same deterministic worlds, and Correct/twin/Wrong/Empty controls.
2. Supervise all four facts with entity-conditioned associative retrieval rather than applying contrast only to the changing anchor.
3. Add within-world value permutations as hard negatives so success requires matching each entity to its own value.
4. Run a small 32-train/8-dev pilot and use candidate ranking as the early gate. Require Correct to separate from twin, Wrong, and within-world permutation with a positive source-world confidence bound.
5. Only after that gate passes, run a fresh-seed formal exact-generation experiment. Resume the forgetting-only controller and counterfactual label construction only after the formal content gate passes.

## 11. Risks and limits

- This is one semantic seed with 32 development worlds; it can reject a clearly weak writer but cannot establish broad generalization.
- Candidate ranking is easier than free generation. Passing it would be necessary evidence, not sufficient proof of exact generation.
- The candidate pool contains fixed deterministic negatives; performance may vary with harder values or more candidates.
- Full-KV and fresh-KV test different information paths. High Full-KV performance establishes prompt readability but does not guarantee writer capacity.
- The conclusion applies to this writer architecture, loss, and training budget. It does not rule out the broader forgetting-only research direction.
