# Native EOS-Safe Layerwise Fast-Weight Admission V4

## 1. Goal and frozen protocol

V4 tests whether decision-time activation support, current loss, uncertainty,
and native fast-weight geometry can select a useful single-layer update while
rejecting harmful updates. The Qwen3-4B backbone and the original
`ttt_conv/ttt_proj` writers are frozen. No plaintext history, fact identity,
answer, future outcome, mechanism label, or oracle action enters the
controller.

The protocol was frozen in
`NATIVE_EOS_SAFE_LAYERWISE_WRITE_GATE_V4_PLAN.md` before train or development
scoring. It uses the independently audited, source-disjoint V3 corpus with 384
train, 96 development, and 96 sealed-test source groups; every group has three
sparse boundaries. V4 was stopped on development, so sealed test was never
opened.

## 2. Native action and controller

Every boundary branches from the same native parent fast-weight state into:

- reject-all;
- seven singleton actions admitting only layer 0, 6, 12, 18, 24, 30, or 35;
- full admit, used as a fixed baseline.

All actions use fresh KV and the same future queries. The 1,854-dimensional
decision input contains current prediction statistics, per-layer fast-weight
geometry, fixed 128-dimensional activation projections, support centroid and
diagonal variance, recent-support cosine and likelihood, numeric position,
and bias. Online features and future-derived labels are physically separate
and bound row-by-row with SHA256.

For each singleton, the offline targets are benefit, protected-NLL change,
protected-accuracy harm, and EOS-increase harm. Three frozen seeds train each
of full, no-support, and support-only variants. At deployment, a layer is
eligible only if predicted benefit is positive, predicted protected-NLL change
is nonpositive, and both harm probabilities are below 0.5. The eligible layer
with maximum predicted benefit is selected; ties and empty sets reject.

## 3. Implementation and modified files

The V4 implementation is opt-in:

- `tasks/native_eos_safe_layerwise_write_gate_v4.py`: native branching,
  feature/label separation, EOS-safe oracle, resource collection, split
  release verification, and summaries;
- `tasks/train_native_eos_safe_layerwise_write_gate_v4.py`: fixed three-seed,
  three-ablation controller training and development release metrics;
- `tasks/evaluate_native_eos_safe_layerwise_write_gate_v4.py`: conditional
  one-shot sealed-test evaluation using frozen checkpoints and thresholds;
- `scripts/audit_native_eos_safe_layerwise_write_gate_v4.py`: independent
  feature/label/action/resource and split-dependency audit;
- `scripts/audit_train_native_eos_safe_layerwise_write_gate_v4.py`: checkpoint
  reload and independent prediction/action/metric/bootstrap reconstruction;
- `scripts/audit_evaluate_native_eos_safe_layerwise_write_gate_v4.py`: sealed
  test reconstruction, present but not executed because development failed;
- `scripts/run_native_eos_safe_layerwise_write_gate_v4.sh`: two-GPU collector
  with train-to-development and development-to-test release locks;
- `scripts/run_train_native_eos_safe_layerwise_write_gate_v4.sh`: controller
  training plus mandatory audit;
- `scripts/run_evaluate_native_eos_safe_layerwise_write_gate_v4.sh`: conditional
  sealed-test evaluation plus mandatory audit;
- `tests/test_train_native_eos_safe_layerwise_write_gate_v4.py`: feature
  ablations, four-head output, EOS veto, release-lock, and registered-gate
  tests.

The first test launch would atomically create `sealed_test_release.json` with
`O_EXCL`, binding the label directory and controller summary/audit hashes.
That marker does not exist in this experiment.

## 4. Execution and resource results

All labels were generated on two RTX 5090 GPUs with four shards and two
workers per GPU.

| Split | Groups | Boundaries | Mean seconds/label | Peak allocated/worker | Peak reserved/worker |
|---|---:|---:|---:|---:|---:|
| Train | 384 | 1,152 | 17.876 | 10.765 GiB | 11.475 GiB |
| Development | 96 | 288 | 18.304 | 10.762 GiB | 11.475 GiB |

Every shard reports `backbone_unchanged=true`. The train and development
collectors, summaries, and independent audits passed all execution checks.

## 5. Oracle/headroom results

| Split | Singleton headroom | Reject J | Full-admit J | EOS-safe oracle J | Oracle gain | Group-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Train | 94.70% | 7.94056 | 12.28404 | 7.85342 | 0.08713 | [0.07955, 0.09570] |
| Development | 95.14% | 7.88854 | 12.15857 | 7.80464 | 0.08390 | [0.07236, 0.09655] |

The development safe oracle writes on 271/288 boundaries. It satisfies the
registered protected-accuracy, protected-NLL, and EOS conditions. Thus the
native writer contains useful state-dependent singleton updates, even though
the all-layer update remains strongly harmful.

## 6. Development baselines

| Policy | J | Write rate |
|---|---:|---:|
| always-write/full-admit | 12.15857 | 100.0% |
| never-write | 7.88854 | 0.0% |
| loss-threshold | 7.88854 | 0.0% |
| norm-threshold | 7.88854 | 0.0% |
| support-in | 7.88871 | 100.0% |
| novelty-only | 7.89252 | 100.0% |
| duplicate-skip | 7.88947 | 33.3% |
| EOS-safe oracle | 7.80464 | 94.1% |

The train-fitted scalar thresholds collapse to never-write. The three direct
support heuristics do not improve J over never-write.

## 7. Learned-controller results

| Seed | Full J | Gain vs never | 95% CI vs never | Oracle gap closed | Write rate | No-support J | Full better than no-support |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 2026101801 | 7.87537 | 0.01317 | [0.00514, 0.02278] | 15.70% | 81.25% | 7.87832 | yes |
| 2026101802 | 7.88354 | 0.00501 | [-0.00951, 0.01574] | 5.97% | 80.90% | 7.87946 | no |
| 2026101803 | 7.87898 | 0.00956 | [-0.00008, 0.02034] | 11.40% | 81.25% | 7.87679 | no |

All three full seeds have a better point estimate than both fixed baselines and
all three easily beat always-write with strictly positive paired bootstrap
intervals. Only seed 2026101801 has a strictly positive interval versus
never-write. All three fail the registered 30% oracle-gap threshold, and two
of three fail the required full-versus-no-support ablation.

Safety passes for all full seeds. Relative to never-write, protected accuracy
changes by -0.347, -0.116, and -0.347 percentage point; protected NLL changes
by +0.00019, +0.00832, and +0.00361; EOS changes by +0.208, +0.486, and +0.486
percentage point. These are within the registered limits.

## 8. Formal decision and claim boundary

Development release is **failed** for all three seeds. The exact failed gates
are:

- seed 2026101801: oracle-gap closure below 30%;
- seeds 2026101802 and 2026101803: bootstrap lower bound versus never-write is
  not strictly positive, oracle-gap closure is below 30%, and full does not
  beat the matched no-support model.

Therefore:

- `development_released_test=false`;
- sealed-test labels, predictions, metrics, and audit were not created;
- `sealed_test_release.json` was not created;
- no formal test claim is made.

The result supports a narrow statement: a learned risk-aware controller can
recover a small fraction of layerwise oracle utility while preserving the
registered safety metrics. It does not show a robust support-specific benefit
across seeds and does not meet the preregistered adaptive-TTT success criteria.

## 9. Audit evidence

The compact immutable evidence is under `artifacts/` in this directory:

- train/development summaries and independent audits;
- eight per-shard resource/hash sidecars;
- controller training summary and independent audit.

The controller auditor reloaded all nine checkpoints and reconstructed every
development prediction, action, metric, paired 20,000-draw source-group
bootstrap, baseline, and release decision. It reports `passed=true`,
`test_released=false`, and `test_opened=false`. The complete 27-test native
fast-weight focused suite passed on the remote conda environment.

## 10. Reproduction commands

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
export PYTHON_BIN=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python
export MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
export DATA_ROOT=/home/ctj/cbf_ttt_native_layerwise_write_gate_v3_data_20261016
export V3_ROOT=/home/ctj/cbf_ttt_native_layerwise_write_gate_v3_20261016
export OUTPUT_ROOT=/path/to/new_v4_label_root

STAGE=train NUM_SHARDS=4 GPU_COUNT=2 \
  bash scripts/run_native_eos_safe_layerwise_write_gate_v4.sh
STAGE=development NUM_SHARDS=4 GPU_COUNT=2 \
  bash scripts/run_native_eos_safe_layerwise_write_gate_v4.sh

LABEL_ROOT="$OUTPUT_ROOT" OUTPUT_ROOT=/path/to/new_v4_controller_root \
  bash scripts/run_train_native_eos_safe_layerwise_write_gate_v4.sh
```

The sealed-test runner must not be used for this result because development did
not release test. All runners refuse to overwrite existing artifacts.

## 11. Compatibility, limitations, and next work

No baseline configuration, Qwen weight, original writer source, training entry
point, or prior checkpoint was changed. All new paths are opt-in.

Limitations and follow-up questions:

- the controller closes only 6--16% of the oracle gap;
- the support ablation is inconsistent, so activation support has not shown a
  reproducible incremental contribution beyond loss/uncertainty/geometry;
- candidate-NLL fresh-KV evaluation does not establish open-ended generation
  quality;
- the controlled 4,096-token construction differs from natural long-form
  sessions;
- a new method may improve calibration or exploit sequence-level structure,
  but it requires a newly frozen protocol and fresh development evidence;
- the current sealed test must remain untouched and cannot be used for that
  redesign.
