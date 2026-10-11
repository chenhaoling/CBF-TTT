# Candidate-Aligned Listwise Binding Experiment

## Question

The all-fact permutation auxiliary loss did not decrease during training and did not improve candidate ranking. This experiment makes entity-conditioned value selection the primary objective: under one Correct memory, every query must choose its own value from the four values stored in that context.

## Fixed comparison

- Reuse the audited dynamic packed data, the first 32 train worlds and first 8 development worlds, and the frozen Qwen3-4B backbone.
- `retrieval_only`: mean four-query listwise cross-entropy only.
- `retrieval_plus_lm`: the same listwise loss plus `0.25 * balanced_language_model_loss`.
- Both arms use one context write, four batched four-candidate query reads, the same initialization/order/optimizer, four rounds, 1,024 question exposures, 256 updates, and checkpoints 0/512/1,024.
- Candidate scores are negative digit NLL. Candidate values are exactly the four queried values from the current context. The correct target is the entity's assigned value.
- Run a real one-update GPU smoke for each arm before formal execution. Stop on nonfinite gradients, OOM, or `||delta W||/||W|| > 1`.

## Evaluation and decision

Reuse exact generation plus `global16` ranking under Correct, within-context permutation, twin, Wrong, Empty, and Full-KV conditions. Test remains sealed.

The fixed final `retrieval_plus_lm` development gate is unchanged: Full-KV code accuracy at least 0.80, Correct `global16` top-1 at least 0.25, Correct top-1 at least 0.10 above permutation and Wrong, and both corresponding digit-NLL bootstrap lower bounds above zero.

Additional optimization sanity checks are required:

- final train-probe Correct top-1 must exceed permutation by at least 0.10;
- mean training four-way listwise accuracy over the last 32 updates must exceed 0.50;
- final retrieval loss must be lower than the initial chance loss `log(4)`.

No checkpoint selection is allowed. Passing authorizes a fresh-seed formal generation study, not forgetting-controller training.

## Command

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_listwise_binding_v1 \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_listwise_binding.sh
```

