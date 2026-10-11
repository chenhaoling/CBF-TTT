# All-Fact Associative Binding Pilot

## Question

The candidate-ranking diagnostic showed that the current writer lowers true-value NLL relative to Empty memory but does not bind an entity query to its value. This pilot tests whether direct all-fact supervision against a within-context value permutation can create that binding.

No forgetting label, forgetting controller, checkpoint selection, or test score is allowed.

## Fixed arms and data

- Reuse the audited `dynamic_binding_v1` packed data and frozen Qwen3-4B backbone.
- Select the first 32 train worlds and first 8 development worlds. Keep the semantic test split sealed.
- Use four rounds, 1,024 question exposures, 256 optimizer updates, and checkpoints at 0/512/1,024 exposures.
- Both arms use the same initialization, context order, digit-weighted positive loss, optimizer, learning rate, and exactly two differentiable context writes per update.
- `anchor_twin`: the previous objective, with one contrast for the changing anchor against its twin context.
- `all_fact_permutation`: four contrasts, one per queried fact, against a context that keeps the same background, entities, and value multiset but cyclically permutes the four queried values.

For each contrasted question:

```text
contrast = softplus(0.2 + correct_memory_digit_nll - negative_memory_digit_nll)
loss = balanced_positive_loss + 0.5 * mean(contrast)
```

## Evaluation and fixed gate

At every fixed checkpoint, evaluate the first eight train worlds and all eight development worlds under Correct, within-context permutation, twin, Wrong, Empty, and Full-KV conditions. Record complete-code generation metrics and fixed 2/5/16 candidate rankings.

The final `all_fact_permutation` development `global16` result is primary. The pilot passes only if:

- Full-KV complete-code accuracy is at least 0.80;
- Correct top-1 is at least 0.25;
- Correct top-1 exceeds both permutation and Wrong by at least 0.10;
- permutation-minus-Correct and Wrong-minus-Correct true digit-NLL gains both have source-world bootstrap 95% lower bounds above zero.

The fixed endpoint is used for the decision. Midpoint results are diagnostic and cannot select a checkpoint. Passing this ranking gate authorizes a fresh-seed formal generation experiment; it does not itself authorize forgetting-policy training.

## Command

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_associative_binding_v1 \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_associative_binding.sh
```

## Completed result

The two-GPU pilot completed on 2026-10-11 and the independent audit passed. Both arms completed 256 updates and 1,024 exposures without OOM, NaN, or a safety stop; test remained sealed.

The fixed final all-fact development `global16` Correct top-1 was 0.0469, versus 0.0625 for permutation, 0.0469 for Wrong, and chance 0.0625. Permutation-minus-Correct digit-NLL gain was 0.0017 with 95% CI [-0.0048, 0.0082]. Complete-code accuracy remained zero. The gate failed.

At the midpoint the permutation NLL gain was 0.0125 with CI [0.0079, 0.0183], but top-1 did not change and the effect disappeared at the fixed endpoint. The training permutation contrast also remained flat, from 0.7989 in the first 32 updates to 0.8072 in the last 32. The next experiment should use entity-conditioned candidate selection as the primary listwise objective rather than a 0.5-weight auxiliary contrast. Full results are in [`experiments/cbf_ttt/associative_binding_v1/REPORT.md`](experiments/cbf_ttt/associative_binding_v1/REPORT.md).
