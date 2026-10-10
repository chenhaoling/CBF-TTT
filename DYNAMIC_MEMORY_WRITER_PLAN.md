# Dynamic Memory Writer Pilot (preregistered before GPU execution)

## Question

The previous forgetting-only validation failed its prerequisite: with fresh attention KV, a correct fast memory was not reliably better than a wrong or empty fast memory. This pilot asks whether the original Qwen3-4B writer path can first learn to store randomly assigned, previously unseen session facts. It also checks whether supervising all four questions that share one write context reduces cross-fact optimization interference compared with one-question updates.

No forgetting controller, counterfactual forgetting labels, or test-set model selection is allowed in this stage. A later forgetting experiment is conditional on the content gate below.

## Frozen design

- **Semantic worlds:** seed `20261012`; train/dev/test = `64/16/16` disjoint source groups. Each group has two twin sessions. Only the anchor value differs between twins; three protected values and the question strings are identical.
- **Natural background:** one unique document per group, alternated between `fineweb_edu` and `longcrawl64`, read from `/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl`. The first tokens are combined with the dynamic fact suffix into exactly 4096 tokens.
- **Sealed test:** test semantic text and truth are materialized for later use, but the packer reads and tokenizes train/dev only. No test model score is produced in this pilot.
- **Model:** audited official Qwen3-4B base at `/home/ctj/models/Qwen3-4B`; frozen backbone; seven original In-Place TTT conv/proj writer pairs are trainable; BF16 forward and FP32 writer/AdamW state.
- **Optimization:** AdamW, learning rate `1e-7`, weight decay 0, global clip 1, writer initialization seed 301, order seed 503, four rounds.
- **Matched exposure:** 64 train groups × 2 contexts × 4 questions × 4 rounds = 2048 question exposures per arm.
  - `sequential`: one question per optimizer update, 2048 updates and 2048 writes.
  - `joint_context`: four questions sharing one differentiable memory, mean CE, 512 updates and 512 writes.
- **Fixed checkpoints:** 0, 1024, and 2048 question exposures. The endpoint is primary; the midpoint diagnoses trajectory and is not used to select a checkpoint.
- **Evaluation:** fixed first eight train groups as a training probe and all 16 dev groups. Every row uses fresh attention KV for Correct, Wrong, Empty, and twin-memory readout. Full-KV is a readability control. Wrong memory comes from the next group in sorted split order.
- **Smoke:** the same packed corpus restricted to 8 train and 2 dev groups, one round. Both arms run. Smoke only checks execution, accounting, memory, and audit logic; its accuracy is not a stage decision.

## Preregistered content gate

An arm passes only at its fixed 2048-exposure endpoint when all conditions hold on the 16 dev source groups:

1. Full-KV `code_correct >= 0.80`.
2. Correct-memory `code_correct >= 0.60`.
3. Correct-memory code accuracy exceeds Wrong by at least 0.20.
4. Correct-memory code accuracy exceeds Empty by at least 0.20.
5. For both Wrong−Correct and Empty−Correct digit NLL, the source-group mean gain is at least 0.10 and the fixed 5000-draw source-group bootstrap 95% lower bound is above zero.

The stage passes if either arm passes all conditions. Training-probe accuracy is diagnostic and cannot rescue a failed dev gate. The two-arm comparison is descriptive because this pilot has one initialization/order seed and different numbers of optimizer updates and writes.

## Decisions after this stage

- If neither arm passes, stop. Diagnose writer representation/capacity or broaden independent dynamic-world training; do not train a forgetting controller on unreadable memory.
- If at least one arm passes, freeze the passing writer recipe and run the next oracle experiment over retain, local forgetting, and global reset actions. Only after an oracle advantage is established should controller features and labels be collected.

## Reproduction

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_dynamic_memory_v1 \
MODEL=/home/ctj/models/Qwen3-4B \
CORPUS=/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
bash scripts/run_cbf_dynamic_memory.sh
```

