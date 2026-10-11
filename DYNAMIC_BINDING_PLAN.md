# Dynamic Value-Binding Writer Experiment

## Preregistered question

The 64/16-world dynamic-memory experiment showed that joint-context training reduced answer NLL but produced zero correct random codes on dev. This experiment tests whether the missing signal is value binding rather than general answer formatting. It compares ordinary joint CE against a loss that emphasizes random digits and contrasts the anchor under its correct and twin-session memories.

No forgetting action, controller, counterfactual forgetting label, or test score is used in this stage.

## Frozen design

- Semantic seed `20261013`; train/dev/test = `128/32/32` disjoint worlds.
- One unique natural 4096-token background per world, alternating FineWeb-Edu and LongCrawl64. Each world has two twin contexts and four questions per context.
- Test is semantic text only. The packer reads and tokenizes train/dev only.
- Original audited Qwen3-4B backbone is frozen. Seven conv/proj writer pairs are initialized identically with seed 301 in both arms.
- AdamW FP32 state, BF16 forward, learning rate `1e-7`, no weight decay, clip 1, order seed 607.
- Four rounds give 128 × 2 × 4 × 4 = `4096` question exposures and `1024` optimizer updates per arm.
- Checkpoints are fixed at 0, 2048, and 4096 exposures. The final endpoint is primary.

### Arms

1. `joint_ce`: one context write; mean full answer plus EOS CE over its four questions.
2. `joint_binding`: correct-context write plus differentiable twin-context write. Its positive loss is the mean of `0.75 × digit NLL + 0.125 × prefix NLL + 0.125 × EOS NLL` over four questions. For the single changed anchor, add `0.5 × softplus(0.2 + correct-memory digit NLL − twin-memory digit NLL)`.

The arms have identical question exposure, data order, initialization, optimizer settings, and evaluation. `joint_binding` performs two writer calls per update and therefore is not compute matched.

## Evaluation and gate

- Training probe: fixed first eight train worlds, 64 questions.
- Development: all 32 dev worlds, 256 questions.
- Fresh-KV policies: Correct, Wrong, Empty, and twin for anchors. Full-KV is the readability control.
- Evaluation streams four memories per source world to avoid cumulative GPU memory growth.

An arm passes only at 4096 exposures if dev satisfies all of:

1. Full-KV code accuracy ≥ 0.80.
2. Correct-memory code accuracy ≥ 0.60.
3. Correct-memory code accuracy exceeds Wrong and Empty by at least 0.20 each.
4. Wrong−Correct and Empty−Correct source-group digit-NLL gains are each ≥ 0.10 and their fixed 5000-draw bootstrap 95% lower bounds are above zero.
5. Every generated delta obeys `||ΔW||/||W|| ≤ 1`; violation stops the arm before its optimizer update.

The stage passes if either fixed endpoint passes. Midpoint results cannot select a checkpoint. If both fail, do not train a forgetting controller. If binding passes, the next experiment must ablate digit weighting from twin contrast before the forgetting oracle.

## Commands

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_dynamic_binding_v1 \
MODEL=/home/ctj/models/Qwen3-4B \
CORPUS=/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
bash scripts/run_cbf_dynamic_binding.sh
```

