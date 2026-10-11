# Candidate-Ranking Localization Experiment

## Question

The dynamic binding writer lowered digit NLL but generated zero complete random codes. This diagnostic asks whether the correct value is nevertheless represented well enough to rank above explicit alternatives, or whether the writer lacks entity-to-value binding before decoding.

No writer training, checkpoint selection, forgetting label, controller, or test score is allowed.

## Frozen inputs

- Reuse the audited `dynamic_binding_v1` data and the six `joint_ce` / `joint_binding` writer checkpoints at 0, 2048, and 4096 question exposures.
- Score the fixed first eight train worlds and all 32 development worlds. The semantic test split remains unopened by the packer and unscored.
- Use the same frozen Qwen3-4B backbone and fresh-KV Correct, twin, Wrong, and Empty memories.
- Reuse the prior audited Full-KV dev code accuracy as the positive readability control, with its source-summary hash. Do not repeat long-context forwards.

## Candidate sets

For every scored query:

1. `pair2`: the correct and twin anchor values. This is defined only for anchor queries.
2. `world5`: the five unique values across the two twin contexts in the same world.
3. `global16`: `world5`, all values in the fixed Wrong context, then deterministic same-split negatives until exactly 16 unique values.

Candidate order is a deterministic SHA-256 ordering keyed by row id. The sets and labels are fixed before any model score is read. `pair2` is a subset of `world5`, which is a subset of `global16`.

Each candidate receives teacher-forced full-answer and digit-only NLL. Primary ranking uses digit NLL. Report top-1 accuracy, mean reciprocal rank, mean rank, best-negative margin, and true-candidate digit NLL by arm, checkpoint, split, memory policy, and candidate set.

## Bridge and integrity checks

- For a fixed real row at every loaded checkpoint, compare the new batched effective-weight scorer against the established CBF fresh-KV scorer for the true candidate under Correct and Empty memory. Maximum full-answer and digit-NLL error must be at most `2e-5`.
- Verify writer checkpoint and data hashes before and after scoring.
- Independently reconstruct all candidate sets, rankings, aggregates, and 5000-draw source-world bootstrap intervals from raw JSONL.
- Require identical checkpoint-0 writer tensors across arms and identical checkpoint-0 scores.
- Reject missing rows, non-finite scores, candidate leakage, test access, or changed source artifacts.

## Fixed interpretation

The final `joint_binding` development `global16` result is primary.

- **Strong latent binding:** Correct top-1 >= 0.50; Correct exceeds Wrong and twin by >= 0.20; and Wrong-minus-Correct true digit-NLL source-world bootstrap 95% lower bound is above zero.
- **Weak latent binding:** Correct top-1 >= 0.25 and at least one of the Correct-versus-Wrong/twin NLL gains is positive, while the strong rule fails.
- **No localized binding:** neither rule holds.

These labels localize the bottleneck. They do not pass the previous generation gate and cannot authorize forgetting-controller training.

## Command

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_candidate_ranking_v1 \
SOURCE=/home/ctj/cbf_ttt_dynamic_binding_v1/formal \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
bash scripts/run_cbf_candidate_ranking.sh
```
