# Low-Rank Fast-Weight Forgetting Experiment

## Motivation

The structured-memory pilot establishes that state-dependent local and global forgetting can outperform fixed policies, but exact dictionary lookup is easier than TTT fast-weight retrieval. This experiment inserts a controlled low-rank fast-weight substrate between the controller and the original dense writer.

Each visible assignment writes an outer product between a deterministic entity key and value vector. A batch delta is the sum of its assignment outer products, and the session state is their weighted sum. This preserves the additive fast-weight form while making every component auditable.

## Counterfactual actions

At each registered boundary, compare four actions from the identical pre-boundary state:

- `retain`: global alpha 0;
- `half`: global alpha 0.5;
- `local`: remove only the explicitly referenced batch;
- `global`: global alpha 1.

All actions then add the same current batch delta. Labels minimize current-query mean NLL, with ties resolved by the smallest measured memory change. Controller features include relative memory/delta norms, alignment, referenced-component state, recency, lexical structure, overlap, duplication, and capacity statistics. No answers, future blocks, family, split, entity IDs, or value IDs enter the controller.

## Robustness and controls

Use a fresh 128/32/32 train/dev/test corpus and three independent deterministic low-rank embedding seeds. Train the full controller, a reference-blind ablation, and a global-only controller restricted to alpha 0/0.5/1. Compare with fixed alpha policies, fixed local deletion, deterministic random, a visible physical rule, and a counterfactual oracle.

The development gate requires:

- oracle top-1 at least 90%;
- learned top-1 at least 88% and at least 85% for every projection seed;
- learned within three points of oracle;
- learned exceeds both the best fixed policy and global-only learned controller by at least eight points;
- the paired source-group bootstrap lower bound over the stronger of those controls exceeds five points.

Test is read exactly once only after the aggregate development gate passes. A pass supports the forgetting mechanism on additive low-rank fast weights; it still does not claim that the current Qwen TTT writer produces an equally readable delta.
