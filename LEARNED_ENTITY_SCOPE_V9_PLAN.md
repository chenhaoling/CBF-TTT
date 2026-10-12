# Learned entity scope V9 preregistration

## Hypothesis

V5 establishes strong relation compatibility, while V6-V8 stop below 95% when
entity identity is inferred with fixed overlap features. V9 learns binary
same-entity evidence from raw text pairs and combines it with the frozen V5
relation posterior using a product rule.

## Entity supervision

For every raw CounterFact row and both paraphrases:

- positive: paraphrase and clean prompt from the same row;
- hard negative: the same paraphrase paired with a clean prompt from a
  different subject but the same relation.

The negative is selected deterministically. For each outer fold, all examples
containing any evaluated subject on either side are excluded. A frozen
Qwen3-4B cross encoding plus six lexical features feeds a linear binary head.
Three seeds train for ten epochs; no development threshold is selected.

## Inference

For each candidate, V9 computes:

`score = log P(same entity | query,key) + log P(same relation | query,key)`.

The second probability is the frozen automatic V5/V6 posterior. Inference sees
only the two texts. Subject strings, relation IDs, chain IDs, targets, ages,
positions, and future queries are excluded.

## Gate

- candidate top-1 ≥96%;
- MRR ≥98%;
- entity-pair ROC AUC ≥97%;
- at least one point gain over V8 with positive paired-bootstrap lower bound;
- at least twelve points gain over V4 with positive lower bound.

A pass freezes scope resolution and starts causal on-policy forgetting. A
failure requires a different entity encoder. LongMemEval-S and LoCoMo remain
sealed.

