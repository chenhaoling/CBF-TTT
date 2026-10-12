# Learned entity scope V9

## Objective and protocol

Learn same-entity evidence from text after V6-V8 exhausted fixed entity
heuristics. Training used 87,676 balanced pairs: each CounterFact paraphrase was
paired with its clean prompt as a positive and with a different subject under
the same relation as a hard negative. Every outer fold removed any pair
containing an evaluated subject on either side.

A frozen Qwen3-4B cross encoding plus lexical features fed three linear entity
heads. Candidate score was the product of the OOF entity probability and the
automatic V5/V6 relation posterior. Inference received only query/key text.

## Result

| Method | Top-1 | MRR |
|---|---:|---:|
| V4 | 82.93% | 90.91% |
| V8 | **94.01%** | **96.97%** |
| V9 | 92.48% | 95.79% |

The learned entity-pair ROC AUC was 96.57%, narrowly below the 97% floor. V9
was 1.53 points below V8, CI `[-3.31,0.25]`, and 9.55 points above V4, CI
`[6.62,12.48]`. Every gate check except the positive V4 confidence interval
failed.

The entity head learned meaningful but imperfect evidence. Multiplying its
probability directly with relation compatibility over-penalized some correct
candidates and did not improve ranking. This rejects the fixed product rule,
not the entity signal itself.

## Next decision

One final nested-OOF fusion may add the OOF entity probability to V8's feature
set. If that fusion does not pass the automatic 95% scope gate, CounterFact
scope iteration stops and the next representation must be a token-level entity
span extractor rather than another score calibrator.

## Resources and claim boundary

- Runtime: 324.84 seconds on one RTX 5090.
- Peak CUDA allocated/reserved: 9.520/12.617 GiB.
- 87,676 training examples; 93,149 unique entity pairs.
- Formal code commit: `6ca909f`.
- LongMemEval-S and LoCoMo remained unscored.

This is observed CounterFact development evidence and does not establish causal
forgetting, native fast-weight benefit, or final benchmark performance.

