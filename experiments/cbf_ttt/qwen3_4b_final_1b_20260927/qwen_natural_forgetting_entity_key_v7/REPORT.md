# Qwen3-4B natural forgetting entity-key V7

## Registered question

Can an inference-time entity anchor prevent unsafe deletion when the memory-top fact and the incoming fact mention different entities, without using gold subject identifiers?

## Result

The independent audit passed all 9 integrity checks. Across five outer folds the entity-key controller produced TP=609, FP=5, FN=69 and TN=1549: 99.19% delete precision, 89.82% safe-delete recall and a 27.51% delete rate. Relative to DAgger V6, true deletes increased from 551 to 609 and false negatives fell from 127 to 69; false deletes fell from six to five.

The registered operational gate did not pass because it allowed at most four false deletes. All other registered metrics passed. The source test split remained sealed.

## Error mechanism

The remaining five false-positive rows correspond to three event types:

1. the same relation expressed for different entities;
2. the same entity queried through different relations; and
3. a local update whose current memory top is another relation for that entity.

These errors show that entity identity alone is insufficient. The next barrier must test the full fact key: entity compatibility and relation compatibility.

## Decision

Stop V7 without opening test. Train the fact-key matcher on an independent CounterFact corpus, freeze it, and transfer it to the V7 out-of-fold decisions with zero ZsRE parameter selection.
