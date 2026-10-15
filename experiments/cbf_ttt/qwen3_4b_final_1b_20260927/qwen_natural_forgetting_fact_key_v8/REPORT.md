# Qwen3-4B lexical fact-key barrier V8

## Registered question

Can prompt-derived entity anchors plus whole-prompt and entity-masked lexical overlap identify a complete fact key on a subject-disjoint CounterFact validation split, before any transfer to ZsRE?

## Independent result

The reconstructed corpus contains 5,238 pairs from 1,746 eligible facts, balanced across a positive paraphrase pair and two hard-negative types. Fold 4 contains 357 examples of each type. Subjects are disjoint across all five folds.

| Metric | Result | Registered floor |
| --- | ---: | ---: |
| Precision | 67.69% | 99% |
| Recall | 49.30% | 90% |
| Same relation, different subject specificity | 100.00% | 98% |
| Same subject, different relation specificity | 76.47% | 98% |

The independent audit passed all eight checks after using a two-ULP-scale float32 reconstruction tolerance (`2e-7`); the maximum probability discrepancy was `1.19e-7` and there were zero action mismatches. ZsRE calibration, ZsRE test and external benchmarks were not loaded.

## Interpretation

The entity-anchor component completely rejects different-subject negatives, but lexical overlap cannot represent relation equivalence across paraphrases. It rejects more than half of true same-fact paraphrases while accepting almost one quarter of same-subject/different-relation pairs. This is a representation failure rather than a small threshold error.

## Decision

Stop V8 before ZsRE transfer. Do not search another lexical threshold. The next registered experiment must use a semantic relation representation, retain subject-disjoint validation, and reuse the already validated V11 token-span extractor to separate entity identity from relation meaning.
