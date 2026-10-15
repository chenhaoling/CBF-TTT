# Qwen3-4B semantic fact-key barrier V9

## Registered question

Can subject-cross-fitted token spans plus a frozen Qwen3-4B semantic relation representation repair V8's lexical fact-key failure without loading ZsRE?

## Result

V9 trained on the unchanged 5,238-pair CounterFact corpus. Every span ensemble excluded all subjects in its evaluated fold. The frozen final 1B-token Qwen3-4B checkpoint supplied 256-dimensional full-prompt and entity-masked relation vectors; the selected pair head was the preregistered linear relation-vector arm.

| Metric | V8 lexical | V9 semantic | V9 floor |
| --- | ---: | ---: | ---: |
| Precision | 67.69% | 69.00% | 99% |
| Recall | 49.30% | 88.52% | 90% |
| Different subject specificity | 100.00% | 94.68% | 98% |
| Same subject, different relation specificity | 76.47% | 65.55% | 98% |

The semantic representation substantially improved paraphrase recall, but did not form a safe relation-equivalence boundary. The independent audit passed all ten checks with zero probability or action mismatch. Runtime was 692.78 seconds; peak allocated/reserved GPU memory was 8.04/8.54 GiB. ZsRE and external benchmarks were not loaded.

## Interpretation and stop

V8 and V9 separate two mechanisms: lexical features cannot align paraphrases, while generic frozen-Qwen sentence geometry aligns paraphrases but also collapses distinct relations about the same entity. Existing independent FewRel and DocRED experiments show the same safety limitation, so another general relation-classifier iteration is not justified.

A subsequent fixed diagnostic of V7's five lifecycle-label false positives found two state-equivalent duplicate facts: the Angora prompts share source identity and target `Ankara`, despite different chain IDs and paraphrases. Deleting the old duplicate before writing the new one preserves the memory state. The next experiment therefore corrects the physical safe-delete target to include exact fact-and-value-equivalent duplicates; it does not add another semantic architecture or tune V9.
