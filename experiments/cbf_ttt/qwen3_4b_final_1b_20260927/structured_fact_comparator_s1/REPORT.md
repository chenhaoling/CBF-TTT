# Structured fact comparator S1 report

## Decision

S1 is rejected by its preregistered one-shot development gate. The SemEval
test split remains sealed. The experiment demonstrates reliable constrained
action scoring and valid JSON generation, but it does not demonstrate learned
semantic forgetting: the constructed update and duplicate examples preserve a
relation template that a lexical rule identifies perfectly.

## Protocol

- Backbone: final 1B-token Qwen3-4B checkpoint.
- Weight training: rank-16 LoRA on 48 FewRel training relations only.
- Calibration: 26,805 constructed pairs from SemEval train.
- One-shot development: 6,140 constructed pairs from SemEval validation.
- Physical action: `duplicate` for equal entity/relation/value, `update` for
  equal entity/relation and unequal value, and `retain` otherwise.
- Candidate scoring: all eight physically consistent JSON states with a shared
  prompt KV cache; confidence is the best-versus-second mean token-NLL gap.
- Training time: 2,434.62 seconds for 1,500 two-GPU steps.

## Calibration

| Arm | Delete precision | Delete recall | Four-field accuracy | Relation specificity |
|---|---:|---:|---:|---:|
| zero-shot | 99.00% | 97.26% | 60.53% | 87.09% |
| LoRA 0.25 epoch | 99.63% | 100.00% | 82.07% | 59.07% |
| LoRA 0.50 epoch | 99.99% | 99.99% | 82.08% | 57.12% |
| LoRA 1.00 epoch | 99.98% | 100.00% | 81.66% | 55.56% |
| LoRA 2.00 epoch | 99.99% | 100.00% | 80.37% | 51.25% |
| exact lexical | 100.00% | 100.00% | 80.00% | 100.00% |
| trigram | 100.00% | 100.00% | 80.00% | 100.00% |

The frozen selector chose the 0.25-epoch adapter and threshold
`0.20833873748779297`. The independent calibration audit reconstructed all
26,805 rows and passed all six integrity checks.

## One-shot development

The selected adapter obtained:

- delete precision 99.8373% (2,455 TP, 4 FP);
- delete recall 99.9593% (1 FN);
- four-field accuracy 82.4430%;
- same-relation recall 98.0456%;
- same-relation specificity 59.0391%;
- relation-macro balanced accuracy 79.2734%;
- same-entity/different-relation deletion specificity 99.6743%;
- unconstrained JSON validity 100%;
- unconstrained exact four-field accuracy 82.2150%.

Four of the nine preregistered checks failed: four-field accuracy, relation
specificity, relation-macro balanced accuracy, and a ten-point safe recall gain
over the best non-adapted baseline. Exact lexical and trigram rules both
obtained 100% delete precision and recall, so the gain was mathematically
impossible. The independent development auditor reconstructed all 6,140 rows,
the metrics and the failed gate; all seven audit checks passed.

## Failure mechanism

S1 constructs an update by replacing the value inside the anchor sentence and
a duplicate by reusing the anchor sentence. After masking subject and value,
both therefore retain exactly the same relation surface. Different-relation
retains use a different donor sentence. A lexical rule consequently separates
all delete-compatible rows without understanding the relation. The LoRA also
learns this shortcut: its action score is nearly perfect while raw relation
specificity is only 59.04%.

This is an evaluation-construction failure rather than evidence that semantic
relation comparison is solved. S1 must not be used for a positive paper claim
or integrated into the TTT writer.

## Release state

- Calibration selection SHA256:
  `50b590a6eeb89f02157210e82ccafbcb8a3b50c00fa044750d191191e783198d`.
- Development result SHA256:
  `ec41cb5f7864a077126932594fe23c2e6fa84f0579938fcd10179ba07a549d07`.
- SemEval test loaded: false.
- Terminal status: `stopped_by_development`.

The next registered experiment is S2: a natural-pair relation retriever and
reranker whose model input masks entity and value strings and whose pair labels
cannot be recovered from template identity.
