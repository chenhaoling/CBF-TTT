# Frozen contextual residual-scale diagnostic

## Freeze point and scope

This diagnostic is fixed after the audited causal-ledger resource/content
pilot and its codebook-geometry diagnostic. The pilot gate remains failed and
cannot be changed. All 17 value-token directions rank their own token first
under the unchanged LM head, while 735 of 1,008 scale-16 rollouts first fail at
the second token because contextual logits prefer the corresponding no-space
token. Development, confirmation and OOD are still absent.

The diagnostic uses only the failed pilot and tests the fixed scale set
`{16, 20, 24, 32, 48, 64}`. It does not train a direction, change prompts,
change values, inspect any downstream split or alter a lifecycle action.

## Measurements

For all 1,008 pilot queries on two RTX 5090 cards, use the correct single-value
residual and generate exactly the registered five answer tokens. At every
scale record:

- free-rollout token and exact accuracy by position;
- teacher-forced full-vocabulary token and sequence accuracy;
- full-vocabulary answer NLL;
- first-error position and common target/generated confusions;
- wall time and CUDA allocated/reserved peak.

The smallest scale reaching at least 85% free exact match and 95%
teacher-forced sequence accuracy is the only scale eligible for a successor.
If none qualifies, scalar-strength repair stops and the successor must use a
contrastive/contextual decoder. The selected scale is diagnostic evidence only;
a new, versioned resource/content protocol must still re-establish all Correct,
Wrong, Zero and Random controls before development can be built.

