# Frozen causal-ledger residual codebook diagnostic

## Freeze point

This diagnostic is fixed after the independently audited causal-ledger pilot
failed its generation check and after the saved-token five-step recomputation,
but before loading the LM head for any geometry measurement. It uses only the
failed pilot and cannot change its gate. Development, confirmation and OOD
remain ungenerated.

## Question

The pilot obtained 100% six-candidate accuracy in all seven lifecycle families
at scales 4, 8 and 16, but zero registered free-generation exact match. Even
when generation is truncated to the registered five-token answer length,
scale 16 reaches only 27.08%. The diagnostic tests whether
`normalize(W[token])` fails to make `token` the highest-scoring item over the
*full* Qwen vocabulary, and whether observed rollout errors select the same
LM-head competitors.

## Fixed analysis

Load the exact final Qwen3-4B checkpoint and tokenizer used by the pilot. From
the hash-verified pilot targets, collect every distinct answer token ID. For
each ID, apply its current unit-row direction to the unchanged LM head and
record:

- full-vocabulary rank and top-5 token IDs/text;
- target and top-competitor logits, margin, row norms and cosine;
- rank under a diagnostic cosine-normalized output head (never used as a model
  score or proposed method).

From the saved scale-16 rollout IDs, report accuracy by answer position, first
error position, and target-to-generated-token confusion counts. The producer
must verify source hashes, checkpoint loading keys, five-token targets, 1,008
query rows and that no downstream split exists.

## Decision

If every current direction ranks its target first under the real LM head, the
next experiment must study contextual hidden-state interference. Otherwise,
the next method may replace the value codebook with a context-independent
full-vocabulary margin direction. No scale sweep or codebook optimization is
allowed inside this diagnostic.

