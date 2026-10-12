# ZsRE pairwise scope resolver plan

## Trigger and diagnosis

The global contrastive key transform raised development locality from 88.26% to 99.09%, but reduced unseen-subject rephrase accuracy from about 98.7% to about 69.5%. All representation seeds reached 100% calibration prompt-to-rephrase retrieval, so the failure is generalization rather than incomplete optimization. ZsRE locality prompts can also be close paraphrases of the edited relation while carrying a ground truth that should remain unchanged. In this protocol, `defer` means use the frozen language model rather than read the session update.

This plan is frozen before fitting a scope resolver or loading the existing sealed test split.

## Question

Can frozen Qwen text evidence decide whether a query is inside the scope of one stored update, without symbolic relation labels, subject IDs, answers, or a global key deformation?

## Data separation

- Keep the existing subject-disjoint 267/267/534 calibration/development/test split unchanged.
- Deterministically order the 34 calibration sessions by `sha256("scope-resolver-v1|" + session_id)`.
- Use the first 27 sessions to fit resolver parameters and the remaining seven only to select a checkpoint.
- Never use development to select an epoch, feature, template, or hyperparameter.
- Load test once only if the primary resolver passes every development gate.

## Frozen pair representation

For every query and every memory prompt in its session (up to eight), encode:

```text
Memory update scope: {memory prompt}
Candidate query: {query}
Same factual scope:
```

Use the final frozen Qwen3-4B checkpoint's L2-normalized last-layer hidden state at the final token. The backbone, tokenizer, input template, session composition, values, and labels are frozen. No subject string, relation annotation, answer, split name, or row identifier is supplied separately.

## Resolver and controls

### Primary: `cross_linear`

- One affine score over the frozen pair hidden state plus one learned scalar `defer` logit.
- For each query, score up to eight session keys and append `defer` as the ninth action. A final short session is zero-padded to eight keys and invalid logits are masked to negative infinity.
- Exact and rephrase queries target their own memory key.
- Both locality queries target `defer`.
- Optimize nine-way cross-entropy for 200 full-batch epochs with AdamW, learning rate 0.01 and weight decay 1e-4.
- Train seeds 7101/7102/7103 from independent Gaussian weight initializations with standard deviation 0.001, zero bias and zero `defer` logit; example order remains deterministic.
- Evaluate epochs 10/25/50/100/200 on the seven calibration-tune sessions and select maximum accuracy, breaking ties by lower NLL and then earlier epoch.

### Control: `bi_cosine`

Use the frozen v1 contextual embeddings and select among the eight keys by cosine similarity or `defer`. For each representation seed, choose the defer logit from `{0.1,...,0.9}` on the same calibration-tune sessions. It receives the same structured eight-key memory but no learned pair representation.

## Development gate

The primary mean across three seeds must satisfy all of:

- exact scope accuracy at least 90%;
- rephrase scope accuracy at least 70%;
- locality/defer accuracy at least 90%;
- minimum seed overall accuracy at least 80%;
- overall improvement over `bi_cosine` at least 5 percentage points;
- source-session paired bootstrap 95% confidence lower bound for that improvement above zero.

Only a full pass releases the 534-subject test once. The test reports both arms, per-type metrics, per-seed stability, and a source-session bootstrap comparison.

## Stop rule

If the linear frozen cross representation fails development, keep test sealed. The next admissible iteration is a calibration-trained low-rank cross-encoder adapter or an externally pretrained semantic-equivalence model. Do not change the query labels, lower the gate, add subject/relation metadata, or sweep hyperparameters on development.

## Claim boundary

A positive result establishes automatic update-scope resolution in an explicit eight-key session memory on a constructed subject-disjoint ZsRE protocol. It does not establish official ZsRE generation, factual correctness after deferral, native dense `ttt_proj`/`ttt_conv` writing, or an end-to-end forgetting gain.
