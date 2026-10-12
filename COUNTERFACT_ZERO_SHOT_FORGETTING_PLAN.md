# CounterFact zero-shot forgetting transfer

## Question

Does the ZsRE-trained state-dependent forgetting controller transfer without retraining to CounterFact completion-style prompts, different relations, and a new source distribution?

This experiment is frozen before any CounterFact model score is produced. The official ROME CounterFact file has SHA256 `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`. Schema inspection found 21,919 records and 785 subjects with at least two distinct relations. Schema counts are construction evidence, not model scores.

## Construction

For every eligible subject, deterministically select two records with different relation IDs. The first becomes a primary counterfactual chain and the second a same-subject auxiliary chain:

1. write the primary `target_true` under the requested rewrite prompt;
2. write the auxiliary `target_true` and retain the primary memory;
3. present the primary CounterFact paraphrase with `target_new` and locally forget only the primary old value.

The requested prompt and first paraphrase are reciprocal exact/rephrase queries. Subjects are packed four per session, matching the ZsRE sequence topology. No CounterFact row trains the controller, tunes the threshold, or selects a feature. The manifest records valid-primary and valid-auxiliary row counts, the number of subjects with a valid distinct-relation primary/auxiliary pair, selected-subject and selected-case-pair hashes, and the source-file hash. Invalid and no-op rows therefore cannot silently change the evaluated population.

## Frozen method and controls

- Qwen3-4B checkpoint after the 1B-token run.
- ZsRE contextual transforms, decomposed scope scorer, 13-feature action head, and development-selected threshold 0.675.
- Primary comparison: learned forgetting versus cosine-local.
- Descriptive controls: fixed retain, fixed global, fixed top deletion, and hidden-chain oracle local deletion.
- Three frozen readout seeds; subject is the paired bootstrap unit.

The current fact is always written. The experiment controls only whether an old session-memory item is removed.

## Registered transfer gate

The zero-shot transfer gate requires all of:

1. query gain over cosine-local has subject-bootstrap 95% lower bound above zero;
2. action gain has 95% lower bound above zero;
3. overall action accuracy is at least 90%;
4. same-subject hard-retain accuracy is at least 90%;
5. primary-update accuracy is at least 85%.

The complete eligible set is scored once. There is no CounterFact development tuning. Oracle is an upper bound and cannot be used as the primary baseline. Query accuracy is candidate-memory correctness, not official CounterFact efficacy or open-vocabulary generation.

## Next decision

If the transfer gate passes, use the frozen controller in a Qwen low-rank residual-memory readout with natural multi-token values. If it fails, diagnose prompt-style transfer using CounterFact only as observed data and reserve another untouched source for the next formal claim.
