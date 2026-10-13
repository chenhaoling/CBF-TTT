# Factorized scope-key candidate V3 preregistration

## Evidence and hypothesis

V2 improved stale-session candidate top-1 from 33.99% to 67.38%, but failed the
95% release floor. Its anchor-only diagnostic reached 79.21%, showing that unit
routing alone is insufficient. Every corpus unit is serialized as a memory
prompt plus a value; the value changes by design between an old fact and its
update. V3 tests the memory-system hypothesis that scope keys and value payloads
must be represented separately for deletion lookup.

V1/V2 development outcomes are open development evidence. The V3 protocol is
frozen here before execution. LongMemEval-S remains locked and LoCoMo remains
unloaded.

## Representation

Parse every audited unit mechanically at the existing `Value:` delimiter:

- scope key: text after `Memory statement:` and before `Value:`;
- payload: the value suffix, excluded from candidate retrieval.

The frozen final Qwen3-4B checkpoint and the same three contextual transforms
embed only unique scope keys. No data or model parameter is added. V3 then
reuses the V2 order-neutral session features and rank-16 listwise resolver. The
primary method still receives no unit position, age, subject, chain, target,
event kind, or split feature.

## Training and controls

Calibration-fit trains; calibration-tune selects checkpoints 10/25/50/100/200;
fit+tune refits; subject-disjoint development is scored after all selections
freeze. Seeds, optimizer, rank, listwise loss, and selection ordering are
unchanged from V2.

Controls are scope-key maximum cosine, top-4 mean, full mean,
projection-consensus maximum, and an ineligible scope anchor-only diagnostic.
The archived primary V2 records form the paired full-text baseline and are
hash-checked before comparison.

## Release gate

V3 advances to causal deletion only if all checks pass:

- three-seed mean top-1 at least 95%;
- minimum seed top-1 at least 93%;
- every length mean top-1 at least 93%;
- mean MRR at least 97%;
- primary-seed gain over V2 has subject-cluster bootstrap lower bound above 15
  points;
- scope-key maximum cosine itself beats full-text V1 maximum cosine by at least
  30 points, confirming the factorization mechanism independently of the MLP.

If the gate fails, feature-level candidate ranking stops. The next admissible
method is a query-conditioned cross encoder or token-level relation router. If
it passes, freeze the scope representation and ranker, then preregister a new
causal high-specificity deletion gate. Passing V3 alone is not evidence of
end-to-end forgetting efficacy.
