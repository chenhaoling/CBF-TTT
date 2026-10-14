# Preregistered causal-ledger development V2

## Freeze point

This protocol is fixed after the fresh scale-32 V2 content validation passed
all 12 registered checks and its independent audit passed all six integrity
checks. No development row, policy outcome or comparator score existed at this
freeze point. Confirmation and OOD remain ungenerated.

## Corpus and fixed readout

Use the already reserved development seed `202610160`, development-only
surface templates and value pool. Construct 64 groups per each of the seven
lifecycle families, two twins per group: 448 groups and 896 episodes. Lengths
8/12/16 remain balanced by deterministic cycling. Use scale 32, exact five-step
generation, fresh KV, the unchanged final Qwen3-4B LM head and the frozen
addressable top-1 key readout. There is no learned controller or threshold.

## Policies

All policies see the same chronological event stream and queries:

1. `no_memory`: always route to the null code;
2. `retain_latest`: ignore reset/revoke and return the newest assignment for a
   key;
3. `global_boundary_clear`: clear all old records at every sparse boundary;
4. `last1` and `last2`: retain only the newest one or two records;
5. `dense_full`: use the correct causal state but sum cosine-weighted values
   over every visible key;
6. `full_ledger`: apply reset, explicit revoke, shadowing/restoration and no-op
   semantics, then perform addressable top-1 readout;
7. `semantic_oracle`: independently supplied causal answer through the same
   Qwen value decoder.

The primary method is `full_ledger`. `semantic_oracle` is excluded from
comparator selection. Exact per-query agreement between them is mandatory.

## Registered development gate

- primary candidate accuracy at least 95% and five-token exact at least 85%;
- current and anchor exact at least 95%, historical at least 85%, absent at
  least 90%, and every family at least 85%;
- at least five-point source-group macro exact gain over the strongest
  non-oracle comparator, with a 20,000-resample group-bootstrap 95% lower bound
  above two points;
- answer-NLL gain bootstrap lower bound above zero;
- no negative exact gain in any family or length bucket;
- exact primary/oracle candidate and generated-ID agreement;
- the V2 content result and independent audit hashes must match the registered
  passing artifacts;
- corpus, source, row, prompt, policy-state, result, resource, terminal and
  sealed-split audits must pass.

A passing independently audited development result releases one matched
confirmation at frozen scale/policies. Failure stops before confirmation and
OOD. This controlled stage does not establish natural-language key discovery
or native `ttt_proj`/`ttt_conv` content binding.

