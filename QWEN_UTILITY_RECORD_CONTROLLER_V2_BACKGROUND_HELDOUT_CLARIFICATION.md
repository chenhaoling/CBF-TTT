# V2 TempLAMA background-held-out construction clarification

## Freeze boundary

This clarification is frozen after the first model-free validation construction
and its exact reconstruction audit, but before loading Qwen or producing any
validation label, feature or controller outcome.  The controller checkpoints,
features, thresholds, relation quotas, target subjects, mechanisms, queries,
release gates and all sealed partitions remain unchanged.

## Model-free finding

The registered reusable validation background bank was necessary because 256
target groups cannot each receive six globally unique background subjects from
the decontaminated validation partition.  The first deterministic construction
used all 115 remaining eligible background subjects in more than one group.
Consequently, the preregistered sensitivity subset that excludes every group
sharing a background subject was empty.  The artifact is retained at
`cbf_ttt_utility_record_controller_v2_templama_development_data_20261016` and
cannot release model evaluation even though its original reconstruction checks
passed.

## Deterministic repair

The retry reserves exactly one target group from each of the nine TempLAMA
relations.  The six rendered background subjects in each reserved group are:

1. distinct inside the group;
2. distinct across the nine reserved groups; and
3. forbidden from all remaining 247 groups.

This creates a fixed 9-group background-held-out subset with 54 globally
unique rendered background subjects.  Remaining background subjects may still
be reused under the original capacity clarification.  Target sampling and
ordering do not change.  The anchor alternate is a value donor whose source
subject is never rendered; it is excluded from the rendered-background
identity analysis in both constructions.

The independent data audit must verify that the reserved subset contains one
group per relation and that every one of its rendered background identities
occurs exactly once in the entire development corpus.  A retry that cannot
satisfy these invariants stops before any model execution.

## Statistical use

Primary release authority remains the 256 target-group clustered interval.
The 9-group background-held-out result and joint target/background sensitivity
are reported diagnostics only.  They cannot select a checkpoint, threshold,
baseline or deletion rule and cannot override a failed primary gate.
