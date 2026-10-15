# Controller V1 inner-safety clarification

## Timing and scope

This clarification is registered after the first train-relation inner-fold run
failed and before any development episode is constructed. The frozen parent
specification requires checkpoint selection by joint utility accuracy subject
to zero extra collateral errors, but it did not define the temporary risk
threshold used for that inner checkpoint comparison. The first implementation
implicitly used risk probability 0.5. That unregistered default produced 3–7
unsafe deletions at epoch 25 despite 89.8%–92.8% record-action accuracy and is
discarded. No checkpoint from that run is used.

## Deterministic inner rule

For each seed, fold and registered checkpoint epoch:

1. mark a utility deletion when delete probability is at least 0.5;
2. among utility deletions whose single-delete collateral-risk target is one,
   take the minimum predicted risk probability;
3. permit deletion only when predicted risk is strictly below that minimum;
4. if no utility deletion has a positive risk target, use threshold one;
5. compute boundary action accuracy after this projection;
6. select the highest boundary accuracy among checkpoints with zero projected
   unsafe deletions, then record accuracy, risk recall and smaller epoch.

This threshold is used only to make the already registered zero-error inner
checkpoint comparison well defined. It is not saved as a deployment threshold.
The deployment utility thresholds, risk penalties and conformal risk levels
remain the finite grid in the frozen parent specification and can be selected
only on the relation-disjoint development partition. Confirmation and external
benchmarks remain sealed.
