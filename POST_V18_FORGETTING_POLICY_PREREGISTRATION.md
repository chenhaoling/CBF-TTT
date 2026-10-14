# Post-V18 selective-forgetting policy preregistration

## Registration boundary

This protocol is frozen while the 320-row V18 formal causal-headroom run is in
progress and before reading any formal outcome.  It uses only the admitted
40-row V18 pilot, the already archived negative native dense-writer studies,
and the already archived positive Qwen low-rank key-value bridge.  A failed V18
formal headroom gate stops this protocol before any controller is trained.

## Claim boundary

V18 textual-memory scoring establishes whether selective deletion has causal
value under a joint update-and-retention endpoint.  It does not by itself show
that the checkpoint's native `ttt_proj`/`ttt_conv` fast weights store readable
facts.  Previous audited experiments found no stable content binding for that
dense writer, while the Qwen low-rank key-value fast memory passed held-out Q
and V content-binding gates.  The paper may claim an effective fast-memory
forgetting method only after the V18 effect is reproduced with additive
low-rank fast parameters and an inference-time policy.

## Stage A: formal causal-headroom release

Release later stages only if the frozen V18 formal run passes every registered
subject/template cluster, update, retention, locality and fold gate.  Candidate
joint success remains primary; free generation is secondary.  No controller
architecture, feature, threshold or natural benchmark may rescue a failed
formal gate.

## Stage B: decomposable Qwen fast-memory bridge

Represent each session fact as additive low-rank Qwen-space components.  A key
is derived from the observed entity-slot text using the frozen Qwen input
geometry.  Each of the five opaque answer positions has a value direction in
the frozen LM-head geometry, so multi-token answers remain autoregressive.  A
memory state is the sum of retained components; deleting a session removes
only its stored factors and never changes the frozen backbone.

Use the same frozen V18 episodes, outer folds, fixed arms and exact deletion
matrix.  Report residual scales 4, 8 and 16 as required by the earlier bridge
study; scale 8 is the primary setting fixed before this experiment.  A scale
is not selected using formal labels.  Fresh KV is used for every query.

The bridge gate requires, at primary scale 8:

- correct-component content accuracy at least 90%;
- wrong-component accuracy at most 20%;
- exact-oracle update and retained-key accuracy at least 90%;
- exact-oracle locality at least 97%;
- oracle joint gain over the best outer-fold fixed policy at least 10 points;
- subject- and template-cluster 95% lower bounds above zero;
- the direction of the joint gain to agree at scales 4, 8 and 16.

Failure limits the V18 result to an explicit-memory diagnosis and blocks a
fast-memory controller claim.

## Stage C: mandatory physical-rule baselines

Construct a conflict graph over observed memory components.  Two components
are candidates for a conflict edge when their key similarity is high and
their value representations disagree.  A newer conflicting component can
suppress an older component; agreement is evidence for useful repetition and
does not trigger deletion.  Unrelated components are conserved.

Evaluate these frozen rules before fitting a learned policy:

1. exact entity-slot match, delete older conflicting values;
2. Qwen key-cosine conflict with a threshold selected inside the training
   portion of each outer fold;
3. most-recent same-key component only;
4. random deletion matched to each rule's deletion count.

The exact-match rule is a required strong control because the controlled V18
keys are explicit.  If it reaches the oracle, the controlled set validates the
physical mechanism but cannot establish a learning advantage.  Learned-policy
claims must then come from unseen semantic aliases and natural data.

## Stage D: causal deletion-utility policy

Fit one score per historical component.  The target is the downstream joint
loss difference between retaining and deleting that component, using only
counterfactual arms available in the frozen training folds.  Query answers,
future losses, fold identity, episode length and absolute position are never
inputs at inference.

Allowed features are available from past state only:

- current-query/key and component-key cosine;
- newest-event key and component-key cosine;
- value agreement or conflict with newer components;
- component readout magnitude and query-conditioned contribution;
- relative age within the same inferred key, repetition count and past access;
- fast-memory norm and component interaction summaries.

Use a two-layer 64-unit MLP with GELU, layer normalization and one scalar
utility output.  Seeds are `18031`, `18037` and `18043`; AdamW learning rate is
`1e-3`, weight decay `1e-4`, gradient norm cap `1`.  Candidate epochs are
25/50/100 and decision thresholds are 0.4/0.5/0.6, selected only inside the
training portion of each outer fold.  The loss combines Huber regression on
causal utility, pairwise ranking against retain-all, conservation on unrelated
components and consistency for useful repetitions.  The four weights are
fixed at 1.0/0.5/0.25/0.25.

Produce out-of-fold predictions for every formal subject and template.  The
primary comparison is the three-seed mean against the strongest outer-fold
fixed arm and the strongest physical rule.  Also report retain-all, global
reset, last-1, last-2, matched random, cosine conflict, the earlier V15 scalar
controller and exact oracle.

The policy gate requires:

- at least five points of subject- and template-macro joint gain over the best
  fixed arm, with both clustered 95% lower bounds above zero;
- non-negative joint gain in every outer fold;
- update and retained-key accuracy at least 90%, locality at least 97%;
- at least 40% of the fixed-to-oracle joint gap closed;
- no seed below the best fixed arm;
- improvement over the exact-match physical rule on a separate semantic-alias
  set.  Matching that rule on exact-key V18 is reported but is not sufficient.

## Stage E: semantic-alias and natural confirmation

Freeze the controller and all thresholds before constructing the confirmation
sets.  The semantic-alias controlled set replaces literal entity/slot matches
with unseen paraphrases and source-disjoint descriptions while preserving the
same causal graph.  The frozen policy must beat exact string matching and
retain at least 90% update/retention accuracy.

Only then evaluate once on MQuAKE-CF, ZsRE, LongMemEval-S and LoCoMo.  MMLU,
RULER and RULER-QA measure slow-knowledge and long-context preservation.  The
natural-data primary comparison is against the strongest fixed and physical
rule, with subject/session clustered intervals and three controller seeds.
External data cannot select scales, thresholds, epochs, features or loss
weights.

## Stop rules and paper interpretation

- Formal V18 failure: no controller training on this corpus.
- Low-rank bridge failure: report explicit-memory headroom only; no fast-weight
  efficacy claim.
- Exact-key rule equals oracle and semantic policy fails: report a deterministic
  conflict-resolution mechanism, not learned forgetting.
- Controlled policy passes but natural confirmation fails: report synthetic
  mechanism evidence and the failed transfer explicitly.
- A publishable main claim requires formal headroom, readable fast parameters,
  a causal inference-time policy or principled physical rule, held-out semantic
  generalization and no material safety regression.

All stages save raw outcomes, split and source hashes, model/tokenizer hashes,
three-seed checkpoints, selection grids, per-label timing and peak GPU memory,
summary, independent audit and an artifact SHA-256 list.
