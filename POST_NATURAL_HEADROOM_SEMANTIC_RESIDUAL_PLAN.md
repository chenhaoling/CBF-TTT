# Learned semantic residual after natural headroom

## Release and claim

This protocol is frozen after the independently audited 320-row natural
semantic headroom confirmation passed and before extracting a learned-stage
embedding or training a controller. The release summary/audit SHA256 values
are `5301773ddf293959ac7c9ac0fd015cc3dfc504f1d44ac0aef821c1db5c13a830`
and `00e8ca36d7275014fb939fc71a7e5bffeff010dc3d446a99ee0410a89f31549d`.

The semantic oracle improved joint success by 55.63 points over exact surface
matching. This stage asks whether an inference-time controller can recover a
substantial part of that gap without relation IDs. The current update remains
mandatory; only historical session components may be forgotten.

## Data isolation

Use the existing 22,496-row audited CounterFact/MQuAKE inventory and its five
relation- and subject-disjoint folds. Source relation IDs supervise training
only. For outer fold `k`, all surfaces and pair labels from relations in fold
`k` are test-only, fold `(k+1) mod 5` is inner validation, and the other three
folds train the controller. No relation, normalized subject, source row or
paraphrase pair crosses roles.

The 320 natural confirmation episodes are evaluated by their registered outer
fold. Their model outcomes, oracle actions and Qwen scores cannot enter
embedding extraction, training, threshold selection or checkpoint selection.
ZsRE dev/test, MQuAKE dev, LoCoMo, RULER, LongMemEval-S and MMLU remain sealed.

## Pair corpus

Construct relation pairs independently inside each source fold:

- within-row positive: two different masked surfaces from one row;
- cross-row positive: same relation, different normalized subject;
- lexical hard negative: different relation with the largest character
  trigram similarity among 64 deterministically hashed candidates;
- random negative: different relation selected by a separate fixed hash.

Rows with fewer than two masked surfaces can provide cross-row examples but
not a within-row pair. Exact duplicate surfaces, identical subjects across a
pair and empty values are rejected. Cap each relation at 512 anchor rows and
publish every deterministic exclusion. Each fold must contain at least 500
positive and 500 negative pairs, both sources when available, and positive
locality-risk labels. Pair construction reads no model outcome.

## Frozen representation

Replace exact subject and answer spans with the inventory's `[SUBJECT]` and
`[VALUE]` masks. Encode each unique masked surface with the final 1B-token
Qwen3-4B slow weights, native TTT disabled, BF16, fresh KV and no gradient.
Mean-pool the final hidden states over non-padding surface tokens and L2
normalize the FP32 result. Inputs longer than 256 tokens are excluded and
listed before training; they are not truncated. Cache surface text, token IDs,
length, vector hash, model/tokenizer provenance, latency and peak memory.

## Projection and controller

For each outer fold and three seeds `22031/22037/22043`:

1. train a bias-free rank-16 projection with symmetric InfoNCE at temperature
   0.07;
2. add a 0.2-margin hard-negative loss with weight 0.5;
3. form inference features from raw/projected cosine, projected positive-to-
   runner-up margin, character-trigram similarity, exact-surface identity,
   value conflict, relative age, later repetition and normalized memory
   length;
4. train a shared 64-unit MLP with utility and deletion-risk sigmoid heads;
5. score a historical component as `utility - lambda * risk`.

The Qwen backbone remains frozen. AdamW learning rate is `1e-3`, weight decay
`1e-4`, clip norm 1.0. Projection/controller epochs are 25/50/100. Nested
selection searches utility thresholds 0.4/0.5/0.6 and risk penalties
0/0.5/1/2. The fixed constraint is at least 80% positive recall, 95%
conservation specificity and at most 3% risk violation. If no setting satisfies
all constraints, select by balanced utility accuracy for diagnosis and record
the constraint failure; every outer fold must satisfy constraints for release.

Exact normalized surface matches are handled deterministically. The learned
residual only considers unresolved history with exact normalized subject match
and a conflicting value. It never receives relation ID, fold, source row,
future query, answer score, oracle action or held-out outcome. The current
update is never a deletion candidate.

## Model-level comparisons

Apply each nested OOF checkpoint to its outer-fold natural confirmation rows
and rescore every resulting retained state with frozen Qwen. Compare:

- retain all / exact surface deletion;
- global reset, last-1 and last-2;
- raw-Qwen cosine deletion;
- projected-cosine deletion;
- learned utility-risk residual;
- relation-ID semantic oracle.

Report joint update-and-retention success, update, retention, locality,
candidate NLL, free generation, deletion precision/recall, relation/source/
fold/seed strata, latency and peak memory. Use 20,000-draw subject- and
relation-cluster bootstrap intervals.

## Registered release gate

Natural benchmark confirmation is released only if the learned residual:

1. improves subject-macro joint success over the strongest non-oracle baseline
   by at least five points;
2. has subject- and relation-cluster 95% lower bounds above zero;
3. has no negative outer fold, controller seed or source stratum;
4. reaches at least 90% update and retained-fact accuracy;
5. reaches at least 97% locality and unrelated-session retention;
6. closes at least 40% of the positive strongest-baseline-to-oracle gap;
7. exceeds projected-cosine deletion by at least two points with a positive
   relation-cluster lower bound;
8. satisfies every nested inner constraint and every construction, feature,
   checkpoint, prompt, provenance and outcome audit.

Failure stops learned semantic-controller iteration on these observed sources.
It cannot be rescued with test benchmarks or by relaxing thresholds.

## Conditional external confirmation

Only a passing learned-stage audit may evaluate the frozen controller once on
LoCoMo and RULER/RULER-QA, followed by MMLU non-regression. LongMemEval-S is
already observed and diagnostic only. The native dense-writer negative result
remains unchanged; a passing controller supports semantic selective forgetting
on a readable explicit/contextual memory substrate.

