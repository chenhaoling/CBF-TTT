# Cross-domain forgetting reranker V2 plan

## Evidence motivating the iteration

V1 is observed development evidence. Its five-fold CounterFact OOF candidate
top-1 accuracy reached 78.47%, passing the registered 75% ranker gate. Hard
retain reached 92.36%. The downstream gate was too conservative: update action
was 63.69%, overall action was 84.93%, and the complete gate failed. V1 ranker
curves selected epoch 10 in three of five folds and overfit rapidly, while its
safe-delete target trained the gate to suppress deletion whenever the current
ranker was wrong.

## Frozen V2 changes

V2 changes only the development training procedure:

1. fix residual-ranker training to 10 epochs, the modal V1 inner selection;
2. use all four non-held-out CounterFact folds plus all observed ZsRE when
   fitting each outer ranker;
3. train the gate for 25 epochs on the semantic conflict label (`local` versus
   `retain`) with inverse-frequency positive weighting;
4. choose the delete threshold from four inner OOF predictions, maximizing the
   smaller of hard-retain and update accuracy, then overall action accuracy;
5. refit ranker and gate on the complete outer-training partition and evaluate
   the untouched outer fold once.

The current fact remains always written. No chain identity, event label, dataset
name, relation ID, or target string enters the model features. The ranker still
starts from the frozen cosine prior and uses only frozen Qwen pair evidence.

## Evaluation and stop rule

The same deterministic five outer folds, metrics, controls, and V1 development
gate are reused. Raw oracle-state event predictions are saved so candidate and
gate metrics can be independently recomputed. V2 advances only if it satisfies
all five preregistered V1 checks:

- candidate top-1 at least 75%;
- overall action at least 85%;
- hard retain at least 85%;
- update at least 75%;
- causal query accuracy above fixed retain, 49.44%.

If V2 fails update while candidate top-1 remains near 80%, the next change must
improve pairwise relation matching rather than further tune a scalar threshold.
LongMemEval-S and LoCoMo remain sealed at their previously recorded hashes.

