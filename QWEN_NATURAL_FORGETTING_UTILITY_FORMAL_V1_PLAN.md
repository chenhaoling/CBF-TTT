# Formal natural counterfactual forgetting-utility labels V1

## Release decision

The 16-session pilot produced 94 audited labels and passed all eight scaling
checks.  Safe counterfactual selection improved future candidate accuracy from
78.3251% to 82.4302% (gain 4.1051 points; session-bootstrap 95% CI
[3.5417, 4.7461]) while adding zero unrelated errors.  Eight canonical actions
were selected, so the formal construction retains the registered action grid
rather than reducing the task to a binary controller before observing the full
calibration distribution.

## Scope and seals

Use all 65 sessions and all 774 events from the sealed calibration split of
the natural causal-memory corpus.  The model, tokenizer, contextual
transforms, decomposed scope checkpoint, V5 checkpoint, scale, reader
threshold, intervention semantics, future replay, control-barrier constraint,
lexicographic label objective and 20,000-draw session bootstrap remain exactly
as defined in `QWEN_NATURAL_FORGETTING_UTILITY_PILOT_V1_PLAN.md`.

Input SHA256 seals remain:

- data manifest:
  `ceb2c85ae2a01fd6de7ef08d68004e58e43b8ae424708bea4fa313344d1a7ba7`;
- V5 checkpoint:
  `3a0ba5a83ece4850dd3824fc2277dc2e0dfdb04902e361dbff764e1b630abadc`;
- contextual transforms:
  `6f6a696005185d80d6b8f468fa6442146dc8d9683e232e7a591676ba9515c378`;
- decomposed scope checkpoint:
  `0ed4dab8b77fd0fb91fdd49720c783261ab2fd5c64cd705ebe36346d99615619`.

No development, test or external benchmark may be loaded.  Construction is
sharded by calibration-session index across the two RTX 5090 devices.  Every
event is labeled, including empty-memory and clear retain anchors, because
these negatives are needed to estimate false-forgetting behavior.

## Action and outcome records

Execute the same 11 unique states representing the complete five-scope by
three-alpha grid.  Keep per-query outcomes, per-label time, cache hits/misses,
peak memory, selected safe and unconstrained oracle actions, and all hashes.
The alpha-one aliases remain explicit in the summary.

The formal corpus is a supervision resource.  Its oracle gain measures
available headroom and does not count as learned-policy evidence.

## Formal corpus gate

Controller training is authorized only if every check passes:

- exactly 65 sessions, 774 unique event labels and the complete action grid;
- safe-oracle gain over retain-all has a session-bootstrap 95% lower bound
  above zero;
- at least 10% of labels choose a non-retain action;
- both `retain_all` and `scope_top1_a0` have at least 50 labels, so the dominant
  safety and local-delete decisions are trainable;
- at least five canonical actions occur and no action exceeds 90% of labels;
- the safe oracle adds zero unrelated candidate errors;
- median label time is at most 120 seconds and peak allocated memory is at most
  28 GiB per GPU;
- every shard and input seal passes an independent audit.

If the gate passes, train a hierarchical controller using grouped
session-level cross validation: first retain versus intervene, then action
scope/strength conditional on intervention.  Compare it with a flat
multiclass model, the source action labels, cosine threshold, FIFO/global
rules, random rate-matched forgetting, retain-all and the counterfactual
oracle.  Hyperparameters and the one-time development release must be
preregistered after inspecting only grouped calibration OOF results.
