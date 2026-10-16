# V16 contextual-to-native fast-weight writer distillation pilot

## Question

Can the original seven-layer `ttt_proj/ttt_conv` writer be taught to produce a
content-specific dense fast weight when ordinary answer likelihood and
Correct-versus-Wrong losses have already failed?

This is a development pilot. It reuses the audited color-binding source groups
0--15, all of which have been observed in earlier diagnostics. It cannot be a
new held-out claim. Groups 0--7 train the writer and groups 8--15 measure
source-disjoint diagnostic transfer. The frozen Qwen3-4B backbone, original
down-projection fast-memory location, 4,096-token native update, and fresh-KV
readout remain unchanged.

## New supervision

The previously validated Qwen rank-one bridge supplies a read-only teacher.
For answer token `y`, the teacher query state is

```text
h_teacher = h_zero + s * normalize(lm_head[y]).
```

The student state is produced only by the native dense fast weight written from
the current chunk. Training minimizes the existing candidate/full-vocabulary
likelihood and Correct-vs-Wrong/Zero terms plus:

1. relative hidden-state MSE from the Correct student to `h_teacher`;
2. a paired margin requiring the Correct memory to be closer to its teacher
   than the counterfactual Wrong memory.

The teacher is never available at inference. It provides training supervision
only. Two fixed arms use teacher scales 4 and 8, the first two nonsaturated
scales that previously gave strong readable residuals. Both use learning rate
`3e-7`, eight rounds, a relative fast-weight norm cap of 0.3, identical data
order and initialization, and 256 optimizer updates.

## Diagnostic gate

On groups 8--15, each arm is evaluated under Correct, Wrong, Zero, and
layerwise norm-matched Random dense fast weights. A promising arm requires:

- Correct candidate accuracy at least 50%;
- Correct exceeds Wrong and Zero accuracy by at least 25 points;
- Wrong-minus-Correct, Zero-minus-Correct and Random-minus-Correct NLL gains
  are all positive with source-group bootstrap 95% lower bounds above zero;
- EOS first-token rate at most 5%;
- the frozen backbone hash and all data/checkpoint hashes reconstruct.

Failure of both arms stops further tuning of this dense writer architecture.
Passing authorizes a fresh-source, multi-seed confirmation before any
forgetting-controller labels are constructed.
