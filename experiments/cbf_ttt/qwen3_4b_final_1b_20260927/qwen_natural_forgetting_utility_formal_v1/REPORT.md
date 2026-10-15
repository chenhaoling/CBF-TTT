# Formal natural forgetting-utility corpus V1

## Corpus and release

The formal construction labels every event in all 65 sealed natural-memory
calibration sessions.  It contains 774 decisions and the complete five-scope
by three-alpha intervention grid represented by 11 unique executions.  It uses
the frozen Qwen3-4B checkpoint, value bridge and V5 reader.  Development, test
and external benchmarks were not loaded.

The independent audit passed all eight checks, covering complete sampling,
raw per-query outcomes, oracle reconstruction, shard/input hashes, aggregate
metrics, bootstrap intervals, release gates and held-out seals.

## Counterfactual headroom

- retain-all candidate accuracy: **80.8691%**;
- safe-oracle candidate accuracy: **83.6438%**;
- gain: **+2.7747 percentage points**;
- 65-session bootstrap 95% CI: **[+2.5423, +3.0115]**;
- non-retain safe labels: **73.64%**;
- additional unrelated errors: **0**.

All twelve preregistered corpus gates passed.  The construction took 751.98
seconds on the two-GPU critical path.  Median/p90 label times were 0.260/0.599
seconds, the maximum was 21.73 seconds, and peak allocated memory was 8.09 GiB.

## Action structure

The safe label distribution is:

- `scope_top1_a0`: 325;
- `retain_all`: 204;
- `global_a0`: 119;
- `last1_a0`: 50;
- `global_a05`: 21;
- `last1_a05`: 20;
- `last2_a0`: 16;
- `hybrid_a0`: 9;
- `scope_top1_a05`: 8;
- `last2_a05`: 2.

Primary updates choose a non-retain action in 257/258 cases, usually exact
deletion of the frozen scope top.  Initial and hard-retain events retain more
often but still contain useful global, recency and partial-decay cases.  This
supports learning an action value rather than hard-coding one deletion rule.

Always applying `scope_top1_a0` reaches 81.5647% but adds 855 unrelated errors.
The non-deployable source-event rule reaches 82.0791% and adds four errors.
The safe oracle reaches 83.6438% without added unrelated errors.  These three
points define the controller target: improve over the strongest fixed action
while learning a safety barrier.

## Next stage

The corpus authorizes grouped nested OOF training of the preregistered
Barrier-Q controller.  Barrier-Q predicts future candidate gain, NLL benefit
and unrelated-memory risk for every action, then selects the highest-value
action inside the predicted safe set.  Natural development remains sealed
until its calibration gate passes.

The raw 226 MB corpus remains on hku-gpu2 and will be preserved in the ignored
compressed archive beside this report.  Readable summary, diagnosis, resource,
status and audit files are tracked here.

