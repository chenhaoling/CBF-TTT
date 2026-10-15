# Natural forgetting-utility pilot V1

## Question and protocol

This calibration-only pilot tests whether interventions on frozen V5 session
memory expose enough safe downstream utility to supervise a learned forgetting
controller.  It evaluates 94 sparse decision points from 16 sealed calibration
sessions with five memory scopes and the retention-strength grid
`{0, 0.5, 1}`.  The Qwen3-4B checkpoint, native writer, value bridge and V5
reader are frozen.  Development, test and external benchmarks were not loaded.

The label is the best future-query action that introduces no additional error
on chains unrelated to the incoming event.  Candidate correctness is the
primary objective; answer NLL and memory mass are lexicographic tie breakers.

## Result

- retain-all future candidate accuracy: **78.3251%**;
- safe-oracle candidate accuracy: **82.4302%**;
- safe gain: **+4.1051 percentage points**;
- session-bootstrap 95% CI: **[+3.5417, +4.7461]**;
- non-retain labels: **84.04%**;
- added unrelated errors: **0**;
- selected actions: **8**;
- dominant action: `scope_top1_a0`, **62/94 labels**.

Every preregistered scaling check passed.  The confidence interval also passed
the stronger evidence-ready criterion.  The independent audit passed all eight
checks, including exact sampling reconstruction, raw outcome aggregation,
oracle reconstruction, artifact seals and held-out-data seals.

## Resource profile

- two-GPU critical path: **173.63 seconds**;
- median label time: **0.254 seconds**;
- p90 label time: **18.21 seconds**;
- maximum label time: **21.29 seconds**;
- peak allocated memory: **8.06 GiB**;
- peak reserved memory: **10.86 GiB**.

The long-tail times occur on the first uncached label of a session.  Subsequent
labels reuse identical query/value/strength Qwen outcomes, which is the
registered formal construction path.

## Interpretation and release

The result establishes usable causal headroom for selective session-memory
forgetting.  It does not establish a learned-controller gain because the
reported policy is an oracle over counterfactual outcomes.  The result
authorizes the preregistered full 65-session, 774-event calibration corpus,
followed by grouped out-of-fold controller training.

The complete raw result is preserved in the ignored tar archive beside this
report.  Readable hashes, resources, summary, status and audit artifacts are
tracked in this directory.

