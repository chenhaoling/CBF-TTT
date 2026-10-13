# CounterFact causal on-policy scope V13

## Objective

Test the final registered CounterFact gate architecture while keeping the V11
candidate resolver, V5 relation heads, Qwen3-4B checkpoint, and query readout
frozen. V13 replaces V12's linear retain/local gate with a rank-16 nonlinear
gate over posterior geometry and adds one round of V12-induced on-policy states.
CounterFact remains observed development data; this is grouped five-fold OOF
evidence rather than an untouched final test.

## Main result

| Metric | V12 | V13 on-policy MLP | Registered floor |
|---|---:|---:|---:|
| Candidate top-1 | 97.07% | 97.07% | 96% |
| Query accuracy | 50.92% | 51.52% | within 0.5 points of oracle |
| Overall action accuracy | 90.40% | 95.29% | 95% |
| Primary initial | 99.62% | 99.75% | descriptive |
| Hard retain | 93.50% | 96.56% | 95% |
| Primary update | 80.00% | 89.55% | 90% |

V13 passed five of the six registered checks. The only failure was primary
update accuracy, which missed its floor by 0.446 percentage points. The overall
development gate therefore failed and the registered next action is to stop
CounterFact controller tuning.

## Paired comparisons

- Against fixed retain, query accuracy improved by **1.949 points**, with
  subject-cluster bootstrap 95% CI **[1.547, 2.368]**. Action accuracy improved
  by **28.620 points**, CI **[27.601, 29.597]**.
- Against V12, query accuracy improved by **0.489 points**, CI
  **[0.245, 0.752]**, and action accuracy improved by **4.883 points**, CI
  **[3.864, 5.945]**.
- V13 was **0.463 points** below oracle-local query accuracy, CI
  **[-0.691, -0.256]**, satisfying the registered half-point proximity check.
- The on-policy arm differed from the oracle-state-only MLP by only
  **+0.045 points** in query accuracy, CI **[-0.0004, 0.102]**, and
  **+0.085 points** in action accuracy, CI **[-0.212, 0.382]**. The intervals
  include zero, so the supported gain comes from nonlinear posterior geometry,
  not from the single on-policy aggregation round.

## Interpretation

The final nonlinear gate resolves most of V12's conservative false-retain
problem without changing candidate resolution. It gives positive causal evidence
that a learned semantic-scope gate can improve a sequential memory policy over
retain-all, and its action decomposition closely matches the independently
scored ZsRE result: both obtain about 95.2% overall action, 96.3% hard retain,
and 89.4% primary update.

The strict gate still fails. This result cannot be described as a fully
successful CounterFact endpoint, and it does not justify another threshold or
architecture search on the same observed data. It also does not establish
native dense `ttt_proj`/`ttt_conv` content writing. Earlier native-writer audits
show that the final dense writer does not reliably bind content, while the
contextual low-rank Qwen memory does; subsequent work must use the readable
substrate or introduce an independently developed writer.

## Protocol integrity and resources

- Five deterministic subject-grouped outer folds with nested inner causal
  epoch/threshold selection.
- Three readout/controller seeds and subject-cluster paired bootstrap.
- No chain identity, relation ID, candidate age, or inference-time subject
  metadata was used as a model feature.
- All five folds selected epoch 50. Four folds met the inner constraints; one
  used the preregistered minimum-deficit fallback.
- Runtime: **756.28 seconds** on RTX 5090.
- Peak CUDA allocated/reserved: **9.026/10.744 GiB**.
- Terminal status: `completed_causal_onpolicy_scope_v13`.

## Artifacts

- `summary.json`: aggregate metrics, paired comparisons, gate, and terminal
  decision.
- `training.json`: selections, state counts, hashes, runtime, and GPU memory.
- `training_manifest.json`: frozen model, source, seeds, feature restrictions,
  and benchmark-sealing metadata.
- `terminal_status.txt`: completed terminal marker.

Large checkpoints and raw per-seed traces remain under
`/home/ctj/cbf_ttt_causal_onpolicy_scope_v13` on `hku-gpu2`.

## Claim boundary and next stage

Supported: nonlinear posterior geometry improves the observed CounterFact
causal policy over V12 and retain-all, while approaching oracle-local query
accuracy.

Not supported: on-policy aggregation as a distinct contribution, a passed
CounterFact final gate, official generative knowledge editing, or native dense
fast-weight memory.

The next confirmatory direction is a frozen cross-domain synthesis followed by
natural sequential evaluation on the already validated contextual low-rank
Qwen memory. CounterFact gate tuning stops here.
