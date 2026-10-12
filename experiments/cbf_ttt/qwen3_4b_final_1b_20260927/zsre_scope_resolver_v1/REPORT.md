# ZsRE frozen cross scope resolver v1

## Objective

Test whether a linear probe over a frozen Qwen joint encoding of one memory prompt and one query can replace symbolic update-scope cues. The primary head chooses among up to eight session keys and one `defer` action. `defer` means leave the query to the frozen model rather than read the session update.

## Protocol

- Existing subject-disjoint ZsRE calibration/development/test split remains frozen.
- Calibration sessions are deterministically divided into 27 fit and seven checkpoint-selection sessions.
- Input contains only a memory prompt and query under one frozen template.
- Frozen final Qwen3-4B checkpoint; normalized final-token hidden state.
- Linear pair score plus one learned defer logit.
- Three initialization seeds; 200 full-batch epochs.
- Checkpoint selection only among epochs 10/25/50/100/200 on calibration-tune.
- Structured bi-encoder cosine retrieval is the control.

The first launch at commit `f601834` stopped before any optimizer step or development/test load because the last calibration session has three rather than eight facts. Commit `00f949f` added padding and invalid-logit masking without changing the method, data, hyperparameters, or gates. Nine server tests passed before the recovery run.

## Development result

| Resolver | Overall | Minimum seed | Exact | Rephrase | Locality/defer | NLL |
|---|---:|---:|---:|---:|---:|---:|
| **Bi-encoder cosine** | **94.51%** | **89.70%** | **100.00%** | **98.38%** | 89.83% | **0.1646** |
| Frozen cross linear | 82.12% | 82.12% | 84.64% | 53.56% | **95.13%** | 0.6064 |

All three linear seeds selected epoch 200. Their calibration-fit/tune accuracy at that epoch was 82.82/81.70%, so the head did not merely fail after checkpoint transfer. The primary improved locality over cosine by 5.62 points, source-session bootstrap 95% CI `[2.49, 9.11]`, but reduced exact by 15.69 points and rephrase by 45.22 points. Overall loss relative to cosine was 12.42 points, CI `[-14.61,-10.00]`.

Exact, rephrase, control-margin, and control-confidence gates failed. Test remained sealed.

## Error decomposition

Across the three bi-encoder seeds, exact was correct for all 801 queries. Rephrase had 788 correct, 11 false deferrals, and only two wrong-key errors. The main remaining error was 163 locality false retrieves.

The cross-linear head created 117 exact and 357 rephrase false deferrals, in addition to 21 wrong-key errors. It therefore damaged an already reliable candidate-key ranking while learning to defer more often.

## Interpretation

The structured cosine control is stronger than the earlier associative readout and nearly passes the locality gate. Candidate selection and scope rejection are empirically different subproblems: cosine already identifies the correct key, while the cross-linear head fails when asked to relearn both key identity and deferral from scratch.

The next experiment will freeze ensemble cosine candidate selection and train only a binary `retrieve`/`defer` gate from top-candidate pair evidence and score geometry. This follows the observed error decomposition and preserves the strong retrieval prior.

## Runtime and audit

- Recovery run: 2026-10-12 17:26:53 to 17:27:47 Asia/Shanghai.
- Task time: 50.06 seconds.
- Peak CUDA allocated/reserved: 8.497/10.988 GiB.
- Formal code commit: `00f949f5c96f34af54c755bedb94ec485e5def2e`.
- Independent source, calibration split, feature/hash, raw-record, aggregation, comparison, gate, and conditional-test audit: passed.

## Artifacts and claim boundary

Compact artifacts are archived here; the head checkpoint and raw JSONL files remain at `/home/ctj/cbf_ttt_zsre_scope_resolver_v1_r1`. This result supports the modular diagnosis above. It does not support test performance, official ZsRE generation, factual correctness after deferral, native dense TTT writing, or natural-sequence forgetting benefit.
