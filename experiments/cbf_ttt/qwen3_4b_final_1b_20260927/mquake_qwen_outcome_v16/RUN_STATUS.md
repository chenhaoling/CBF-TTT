# MQuAKE frozen-Qwen outcome V16 run status

## Current state

The source-only sample construction is complete.  No Qwen outcome has been
scored and no external benchmark has been loaded.  Pilot execution remains
conditional on the independently audited V15 pair and safety gate.

## Frozen samples

| Set | Rule | Rows | Same-relation hard | Marked fallback |
|---|---|---:|---:|---:|
| pilot | two minimum-hash episodes per V15 fold x length | 40 | 40 | 0 |
| formal | one minimum-hash episode per subject x length, preferring matched hard | 272 | 252 | 20 |

The formal sample covers all 68 MQuAKE-T train+tune subjects at lengths
16/32/48/64.  The 20 fallback rows belong to subject-length cells with no
same-relation hard-retain episode anywhere in that cell; they retain the
original corpus builder's hard-retain marker and will be reported separately.

## Artifact hashes

- `manifest.json`: `9eea25d6aa253014d47553e0e45fb01107546a3ff14b86bc3881b6c60b208be1`
- `pilot.jsonl`: `4a66ff3e2adbc1c22466f876149855d62fbf2863a124ea8d292e2861359bbe38`
- `formal.jsonl`: `4c184db9568ad93e04c88f343bd3372d2cb3796d4035e4834938db9bbbaa9a11`
- `data_audit.json`: `9840a50489a3cd9fe1dd7a39071af535b078b2aa4d3f172cf1a58f1d1375d744`
- `prompt_tokenization_audit.json`: `9760a068c8e28e01d5a6c0b0ead8dd721c0d74754a322ad1fd0d6a8b5054ac72`
- `generation_scoring_bridge.json`: `f43dfb473789bfc999c1d72e40182320fd2455c3ce5aad2cc979cd68b63ec0d9`

The source manifest and train/tune/text hashes are recorded inside
`manifest.json`.  The manifest explicitly records `model_scored=false` and all
external-access flags as false.

The independent source reconstruction audit passed all 12 checks: source
protocol and hashes, subject folds, ordered pilot/formal selection, sizes,
relation counts, artifact hashes, causal query visibility, pre-model state and
external benchmark seals.  This audit covers data only; Qwen outcomes still
require a separate scoring and replay audit.

The real Qwen3 tokenizer also validated all 928 pilot candidate boundaries.
The longest `oracle` prompt plus answer is 4,509 tokens and the longest
`no_memory` input is 63 tokens, both below the checkpoint's 40,960-position
limit.  This remains a tokenizer-only audit with `model_scored=false`.

## Gated pilot runner

The two-GPU pilot runner and an independent result auditor are implemented but
have not been launched.  The runner refuses to load Qwen unless V15 is
terminal, independently audited, and records
`jackknife_consensus_supported=true`.  Each of the 320 policy-query labels
(40 episodes x 2 policies x 4 query types) records wall time, candidate count,
starting and peak allocated memory, and starting and peak reserved memory.
The auditor independently reconstructs causal memory membership, prompt
hashes, candidates, predictions, NLL aggregates, disjoint shard coverage,
resource records, V15 dependency hashes, terminal state, and external-set
locks.  Formal rows remain inaccessible to this runner.

The scorer loads the final checkpoint's slow Qwen weights with
`config.ttt_mode=false` and rejects the checkpoint unless the native
`ttt_conv`/`ttt_proj` adapters are explicitly excluded.  Every answer candidate
uses a new standard `DynamicCache`.  This prevents prompts longer than the
4,096-token native update interval from silently activating the failed native
writer and keeps policy arms different only in serialized session text.

A no-data loader smoke on remote GPU1 passed before any outcome score: the
plain model has 4,022,468,096 parameters, `ttt_mode=false`, zero missing keys,
and exactly 14 expected unexpected keys (one `ttt_conv` and one `ttt_proj` for
each of the seven TTT layers).  Loading peaked at 7.545 GiB allocated and
7.547 GiB reserved on an RTX 5090.  The smoke loaded no pilot row and leaves
`model_scored=false`.

The scorer now evaluates each policy-query prompt once and clones its immutable
KV cache into independent answer branches.  A fixed-tolerance synthetic bridge
compared this implementation with the original full prompt-plus-answer forward
for `Paris` and `London`: both candidate NLL errors were exactly zero under the
pre-registered `2e-5` tolerance and the prediction was identical.  The
synthetic query took 0.328 seconds and peaked at 7.562/7.576 GiB
allocated/reserved.  Result SHA-256 is
`a81c98a535ea6bd961395357d0b98df7a8055b209cc92ef0c948b74952ec6a49`.
The first bridge invocation failed before model loading because its CLI names
did not map to the Python function names; that directory is preserved remotely
and the corrected run used a fresh `_retry1` directory.  No pilot row was
loaded in either invocation.

The optional greedy-generation path was then exercised on the same frozen
Qwen checkpoint by a second synthetic-only bridge.  Starting from a cloned
shared prompt cache, it generated the single token `Paris`, stopped at EOS,
and obtained strict and normalized exact match 1.  Candidate NLL still favored
Paris (0.5602 versus 7.6852 for London).  The measured query took 0.435 seconds
and peaked at 7.566/7.580 GiB allocated/reserved.  All eight structural,
loading, decoding, bound, metric and resource checks passed; no experiment or
external row was loaded.

## Formal outcome implementation

The gated two-GPU nine-arm formal scorer, runner and independent auditor are
implemented and tested but remain unlaunched.  They require passed V15, pilot
and frozen-state audits.  Each outer fold selects its fixed comparator from the
other four folds by primary candidate accuracy, lower answer NLL and registered
arm order.  Headline accuracy and NLL differences use subject-cluster bootstrap
with 20,000 draws and seeds 1611/1612.  The scorer reports candidate preference,
answer NLL, strict/normalized greedy exact match, per-kind/fold/subject results,
retained sessions, latency and GPU peaks.  It scores identical within-episode
memory states once and records every reused arm.  The auditor independently
reconstructs memory, prompts, tokenization, generation text, reuse provenance,
fixed-arm selection, intervals and all release checks.

## Pre-model correction

The first build stopped before writing a sample because it required every
selected episode to contain a same-relation hard-retain event.  A source-only
coverage audit found 252 of 272 subject-length cells with such an episode.  The
failed directory is preserved remotely as
`/home/ctj/cbf_ttt_mquake_qwen_outcome_v16_failed_ineligible_safety_20261014`.
The deterministic matched-first fallback rule was frozen before any Qwen score
or V15 outer-fold metric was available.

A second source-only correction fixed the query time immediately after the
primary-current session and restricted hard/unrelated safety units to earlier
visible sessions.  The first successful but pre-causal-visibility sample is
preserved remotely as
`/home/ctj/cbf_ttt_mquake_qwen_outcome_v16_previsible_fix_20261014`; the hashes
above refer exclusively to the corrected sample.

