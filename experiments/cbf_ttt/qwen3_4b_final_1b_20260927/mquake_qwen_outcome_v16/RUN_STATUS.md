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

