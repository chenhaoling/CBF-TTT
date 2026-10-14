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

- `manifest.json`: `6dbe6154ecd9a1c06cb1b60be132d9eac542f8a20f6f86006b43c0c539802981`
- `pilot.jsonl`: `e9a61d5f31227a828311a2fc739bf31ff65435275d228ad2d6cb6d213a0d0222`
- `formal.jsonl`: `352c2cabef9f7ece3d879c3e65bf928674de9341e50a99ffbc738adec2e7748c`

The source manifest and train/tune/text hashes are recorded inside
`manifest.json`.  The manifest explicitly records `model_scored=false` and all
external-access flags as false.

## Pre-model correction

The first build stopped before writing a sample because it required every
selected episode to contain a same-relation hard-retain event.  A source-only
coverage audit found 252 of 272 subject-length cells with such an episode.  The
failed directory is preserved remotely as
`/home/ctj/cbf_ttt_mquake_qwen_outcome_v16_failed_ineligible_safety_20261014`.
The deterministic matched-first fallback rule was frozen before any Qwen score
or V15 outer-fold metric was available.

