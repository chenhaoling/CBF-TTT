# V13 DART capacity stop report

## Decision

The official DART source was locked before parsing and the registered
model-free feasibility stage was reconstructed exactly by a second process.
V13 stops before corpus construction or Qwen execution because no relation
meets the immutable target-material requirement.

## Locked source

- official repository commit:
  `bb644b12ec1d0264f96a31145534b9a94f6ff5f5`;
- eight DART v1.1.1 files totaling 56,328,262 bytes;
- source-manifest SHA256:
  `dc242d6873951b774d36c02783b546a265f2d645f5c236c9d51a176c187ad558`;
- only `dart-v1.1.1-full-train.json` was parsed; and
- development/test files, model checkpoints and sealed benchmarks were not
  opened.

## Capacity result

The frozen filter required exactly one input triple, one exact literal
subject/value occurrence and a human annotation from `WikiTableQuestions_lily`,
`WikiSQL_lily` or `WikiTableQuestions_mturk`. It produced 786 aligned rows and
783 rows after pair deduplication and prior-corpus exclusion, spread across 435
relations. Twenty-five relations could serve as small background donors, but
zero relations had the 140 distinct pairs and seven templates required for a
target relation. Consequently the exact allocator selected no relation and
allocated zero groups.

All reconstruction checks passed: the decision, eligible rows, exclusions,
empty material allocation and registered hashes were reproduced exactly.

## Interpretation

This is a source-capacity failure, not an algorithm result. DART's open-domain
human subset distributes a small number of single-triple examples over many
table-specific relation names. The frozen V11 controllers were never applied.
The next candidate must retain natural text while providing repeated relation
types at substantially larger scale.
