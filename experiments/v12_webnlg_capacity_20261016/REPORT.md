# V12 WebNLG capacity report

The official enriched WebNLG source was locked at commit
`711a8ca37eff2d694d664edcaeb07b0ccf8f56c2` before XML parsing.  The source
manifest covers 185 English v1.6 files and 44,390,346 bytes.

The registered `train/1triples` inspection found 1,788 entries, 4,338 human
lexicalizations, 4,327 valid aligned templates and 227 relations.  Eight
relations passed the coarse 28-row/20-subject/20-value/7-template gate.

Formal construction is infeasible under the registered global entity-pair
uniqueness rule.  Fourteen groups need at least six factual pairs per group,
or 84 pairs per relation, before template and stable-background donors.  The
eight capacities are:

| Relation | Unique pairs |
|---|---:|
| country | 119 |
| creator | 35 |
| ingredient | 63 |
| isPartOf | 102 |
| language | 44 |
| leaderName | 76 |
| location | 82 |
| runwayLength | 48 |

Only two relations meet the immutable lower bound.  V12 stopped before corpus
construction, Qwen loading, labels, features or policy evaluation.  No rule
was relaxed and all previously sealed benchmarks remain unopened.
