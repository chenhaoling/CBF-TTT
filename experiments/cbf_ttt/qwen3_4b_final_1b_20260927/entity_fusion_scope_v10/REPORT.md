# Entity-fusion scope V10 report

## Protocol

V10 was the final preregistered CounterFact score-fusion experiment. It appended
the fold-specific V9 entity probability, log probability, and logit to V8's
thirteen text-only features, then fit the same nested five-fold rank-16
calibrator. Inference used raw query and candidate text only. CounterFact
remained observed development data; LongMemEval-S and LoCoMo remained sealed.

## Results

| Metric | Result |
| --- | ---: |
| CounterFact local-update groups | 785 |
| Top-1 accuracy | 94.65% |
| MRR | 97.29% |
| Gain over V8 | +0.64 points |
| Paired-bootstrap 95% CI versus V8 | [-0.25, 1.66] points |
| Gain over V4 | +11.72 points |
| Paired-bootstrap 95% CI versus V4 | [9.04, 14.52] points |
| Runtime | 50.95 s |
| Peak allocated / reserved GPU memory | 8.65 / 10.53 GiB |

The registered gate failed: top-1 was below 95%, the gain over V8 was below one
point and its confidence interval crossed zero, and the gain over V4 was below
twelve points. MRR and the positive confidence interval versus V4 passed.
Following the preregistered stop rule, no further fusion of the existing V8/V9
scores is allowed.

## Post-hoc error audit

The 42 incorrect predictions were joined to the frozen scoped-delta corpus only
after the primary metrics were computed. Forty-one selected a
`same_subject_different_chain` candidate and only one selected a
`different_subject` candidate. All 42 target, query, and predicted candidate
subject strings were found in their source text. All 42 selected candidates had
a different masked relation string from the query. The oracle-segmented V5
resolver was correct on 32 of these 42 errors.

This evidence revises the next-stage hypothesis. The remaining error is not
primarily entity identity confusion. Automatic subject-span discovery is needed
to remove the subject tokens before relation classification. The next model
therefore learns token-level subject spans and uses them to construct
de-identified relation views; it does not merely append another entity score.

The post-hoc audit is stored in `error_audit.jsonl` with SHA-256
`03b91af56aa4e5fdef157b6ca5b1516949c47d931c26f189b1b045d5574c65a8`.

## Decision

V10 is a negative result for additional score fusion and a positive diagnostic
for factorized scope resolution: entity and relation must be separated before
relation encoding. V11 will be a subject-disjoint token-span experiment with a
new preregistered gate. The causal forgetting-policy experiment remains locked
until the automatic resolver passes that gate.
