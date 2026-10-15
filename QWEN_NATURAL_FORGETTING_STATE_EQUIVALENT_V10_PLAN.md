# State-equivalent safe-delete V10 preregistration

## Motivation and corrected physical target

V7 missed its absolute safety gate by one false-delete row. A reconstruction of
all five false deletes found three genuinely unsafe cases and two duplicate
Angora states. The duplicate states have different constructed chain IDs and
paraphrased prompts, but the same source fact and the same target value
`Ankara`. Delete-then-write replaces one representation of the fact with an
equivalent representation and therefore preserves the readable memory state.

V5--V7 defined safe deletion only as removing the old version of a local
lifecycle update. V10 corrects that physical target. A deletion is safe when
either:

1. the existing lifecycle label says the reader top is the stale version of
   the incoming local-update chain; or
2. the incoming and reader-top records have the same source-row SHA-256 and
   exact target value, making the transition idempotent up to duplicate
   representation.

Source identity, target value, chain ID and event kind construct labels only.
They remain unavailable to the inference controller.

## Fixed model and validation protocol

V10 makes no architecture, feature, threshold-grid, optimizer or fold change.
It reuses V7's ten no-schedule state features, five prompt-derived entity-anchor
features, linear/hidden-16/hidden-32 heads, three seeds, finite optimizer grid,
entity-barrier switch, nested five-fold session isolation and thresholds
0.50/0.70/0.80/0.90/0.95/0.98/0.99. Every outer model is trained without its
evaluated sessions.

The previously attempted V8/V9 fact-key models are not loaded. ZsRE development,
source test and external benchmarks remain sealed.

## Registered gates and next stage

The offline cross-fit passes only with:

- precision at least 98%;
- recall at least 50%;
- at most four genuinely state-changing false deletes;
- at least 95% precision in every active outer fold;
- deletion rate between 5% and 50%;
- exact session isolation and a passing independent audit.

A pass authorizes a new cross-fit on-policy calibration replay. It does not open
test. The replay must improve candidate accuracy over retain-all by at least five
points with a positive session-bootstrap 95% lower bound, add at most four
state-changing errors, have nonnegative gain in every fold, and preserve final
generation. Duplicate-equivalent replacement is audited separately and cannot
be counted as an error or as a stale-update success.

Only a passing audited replay may freeze V10 and authorize the one-time source
test. Failure stops this controller line; V10 cannot change the equivalence rule,
features or thresholds after seeing its cross-fit result.
