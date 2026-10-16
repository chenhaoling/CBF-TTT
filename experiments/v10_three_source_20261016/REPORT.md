# V10 three-source controller result

## Decision

V10 passed all integrity and safety audits but failed its preregistered effect
gate.  The model is not released and the current development domains will not
be used for another weight or threshold search.

## Development result

| Training arm | TempLAMA gain | LAMA gain |
|---|---:|---:|
| TempLAMA + LAMA + FewRel, equal source loss | +2.834 pp | +1.146 pp |
| FewRel removed, equal source loss | +2.595 pp | +0.981 pp |
| Locked V8 comparison | +4.214 pp | +1.189 pp |

Adding FewRel therefore caused a positive +0.239/+0.165 point change, but the
LAMA removal effect was below the registered +0.2-point threshold.  The
primary controller remained 0.043 LAMA point and 1.380 TempLAMA points below
V8.  It closed 65.76% and 59.46% of the two safe-oracle gaps.

Every seed improved both domains, every selected policy added zero unrelated
errors, and overwrite/stale-repetition effects were nonnegative.  The negative
release decision is an effect-size failure rather than an integrity or safety
failure.

## Training support

- TempLAMA: 4,392 examples, 685 benefit positives, 201 risks.
- LAMA: 4,684 examples, 747 benefit positives, 160 risks.
- FewRel: 6,144 examples, 1,105 benefit positives, 420 risks across all 672
  groups and 48 relations.

All six checkpoints and both-domain results were independently reconstructed.
The next registered action is a frozen, no-refit diagnostic; it cannot release
V10 or authorize another search on the opened development domains.
