# On-policy sequential natural forgetting V3

## Registered result

The frozen Factor Gate V2 ensemble was evaluated once on 64 subject-disjoint
development sessions while every decision changed the memory used by later
decisions.  The scientific gate stopped before test because one of ten checks
failed: the primary controller added more collateral errors than the
privileged source-action rule.

| Policy | Accuracy | Final accuracy | Final generation | Added / recovered errors |
|---|---:|---:|---:|---:|
| Retain all | 84.82% | 71.88% | 60.74% | 0 / 0 |
| Always scope-top | 28.89% | 27.54% | 17.38% | 3144 / 280 |
| Source-action rule | 96.93% | 96.19% | -- | 0 / 620 |
| Factor ensemble | **93.32%** | **89.84%** | **75.78%** | **113 / 548** |
| Benefit-only | 91.05% | 90.72% | -- | 305 / 624 |
| No-schedule ablation | **93.54%** | 89.84% | -- | **27 / 473** |

The factor ensemble improved over retain-all by 8.50 points with a paired
session-bootstrap 95% CI of [7.48, 9.41] points.  All three seed members
improved retain-all.  The risk gate reduced collateral errors from 305 to 113
and improved accuracy by 2.27 points relative to benefit-only.  It also raised
final fixed-length generation exact match by 15.04 points over retain-all.

Nine of ten registered checks passed.  The failed check required primary
collateral errors to be no greater than the source-action rule; the values were
113 versus zero.  The ablation cannot replace the primary after results were
opened, even though its 93.54% accuracy and 27 added errors identify a stronger
candidate for subsequent confirmation.

## Failure localization

The exploratory score-free diagnosis found that all 113 added errors occurred
because the current active record had been deleted, rather than because the V5
reader misread a present record.  The primary ensemble made:

- 166 correct local deletions;
- 74 missed local deletions;
- 16 wrong-top deletions on local updates;
- 42 false deletions on retain events, including 41 hard-retain events.

No-schedule made 161 correct local deletions, missed 95, made no wrong-top
local deletion and made 29 retain false deletions.  Its 27 added errors all
occurred at the initial checkpoint.  The next experiment therefore targets
on-policy action-risk calibration and cross-view consensus; it does not modify
the frozen reader or writer.

## Audit recovery

The initial auditors rebuilt one combined 64-session embedding table while
collection used two registered 32-session tables.  Different pair ordering
changed GPU batches and nearly tied cosine values.  Both failed audit JSON
files are retained.  The repaired auditor reconstructs the exact registered
shards and changes no scores, policies, thresholds, summary or gate.  It passed
all 10 checks, including complete state transitions, features, checkpoint
probabilities/actions, query coverage, generation consistency, hashes,
bootstrap intervals and held-out seals.

## Resources and held-out state

The two-GPU critical path was 740.95 seconds. Peak allocated memory was
8.23/8.17 GiB and peak reserved memory was 11.57/9.75 GiB. ZsRE test and all
external benchmarks remained unopened. The source manifest reports a separate
unscored test split with 129 sessions and 516 subjects for later confirmation.
