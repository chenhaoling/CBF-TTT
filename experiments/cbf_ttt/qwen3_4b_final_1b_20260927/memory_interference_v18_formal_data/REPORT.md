# V18 formal controlled-interference corpus

## Status

- Terminal status: `completed_controlled_interference_v18_formal_data`
- Runner exit code: `0`
- Independent construction audit: passed, 11/11 checks
- No formal row had been scored when this corpus was frozen.
- No external benchmark was loaded.

## Frozen split

- 320 episodes
- 80 subjects, four mechanisms per subject
- 20 formal-only template groups
- Five outer folds, 64 episodes per fold
- 80 episodes at each memory length: 8, 16, 32 and 64
- 80 episodes for each mechanism
- 9,280 single-session deletion states
- 1,920 historical-key deletion states
- 1,600 fixed-arm entries

Every subject and template group belongs to exactly one fold.  All entity
prefixes, 2,560 answer values and surface templates are disjoint from the
admitted 40-row pilot.

## Frozen hashes

- Manifest: `8312c232a10e8e7f66cb069bd4738c99dbc2dcfe57dd01297f65c6e80a129de6`
- Formal episodes: `1975f8c5ca221d707f25665b8974cf1435eae63a62285a57ae8076efd0b365f2`
- Formal states: `87b4727324282e6f608e642674d9e5521924bae454a384633cf148959f940192`

The first runner attempt is not part of this archive.  It failed before
writing any row because the runner created the builder output directory in
advance.  This archive is the clean `_retry1` run with the unchanged protocol
and seed.
