# Original continual-data availability verdict

Captured on 2026-10-19 before the corrected public-data corpus completed.

## Primary-source findings

The paper's Appendix 9.1 says that the authors collected the continual
pretraining data themselves. It describes the distribution at category level:
the short-document component resembles pretraining data, while the long
component combines books, repository-level code, retrieval-augmented synthetic
data and long-context QA constructions. It defines 32K and 128K subsets, but
does not publish record identifiers, source revisions, mixture weights,
deduplication rules, preprocessing code, ordered samples or artifact hashes.
See the [paper](https://arxiv.org/abs/2604.06169).

The official repository at frozen upstream commit
`be2324829b0e91c8fd10a74d4b43714fde6676e1` instructs users to provide their
own processed dataset through `data.train_path`. Its tracked tree contains no
data or preprocessing path, and the recommended Qwen config uses a placeholder
training path. See the [official repository](https://github.com/ByteDance-Seed/In-Place-TTT)
and `official_repo_snapshot.txt` in this directory.

The official materials do disclose enough to align the model, two stage
sequence lengths and approximate token budgets, AdamW learning rate and weight
decay, Stage-2 YaRN, convolution size, every-sixth-layer placement, inference
delta clipping and OpenCompass RULER evaluation. Those items are audited
elsewhere in the baseline chain.

## Verdict

As of this snapshot, an exact record-level reproduction of the paper's
continual-pretraining data cannot be executed from public artifacts. The
current FineWeb-Edu plus LongCrawl64 construction is therefore classified as:

`public-data matched replication (Level B)`

It is not classified as:

`private-corpus exact reproduction (Level C)`

FineWeb-Edu supplies a public short/general component and LongCrawl64 supplies
public long natural documents. Their equal-token mixture is a frozen project
choice, not a paper-disclosed mixture. Results may be compared with Table 1
only when this data difference and the reduced training scale are stated.

Level C would require the authors' record-level 32K/128K mixture, preprocessing
and missing recipe details, followed by the approximately 20B plus 15B matched
training curriculum. A later official data release would require a new
pre-outcome plan and new hashes; it must not be silently substituted into the
active chain.

This availability verdict does not authorize training, accept a formal
release, or open update-admission experiments.
