# In-Place TTT baseline completion matrix

Status: **live evidence index; not an execution plan or release record**  
Initial snapshot: 2026-10-19 (Asia/Shanghai)

## 1. Purpose

This file defines the evidence needed to close the active baseline objective
without weakening any frozen plan. It does not authorize a run, change a hash,
accept a formal release, or replace an independent audit. The experiment is
complete only when every required row below has authoritative passing evidence.

The current executable target is the public-data, resource-constrained Level-B
replication on `hku-gpu2`. It uses Qwen3-4B-Base, token-balanced FineWeb-Edu and
LongCrawl64, matched no-TTT/native-TTT arms, and the official OpenCompass RULER
task set. The private paper corpus and missing recipe details remain unavailable,
so no Level-C or exact-paper-number claim is permitted.

## 2. Required evidence

| Stage | Required completion evidence | Current state at initial snapshot |
|---|---|---|
| Released-code Level A | Clean upstream revision; environment manifest; paired finite optimizer steps; official conversion/reload; audited RULER score extraction | **Superseded for the Base claim.** The old post-trained-Qwen smoke proves only that the execution path can run. The matched Level-A smoke must be repeated with the corrected Base checkpoint and corpus. |
| Public source inventory | Passing independent inventory audit binding every FineWeb-Edu and LongCrawl64 source artifact | **Complete for Base.** The rebuilt Base-bound inventory covers 140 FineWeb-Edu files and 256 LongCrawl64 shards; all seven registered checks pass. |
| Continuous packing pilot | Passing independent two-stage audit proving exact lengths, source balance, continuity, no replay, hashes and token totals | **Complete for Base.** The rebuilt 32K-to-128K pilot has 144 records, 6,291,456 tokens and zero synthetic EOS; all ten registered audit checks pass. Old post-trained-tokenizer transform/dataloader probes are superseded and must be repeated on the corrected corpus. |
| Full public corpus | Atomically published full directory; no `.partial`; exact 724,800 records and 34,999,369,728 training tokens; passing canonical `independent_audit.json` with all registered checks true and exact manifest/dependency bindings | **Running with Base tokenizer.** The CPU-only builder is active on `hku-gpu2` in a new immutable root; no corrected canonical full audit exists yet. The old partial corpus is superseded. |
| Qwen3-4B-Base identity correction | Passing identity audit for an exact pinned `Qwen/Qwen3-4B-Base` checkout; corrected tokenizer/corpus decision; old post-trained-checkpoint chain contained and superseded | **Complete.** Exact revision `906bfd4b...` passes nine identity checks; all 14 old sessions are absent. Reserved-token literals in both old stages require a full source rebuild. |
| 8K activation-offload gate | Both corrected-Base matched arms complete one optimizer step; finite loss/gradient; passing child and paired audits; at least 5% GPU headroom; registered host-memory and swap limits | **Waiting for the Base identity and corrected-corpus gates.** The old watcher chain is superseded. |
| Stage-1 resource ladder | Audited paired 16K and 32K boundaries from the same registered inputs, no-TTT before TTT, with all resource stop rules satisfied | **Waiting for the audited 8K gate.** |
| Stage-1 timing gate | Audited paired three-step 32K GB64 timing run on the full corpus; finite measured projections for both arms and the sequential pair | **Waiting for the audited Stage-1 ladder and full corpus audit.** |
| Formal Stage-1 release | A separately written `accepted=true` release binding the exact passing timing-audit SHA256, reviewer, measured projection and decision note | **Sealed.** The release file must not be created before measured timing review. |
| Formal Stage-1 training | Both registered 27-step arms complete the same 56,623,104-token stream; final DCP and converted HF checkpoints exist; passing paired independent audit | **Not run.** |
| Stage-2 resource ladder | Each arm starts from its own audited Stage-1 HF checkpoint; paired 64K and 128K boundaries pass their child and aggregate audits | **Waiting for audited formal Stage 1.** |
| Stage-2 timing gate | Audited paired multi-step 128K timing run on the full corpus with finite measured projections | **Waiting for the Stage-2 ladder.** |
| Formal Stage-2 release | A separately written `accepted=true` release binding the exact passing Stage-2 timing audit and review decision | **Sealed.** |
| Formal Stage-2 training | Both registered five-step arms complete the same 41,943,040-token stream from their corresponding Stage-1 checkpoints; final DCP/HF checkpoints and paired audit pass | **Not run.** |
| Primary RULER resource gate | Paired no-TTT/native-TTT resource runs pass for every registered context from 4K through 128K with identical task inputs and required headroom | **Waiting for audited formal Stage 2.** Frozen-environment import resolution already confirms both arms use the clean upstream native-TTT model classes rather than the later CBF-modified tool copy. |
| Primary RULER matrix | All 13 official tasks, 100 examples per task, seed 42 and deterministic decoding complete for both arms at 4K, 8K, 16K, 32K, 64K and 128K; all 15,600 raw predictions and the paired matrix audit pass | **Not run.** |
| Separate RULER 256K extrapolation | Its paired resource audit passes; an explicit timing/resource-bound release is accepted; both 13-task, 100-example arms and the independent audit pass | **Sealed and excluded from the primary 4K--128K mean.** |
| Baseline conclusion | Report per-task and macro scores, `TTT - no-TTT` at every length, paired input counts/failures, resource measurements, hashes, confidence limits where registered, and differences from Table 1 under the public-data claim boundary | **Incomplete.** Smoke scores cannot satisfy this row. |
| Scale and feasibility boundary | Label the 98,566,144-token pair only as the first resource-constrained end-to-end tier; report measured 32K/128K projections; either register and execute a larger tier or document why the approximately 35B-token Level-B run is infeasible on the user-restricted hardware | **Incomplete.** The old 100M selected-stream composition was computed from superseded packing and must be rerun after the corrected full corpus audit. The tier cannot be relabelled as paper-scale training. |
| Update-admission exploration | A new frozen gate plan may start only after the corrected-Base matched 100M milestone and registered RULER matrix have passing audits and the scale/claim boundary above is explicitly reported; any approximately 1B expansion requires its own pre-outcome addendum | **Sealed.** No baseline outcome currently authorizes gate training or test release. |

## 3. Completion rules

The following evidence is insufficient on its own:

- a live tmux session or state file without its process handle;
- a smoke score, one-example resource score, or shortened context subset;
- one completed arm without its matched peer;
- a summary without raw predictions and an independent recomputation;
- a checkpoint without its manifest, conversion evidence and independent audit;
- a public-data gain described as reproduction of the private paper corpus;
- a 256K extrapolation substituted for a missing primary 4K--128K point;
- an unreviewed or automatically inferred formal release.

The 100M paired tier and registered 4K--128K matrix are the first required
end-to-end milestone, not a substitute for the paper's approximately 35B-token
continual-pretraining scale. They can close the resource-constrained baseline
prerequisite for later gate work only together with an explicit measured
feasibility and claim-boundary report. Expansion to the approximately 1B tier
requires its own addendum after the 100M result. A full public-data Level-B
scale claim requires the registered 35B curriculum; exact Level-C reproduction
remains unavailable unless the private corpus and missing paper settings are
obtained and executed.

## 4. Authoritative live locations

- Correct Base model: `/home/ctj/models/Qwen3-4B-Base`
- Correct Base source inventory: `/home/ctj/in_place_ttt_public_corpus_inventory_qwen3_4b_base_906bfd4b_20261019`
- Correct Base pilot: `/home/ctj/in_place_ttt_public_corpus_pilot_qwen3_4b_base_906bfd4b_20261019`
- Correct Base full corpus: `/home/ctj/in_place_ttt_public_corpus_qwen3_4b_base_906bfd4b_20261019`
- Correct full-corpus wrapper status: `/home/ctj/in_place_ttt_public_corpus_qwen3_4b_base_run_20261019/status.txt`
- Correct full-corpus tmux session: `public_corpus_qwen3_4b_base_full_20261019`

The prior checkpoint-bound corpus, activation, Stage-1, Stage-2 and RULER
paths are preserved but superseded. No path in that chain is an authoritative
live input for the corrected Base reproduction.

All GPU execution is restricted to `hku-gpu2`. The paths above are evidence
locations, not permission to bypass any predecessor, release or audit gate.
