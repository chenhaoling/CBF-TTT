# In-Place TTT baseline completion matrix

Status: **live evidence index; not an execution plan or release record**  
Initial snapshot: 2026-10-19 (Asia/Shanghai)
Last reconciled: 2026-10-19 09:22 (Asia/Shanghai)

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

| Stage | Required completion evidence | Current authoritative state |
|---|---|---|
| Released-code Level A | Clean upstream revision; environment manifest; paired finite optimizer steps; official conversion/reload; audited RULER score extraction | **Waiting for corrected full corpus.** The old post-trained-Qwen smoke is superseded. The new hku-gpu2-only watcher is live in `WAITING_BASE_CORPUS`, with zero training children; Base-bound sparse-input and terminal audits are frozen and tested. |
| Original continual-data availability | Paper and official-release audit establishing whether the record-level 32K/128K mixture, revisions, preprocessing and ordered samples are publicly executable | **Unavailable for exact reproduction; boundary complete.** The paper provides category-level descriptions of a self-collected mixture, while upstream `be232482...` requires a user-supplied processed path and ships no processing scripts. FineWeb-Edu/LongCrawl64 is therefore Level B public-data matched replication, never Level C private-corpus reproduction. |
| Public source inventory | Passing independent inventory audit binding every FineWeb-Edu and LongCrawl64 source artifact | **Complete for Base.** The rebuilt Base-bound inventory covers 140 FineWeb-Edu files and 256 LongCrawl64 shards; all seven registered checks pass. |
| Continuous packing pilot | Passing independent two-stage audit proving exact lengths, source balance, continuity, no replay, hashes and token totals | **Complete for Base.** The rebuilt 32K-to-128K pilot has 144 records, 6,291,456 tokens and zero synthetic EOS; all ten registered audit checks pass. Old post-trained-tokenizer transform/dataloader probes are superseded and must be repeated on the corrected corpus. |
| Full public corpus | Atomically published full directory; no `.partial`; exact 724,800 records and 34,999,369,728 training tokens; passing canonical `independent_audit.json` with all registered checks true and exact manifest/dependency bindings | **Running with Base tokenizer.** The CPU-only builder is active on `hku-gpu2`. A frozen read-only live smoke independently re-tokenized the first 16 records, validated recent provenance through index 65,354, passed all five checks and left the builder live. This is early fault detection only; no corrected canonical full audit exists yet. |
| Corrected 100M selected stream | Installed VeOmni/Hugging Face/StatefulDataLoader path replayed twice on the corrected provenance; exact per-rank ordered hashes and source counts; independent reconstruction; no plaintext/token arrays; both formal releases bind the passing audit | **Registered and waiting for the corrected full corpus.** The hku-gpu2 CPU watcher is live at `waiting_for_corrected_base_corpus`. A no-plaintext synthetic smoke passes both replay and independent audit for all 98,566,144 selected tokens. The config requests two workers, while the single JSONL shard makes the installed runtime use one effective worker; both values are recorded. The old selected-stream audit is superseded. |
| Qwen3-4B-Base identity correction | Passing identity audit for an exact pinned `Qwen/Qwen3-4B-Base` checkout; corrected tokenizer/corpus decision; old post-trained-checkpoint chain contained and superseded | **Complete.** Exact revision `906bfd4b...` passes nine identity checks; all 14 old sessions are absent. Reserved-token literals in both old stages require a full source rebuild. |
| Training-recipe semantics | Executable audit of all four reduced-scale paired configs; clean live-checkout hash agreement; Base layer count and native writer structure; explicit separation of paper requirements from public-replication assumptions | **Complete before corrected training.** The CUDA-hidden audit passes with zero errors and binds 98,566,144 tokens per arm at world size 2. Seven live tool snapshots contain byte-identical configs/auditor; the 36-layer Base model makes the registered layer set exactly every sixth layer, and clean upstream code confirms depthwise Conv1d kernel 5. This is a definition audit, not training completion or a paper-scale claim. |
| Native TTT initialization semantics | Clean upstream runtime proves Base state loading changes no shared slow weight, Conv5 starts at zero, target projection is sparse diagonal, the first native update is zero and initial logits equal no-TTT | **Complete on CPU before corrected training.** All 13 checks pass on `hku-gpu2`: exact missing-key set, bitwise-equal shared weights and logits, zero convolution/target output, finite nonzero projection diagonal and zero off-diagonal. This tiny fixture validates official initialization code only; the full 4B resource gate remains required. |
| Native writer autograd reachability | Official-logit next-token backward proves zero initialization still trains the convolution and that projection gradients open after the convolution leaves zero; no tensor payload is retained | **Complete on CPU before corrected training.** All ten checks pass: finite losses, positive first/second Conv5 gradients, exact-zero first projection gradient and positive second projection gradient after the frozen diagnostic nudge. A result-serialization typo in the first persistent attempt is recorded and superseded. |
| Liger/fused-AdamW GPU environment | On one isolated `hku-gpu2` RTX 5090, the official Liger fused loss and VeOmni fused AdamW execute two finite native-writer steps; optimizer membership and zero-start gradient staging are audited without tensor payloads | **Complete as a tiny single-GPU preflight.** All 19 checks pass on physical GPU 0: Liger forward and fused AdamW are active, the first step updates Conv5, and the second backward opens a positive projection gradient. This does not validate FSDP2, distributed clipping, full 4B memory or convergence. |
| Exact 4B checkpoint writer integration | Exact Base revision loads the six registered writer layers on one `hku-gpu2` GPU; missing-key set, initialization, initial-output equivalence and two fused-loss gradient passes are checked without saving tensors or a checkpoint | **Strict preflight failed, preserved as diagnostic evidence.** 22/24 checks pass, including exact loading and all per-layer gradient checks. CPU BF16 missing-weight initialization leaves 7--20 exact-zero diagonal entries per layer, and chunked-einsum versus linear BF16 execution yields a 0.25 maximum final-logit difference, failing the frozen nonzero/allclose gates. The failure does not authorize changing released code or bypassing Level A. |
| Formal FSDP2/meta writer initialization | Exact Base revision follows `tasks/train_torch.py` through FP32 meta construction, two-rank FSDP2 sharding, CUDA checkpoint load and missing-key initialization; both ranks independently reconstruct all six global writers | **Complete for initialization only.** All 12 checks pass on `hku-gpu2`: twelve CUDA FP32 DTensors, zero Conv5 tensors, exactly 2560 finite nonzero projection diagonal values per layer, zero off-diagonal values and identical rank reconstructions. This resolves the CPU-BF16 zero-diagonal scope but executes no forward, optimizer or checkpoint. |
| Formal FSDP2 8K writer autograd | Exact Base revision runs two 8192-token fused-loss backwards through the formal two-rank FSDP2, mixed-precision, gradient-checkpointing and activation-offload graph; only a frozen Conv5 diagnostic perturbation separates the passes | **Complete as an autograd diagnostic.** All 23 checks pass on `hku-gpu2`. The first backward gives positive Conv5 and exact-zero projection gradients at all six writers; after the registered Conv5 nudge, the second gives positive Conv5 and projection gradients at all six. Peak allocated/reserved memory is 18.51/24.00 GiB per GPU. No optimizer or checkpoint was created, so corrected Level A remains mandatory. |
| Baseline tooling regression | All registered `test_in_place_ttt*.py` tests plus the frozen-environment paper-clipping runtime test execute without GPU use | **Complete at current source.** The tracked-clean local suite passes 133 tests with one environment-only skip; that exact skipped test passes separately under the frozen `hku-gpu2` evaluation environment with CUDA hidden. Aggregate executable coverage is 134/134. This validates tooling and gates, not experimental outcomes. |
| 8K activation-offload gate | Both corrected-Base matched arms complete one optimizer step; finite loss/gradient; passing child and paired audits; at least 5% GPU headroom; registered host-memory and swap limits | **Waiting for the corrected full corpus.** The Base identity already passes; the live Level-A watcher remains at `WAITING_BASE_CORPUS` and has no training child. |
| Stage-1 resource ladder | Audited paired 16K and 32K boundaries from the same registered inputs, no-TTT before TTT, with all resource stop rules satisfied | **Waiting for corrected Level A.** The hku-gpu2-only watcher is live at `waiting_for_corrected_level_a`; its future result root is absent and no 16K/32K child has run. The clean remote focused suite passes 81/81. |
| Stage-1 timing gate | Audited paired three-step 32K GB64 timing run on the full corpus; finite measured projections for both arms and the sequential pair | **Registered and waiting for the corrected Stage-1 ladder.** The repaired hku-gpu2-only watcher is live at `waiting_for_corrected_stage1_ladder`; its clean remote worktree passes 84/84 focused tests, the result root is absent and no timing child has been released. |
| Formal Stage-1 release | A separately written `accepted=true` release binding the exact passing timing-audit SHA256, corrected selected-stream audit, reviewer, measured projection and decision note | **Sealed and implemented fail-closed.** A live CPU watcher first waits for both prerequisites and creates only an immutable non-authorizing review packet. The only supported transition rehashes and recomputes that exact packet before requiring explicit human acceptance; it is unreachable from every watcher. No review packet or release exists yet. |
| Formal Stage-1 training | Both registered 27-step arms complete the same 56,623,104-token stream; final DCP and converted HF checkpoints exist; passing paired independent audit | **Registered and waiting for corrected timing.** The hku-gpu2-only watcher is live at `waiting_for_corrected_stage1_timing`; its clean remote worktree passes 88/88 tests, while release/result paths remain absent. |
| Stage-2 resource ladder | Each arm starts from its own audited Stage-1 HF checkpoint; paired 64K and 128K boundaries pass their child and aggregate audits | **Registered and waiting for corrected formal Stage 1.** The live hku-gpu2-only watcher is at `waiting_for_corrected_formal_stage1`; its clean remote worktree passes 91/91 tests and no Stage-2 child has run. |
| Stage-2 timing gate | Audited paired multi-step 128K timing run on the full corpus with finite measured projections | **Registered and waiting for the corrected Stage-2 ladder.** The hku-gpu2-only watcher is live at `waiting_for_corrected_stage2_ladder`; its clean remote worktree passes 98 focused tests, and the result root is absent. |
| Formal Stage-2 release | A separately written `accepted=true` release binding the exact passing Stage-2 timing audit, corrected selected-stream audit and review decision | **Sealed and implemented fail-closed.** A live CPU watcher first waits for both prerequisites and creates only an immutable non-authorizing review packet. The only supported transition rehashes and recomputes that exact packet before requiring explicit human acceptance; it is unreachable from every watcher. No review packet or release exists yet. |
| Formal Stage-2 training | Both registered five-step arms complete the same 41,943,040-token stream from their corresponding Stage-1 checkpoints; final DCP/HF checkpoints and paired audit pass | **Registered and waiting for corrected Stage-2 timing.** The CPU-only watcher is live at `waiting_for_corrected_stage2_timing`; its clean remote worktree passes 102 focused tests, while release/result paths remain absent. |
| Paper inference semantics | Every paper-facing native Qwen3 fast-weight delta is independently clipped to Frobenius norm `tau=1e-5`; default released-code behavior remains opt-in; generated source, import resolution, numerical behavior and config override pass independent checks | **Complete and frozen before corrected-Base scoring.** The immutable overlay starts from all seven hash-verified clean upstream inference files, changes only Qwen3 config/modeling, passes CPU numerical and `from_pretrained` validation without CUDA, and is required by every 2x5090 single/resource/matrix audit. Released-code Level A remains explicitly unmodified and is not a paper-inference efficacy result. |
| Primary RULER resource gate | Paired no-TTT/native-TTT resource runs pass for every registered context from 4K through 128K with identical task inputs and required headroom | **Registered and waiting for corrected formal Stage 2.** The Base wrapper watcher is live at `waiting_for_corrected_formal_stage2`; it requires both Stage-2 audits and the frozen `tau=1e-5` overlay. The shared clean remote suite passes 106 focused tests and no resource child exists. |
| Primary RULER matrix | All 13 official tasks, 100 examples per task, seed 42 and deterministic decoding complete for both arms at 4K, 8K, 16K, 32K, 64K and 128K; all 15,600 raw predictions and the paired matrix audit pass | **Registered and waiting for the corrected RULER resource ladder.** A CUDA-hidden runtime preflight loaded all seven 4K--256K configs through `mmengine.Config.fromfile`; each resolved the expected 13 tasks, 100 samples, seed 42, deterministic generation and exact official summary group, and all installed module/data hashes matched the frozen auditor. The CPU-only matrix watcher remains at `waiting_for_corrected_ruler_resource`; no matrix child, raw prediction or score exists. |
| Separate RULER 256K extrapolation | Its paired resource audit passes; an explicit timing/resource-bound release is accepted; both 13-task, 100-example arms and the independent audit pass | **Full resource/review/formal chain registered.** The resource watcher waits for the corrected primary matrix; the CPU review watcher waits for corrected 256K resource; the final watcher waits for an accepted review-bound release. No packet, release, GPU child or score exists. The final wrapper requires both child paper-overlay hashes and records `update_admission_released=false`. Formal 13×100 remains sealed and excluded from the primary mean. |
| Baseline conclusion | Report per-task and macro scores, `TTT - no-TTT` at every length, paired input counts/failures, resource measurements, hashes, confidence limits where registered, and differences from Table 1 under the public-data claim boundary | **Full conclusion chain registered and waiting for corrected inputs.** The CPU-only watcher is live at `waiting_for_corrected_table1_inputs`. Its frozen reporter covers seven contexts, 13 tasks, counts/failures, resource evidence, paper differences and a seed-42 10,000-resample paired-task interval; the independent auditor recomputes JSON and Markdown exactly. No conclusion artifact exists yet. |
| Scale and feasibility boundary | Label the 98,566,144-token pair only as the first resource-constrained end-to-end tier; report measured 32K/128K projections; either register and execute a larger tier or document why the approximately 35B-token Level-B run is infeasible on the user-restricted hardware | **Projection chain registered and waiting for corrected timing.** The CPU-only hku-gpu2 watcher waits for both independently audited 32K and 128K timing gates, then reports per-arm 20B/15B and sequential matched-pair 70B wall-time projections plus 30/90-day speed requirements. It cannot authorize training, relabel the 98,566,144-token pair as paper scale, or open gate experiments. No projection artifact exists yet. |
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
- Original-data availability evidence: `experiments/baseline_reproduction_qwen3_4b/remote/official_data_availability_preflight_20261019/`
- Correct Base source inventory: `/home/ctj/in_place_ttt_public_corpus_inventory_qwen3_4b_base_906bfd4b_20261019`
- Correct Base pilot: `/home/ctj/in_place_ttt_public_corpus_pilot_qwen3_4b_base_906bfd4b_20261019`
- Correct Base full corpus: `/home/ctj/in_place_ttt_public_corpus_qwen3_4b_base_906bfd4b_20261019`
- Correct full-corpus wrapper status: `/home/ctj/in_place_ttt_public_corpus_qwen3_4b_base_run_20261019/status.txt`
- Correct full-corpus tmux session: `public_corpus_qwen3_4b_base_full_20261019`
- Corrected selected-stream watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_selected_stream_2b8644a0_20261019`
- Corrected selected-stream tmux session: `baseline_qwen3_4b_base_selected_stream_2b8644a0_20261019`
- Corrected Level-A output root: `/home/ctj/in_place_ttt_qwen3_4b_base_level_a_6a3bb2bf_20261019`
- Corrected Level-A tmux session: `baseline_qwen3_4b_base_level_a_6a3bb2bf_20261019`
- Corrected Stage-1 ladder watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_stage1_ladder_97c4d786_20261019.watch`
- Corrected Stage-1 ladder tmux session: `baseline_qwen3_4b_base_stage1_ladder_97c4d786_20261019`
- Corrected Stage-1 timing watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_stage1_timing_226a1198_20261019.watch`
- Corrected Stage-1 timing tmux session: `baseline_qwen3_4b_base_stage1_timing_226a1198_20261019`
- Corrected formal Stage-1 watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_100m_stage1_99774817_20261019.watch`
- Corrected formal Stage-1 tmux session: `baseline_qwen3_4b_base_100m_stage1_99774817_20261019`
- Future corrected Stage-1 release: `/home/ctj/in_place_ttt_qwen3_4b_base_100m_stage1_release_99774817_20261019.json`
- Corrected Stage-1 release-review watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_stage1_release_review_d6d47cb0_20261019`
- Corrected Stage-1 release-review tmux session: `baseline_qwen3_4b_base_stage1_release_review_d6d47cb0_20261019`
- Corrected Stage-2 ladder watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_stage2_ladder_1098b08d_20261019.watch`
- Corrected Stage-2 ladder tmux session: `baseline_qwen3_4b_base_stage2_ladder_1098b08d_20261019`
- Corrected Stage-2 timing watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_stage2_timing_80c414c7_20261019.watch`
- Corrected Stage-2 timing tmux session: `baseline_qwen3_4b_base_stage2_timing_80c414c7_20261019`
- Corrected formal Stage-2 watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_100m_stage2_a21afd27_20261019.watch`
- Corrected formal Stage-2 tmux session: `baseline_qwen3_4b_base_100m_stage2_a21afd27_20261019`
- Future corrected Stage-2 release: `/home/ctj/in_place_ttt_qwen3_4b_base_100m_stage2_release_a21afd27_20261019.json`
- Corrected Stage-2 release-review watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_stage2_release_review_d6d47cb0_20261019`
- Corrected Stage-2 release-review tmux session: `baseline_qwen3_4b_base_stage2_release_review_d6d47cb0_20261019`
- Reviewed-release transition implementation: commit `ccf17a70bc41b899b976d69be03730abfdebb36a` (CPU-only; never called by a watcher)
- Corrected RULER resource watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_ruler_resource_c89c6f8c_20261019.watch`
- Corrected RULER resource tmux session: `baseline_qwen3_4b_base_ruler_resource_c89c6f8c_20261019`
- Corrected RULER matrix watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_ruler_matrix_c89c6f8c_20261019.watch`
- Corrected RULER matrix tmux session: `baseline_qwen3_4b_base_ruler_matrix_c89c6f8c_20261019`
- Corrected RULER semantics preflight evidence: `experiments/baseline_reproduction_qwen3_4b/remote/hku_gpu2_qwen3_4b_base_ruler_semantics_preflight_20261019/`
- Corrected training-semantics preflight evidence: `experiments/baseline_reproduction_qwen3_4b/remote/hku_gpu2_qwen3_4b_base_training_semantics_preflight_20261019/`
- Native TTT initialization audit evidence: `experiments/baseline_reproduction_qwen3_4b/remote/hku_gpu2_qwen3_4b_native_initialization_audit_20261019/`
- Native writer autograd audit evidence: `experiments/baseline_reproduction_qwen3_4b/remote/hku_gpu2_qwen3_4b_native_writer_autograd_audit_20261019/`
- Liger/fused-AdamW GPU preflight evidence: `experiments/baseline_reproduction_qwen3_4b/remote/hku_gpu2_qwen3_4b_liger_adamw_gpu_preflight_20261019/`
- Full-checkpoint writer preflight and diagnosis: `experiments/baseline_reproduction_qwen3_4b/remote/hku_gpu2_qwen3_4b_full_checkpoint_writer_preflight_20261019/`
- FSDP2/meta writer initialization evidence: `experiments/baseline_reproduction_qwen3_4b/remote/hku_gpu2_qwen3_4b_fsdp2_initialization_audit_20261019/`
- FSDP2 8K writer autograd evidence: `experiments/baseline_reproduction_qwen3_4b/remote/hku_gpu2_qwen3_4b_fsdp2_autograd_audit_20261019/`
- Baseline tooling regression evidence: `experiments/baseline_reproduction_qwen3_4b/local/baseline_tooling_regression_20261019/`
- Corrected RULER 256K resource watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_ruler_256k_resource_1eb9dc07_20261019.watch`
- Corrected RULER 256K resource tmux session: `baseline_qwen3_4b_base_ruler_256k_resource_1eb9dc07_20261019`
- Corrected RULER 256K release-review watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_ruler_256k_release_review_55fc42a0_20261019`
- Corrected RULER 256K release-review tmux session: `baseline_qwen3_4b_base_ruler_256k_release_review_55fc42a0_20261019`
- Future corrected RULER 256K release: `/home/ctj/in_place_ttt_qwen3_4b_base_ruler_256k_release_55fc42a0_20261019.json`
- Corrected formal RULER 256K watcher: `/home/ctj/in_place_ttt_qwen3_4b_base_ruler_256k_formal_fd252a62_20261019.watch`
- Corrected formal RULER 256K tmux session: `baseline_qwen3_4b_base_ruler_256k_formal_fd252a62_20261019`
- Corrected Table 1 report root: `/home/ctj/in_place_ttt_qwen3_4b_base_table1_report_0299fb76_20261019`
- Corrected Table 1 report tmux session: `baseline_qwen3_4b_base_table1_report_0299fb76_20261019`
- Corrected scale-feasibility report root: `/home/ctj/in_place_ttt_qwen3_4b_base_scale_feasibility_4e60608f_20261019`
- Corrected scale-feasibility tmux session: `baseline_qwen3_4b_base_scale_feasibility_4e60608f_20261019`

The prior checkpoint-bound corpus, activation, Stage-1, Stage-2 and RULER
paths are preserved but superseded. No path in that chain is an authoritative
live input for the corrected Base reproduction.

All GPU execution is restricted to `hku-gpu2`. The paths above are evidence
locations, not permission to bypass any predecessor, release or audit gate.
