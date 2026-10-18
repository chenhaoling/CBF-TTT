# Corrected Qwen3-4B-Base paper-facing RULER plan

Status: **frozen before any corrected-Base formal RULER score**

Freeze date: 2026-10-19 (Asia/Shanghai)

## Scope and checkpoint release

Evaluate only the two final HF checkpoints released by both the underlying
formal Stage-2 audit and the corrected Qwen3-4B-Base wrapper audit. Resource
or timing checkpoints and checkpoints from another identity chain are
forbidden. All GPU execution is restricted to `hku-gpu2`: physical GPU 0
runs no-TTT and physical GPU 1 runs native TTT within each paired context.

Keep the original Qwen3-4B tokenizer, deterministic decoding, seed 42,
`enable_thinking=false`, 64 maximum new tokens and
`fix_mistral_regex=false`. Paper-facing inference must use the separately
frozen opt-in overlay that clips every native fast-weight update delta to
Frobenius norm `1e-5`; the clean upstream checkout and trained checkpoints
remain unchanged.

## Resource ladder

Before opening formal scores, run paired 4K, 32K, 64K and 128K boundaries.
Each arm evaluates all 13 registered RULER tasks with exactly one example per
task. Discard every resource score. Require matching prompts/references,
passing single-run audits and at least 5% GPU memory headroom for both arms at
every boundary. Stop on any OOM, dependency/model drift or failed audit.

## Formal matrix

After the corrected resource wrapper passes, run paired 4K, 8K, 16K, 32K,
64K and 128K contexts sequentially. At each context, each arm evaluates all
13 registered RULER tasks with exactly 100 examples. Preserve raw prompts,
references, predictions, task results, summaries, logs, model/dependency
hashes and GPU traces. Independently recompute every task/context macro,
`TTT - no-TTT` delta and the six-context mean from raw results.

The registered paper-facing matrix ends at 128K. A 256K YaRN extrapolation
requires its separate resource gate and release and may not alter the main
matrix if it fails.

## Claim boundary

Only a complete paired matrix with both underlying and corrected audits may
support the public-data baseline claim. It must be reported as a 100M-tier
matched reproduction, alongside the paper's reference numbers; it is not an
exact reproduction of the paper's private-data full-scale training result.
Update-admission experiments remain sealed until this corrected matrix is
complete and independently audited.
