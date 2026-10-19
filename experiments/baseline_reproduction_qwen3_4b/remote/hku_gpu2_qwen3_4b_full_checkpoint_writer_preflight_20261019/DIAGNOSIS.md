# Failure diagnosis

## Frozen outcome

The preregistered V1 result remains failed. `audit.json` is the authoritative
outcome: 22/24 checks passed and two strict numerical checks failed. The
follow-up probe was executed only to identify the cause and cannot change the
outcome.

## Projection diagonal

The writer projection is structurally diagonal in every selected layer:
`projection_nonzero == projection_diagonal_nonzero`, so every off-diagonal
entry is exactly zero. The six diagonals contain respectively 10, 17, 20, 12,
7 and 17 exact zeros out of 2560 entries.

The fixed-seed mechanism probe directly sampled the initialization operation
`randn(2560, dtype=bfloat16) * 0.02`. On CPU it produced 4--24 exact zeros per
vector, with mean 12.8125 over 32 seeds. On the RTX 5090 it produced none for
the same seeds. This is consistent with CPU BF16 random-number discretization
in the `from_pretrained` missing-weight initialization path. It is inconsistent
with an off-diagonal initialization error.

## Initial output difference

The six convolutions are exactly zero, so no learned fast-weight delta is
present. However, the active native path computes the base down projection by
a chunked `einsum`, while the bypass path uses `F.linear`. A fixed-seed BF16
probe at the real `[64, 9728] x [2560, 9728]` dimensions measured maximum,
mean and RMS absolute differences of `0.0625`, `0.00261624` and `0.00565958`
for this operation alone. It failed the same `1e-3` allclose threshold.

Propagation through six selected layers is therefore a concrete numerical
explanation for the observed maximum final-logit difference of `0.25`. This is
an inference from the isolated kernel probe, not a bitwise decomposition of
the full-model difference.

## Supported conclusion

The exact Base checkpoint, six writer modules and Liger fused-loss gradient
path integrate successfully, but the stronger claim that full-checkpoint BF16
initialization is numerically identical to a no-TTT branch is false under this
execution path. The baseline must preserve and report this released-code
behavior. This diagnostic supplies no efficacy evidence and does not release
formal training or gating.
