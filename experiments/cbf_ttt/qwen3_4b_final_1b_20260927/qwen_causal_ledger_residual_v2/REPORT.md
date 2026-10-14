# Scale-32 causal-ledger content validation V2

## Outcome

The fresh V2 validation passed all 12 preregistered checks and all six
independent audit checks. It releases the development corpus at frozen scale
32. Confirmation and OOD remained ungenerated.

The primary addressable top-1 ledger reached 100% six-candidate accuracy and
100% five-token exact generation on all 1,008 queries. It exactly agreed with
the semantic lookup oracle. Every lifecycle family, current/historical/anchor
role and reset-induced absent/null query reached 100% exact.

## Fresh-data protocol

- Seed `202610163`.
- New `resource_v2` surface template.
- Disjoint `Y ...` five-token value namespace; V1 used `Z ...`.
- Seven lifecycle families, eight groups per family and two twins per group:
  56 groups and 112 episodes.
- Three boundaries and three queries per boundary: 1,008 queries.
- Frozen scale 32 selected by the preceding independently audited fixed grid.
- Exactly five decoding steps for every primary, control and twin trace.

## Main results

| Arm | Candidate accuracy | Truth exact | Answer NLL |
|---|---:|---:|---:|
| Single correct | 100.00% | 100.00% | 0.01014 |
| Single wrong | 0.00% | 0.00% | 15.94916 |
| Zero | 18.06% | 0.00% | 6.30756 |
| Random | 19.44% | 0.00% | 6.53752 |
| Dense visible sum | 60.42% | 3.37% | 14.77229 |
| Addressable top-1 | 100.00% | 100.00% | 0.01014 |
| Exact lookup oracle | 100.00% | 100.00% | 0.01014 |

The wrong residual generated the injected wrong value in 100% of queries.
Among the 234 rows in 39 groups where paired twins retained different causal
answers, the twin residual generated the twin value in 100% and the factual
answer in 0%. Reset removes the old twin difference by construction and is
recorded as causal equality rather than mislabeled as a wrong twin.

Group-bootstrap 95% confidence intervals for correct-answer NLL improvement
over Wrong, Zero and Random were respectively `[15.7895,16.0882]`,
`[6.2695,6.3262]` and `[6.4888,6.5632]`.

## Mechanistic conclusion

Scale 32 fixes the contextual full-vocabulary margin problem found in V1, but
scale alone does not make a multi-record memory readable. Summing every
visible Qwen-key similarity produced severe cross-talk: only 60.42% candidate
accuracy and 3.37% free exact. Record-addressable top-1 routing removed that
interference and matched the exact state oracle on every row.

This supports a controlled architectural conclusion: the useful fast memory is
an addressable causal ledger with explicit reset/revoke/shadow semantics and a
sparse record read, rather than a dense sum of all session values. The current
test supplies exact structured keys and an explicit absence check. It does not
yet prove natural-language key discovery or native dense TTT writer binding.

## Execution and audit

- Two RTX 5090 critical path: 322.1695 seconds.
- Peak allocated/reserved CUDA memory: 7.6678/7.6934 GiB.
- Final Qwen3-4B 1B-token checkpoint loaded without missing, unexpected or
  mismatched keys.
- Independent audit recomputed source alignment, state counts, candidates,
  every five-token control/twin trace, bootstrap intervals, all gate checks,
  resources, hashes and split sealing.

An initial formal attempt was stopped after 199 partial rows because code
review found that the generic decoder could still terminate before five steps
on EOS/newline. That attempt is excluded. The fixed version records
`fixed_length=true`, was re-smoked on a different episode, and generated all
five tokens for every control before this formal retry.

## Next stage

`QWEN_CAUSAL_LEDGER_DEVELOPMENT_V2_PLAN.md` freezes 64 groups per family on the
reserved development seed. It compares the full ledger with no memory,
retain-latest, global boundary clear, last-1, last-2 and dense-full policies,
plus a semantic oracle. A one-time matched confirmation is released only after
a positive independently audited development gain over the strongest
non-oracle comparator.

