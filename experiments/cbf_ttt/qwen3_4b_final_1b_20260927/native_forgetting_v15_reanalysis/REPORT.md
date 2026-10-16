# V15 native fast-weight forgetting-only reanalysis

## Conclusion

The corrected forgetting-only analysis does not support training a controller
on the current native writer. It uses the actual dense fast weights produced by
the final Qwen3-4B checkpoint's `ttt_proj/ttt_conv`, fixes the current write at
`g=1`, and varies only the method's forgetting fraction in `{0,0.5,1}`.

| Fixed forgetting fraction | Mean continuation NLL |
|---:|---:|
| 0.0 (retain all old memory) | 2.46416 |
| 0.5 | 2.44144 |
| 1.0 (clear all old memory) | **2.40752** |
| Three-point rowwise oracle | 2.40642 |

The oracle improves over the strongest fixed action by only **0.00110 NLL**;
its source-group bootstrap 95% interval is `[0.00031, 0.00201]`, below the
preregistered 0.005 material-effect threshold. It selects full forgetting in
39/48 rows (81.25%), no forgetting in eight, and 0.5 in one. Thus the apparent
state dependence is too small to justify controller training.

The writer diagnostic is also negative. After optimizing retention separately,
the best `g=1` branch is worse than the best `g=0` branch by **0.04045 NLL** on
average. Expressed as write gain (`best g=0 NLL - best g=1 NLL`), the
source-group bootstrap 95% interval is `[-0.05825, -0.02460]`. Only 7/48 rows
have positive write gain.

## Mechanism breakdown

Full forgetting has the best mean NLL in every mechanism family. In the
`old_only` condition it is selected in all 12 rows; retaining all old memory
raises NLL from 2.63160 to 2.77054. This is the opposite of the behavior needed
to show useful historical session memory. The result is consistent with the
earlier native accumulation and Correct/Wrong/Zero interventions: the current
writer mostly injects harmful perturbations rather than a reliably readable
memory.

## Evidence boundary

This is a model-free reanalysis of 48 previously observed natural-text traces,
not a new held-out evaluation. It is nevertheless a valid substrate correction
because every compared branch was generated through `CBFSession` from the same
pending native dense update, same KV state, same future content, and frozen
backbone. Runtime retention is converted to the paper's forgetting fraction by
`alpha_forget=1-alpha_retain`.

The positive V6--V13 utility-record results remain useful proxy evidence for
selective deletion in an addressable memory. They cannot be cited as evidence
for native dense fast-weight forgetting.

## Decision

The V15 gate fails. Do not train the native forgetting controller yet. The next
experiment must repair and validate `ttt_proj/ttt_conv` content utility on
source-disjoint natural continuation data. Controller labeling resumes only
after Correct fast weights beat Wrong, Random, and Zero controls and old-memory
retention helps in at least one predefined condition.

## Reproduction

```bash
python -m scripts.analyze_native_forgetting_slice_v15 \
  --inputs experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_natural_pilot/*_joint_labels.jsonl \
  --output /tmp/native_forgetting_v15_summary.json
```

The tracked aggregate is `summary.json`; raw labels retain their existing local
data policy.
