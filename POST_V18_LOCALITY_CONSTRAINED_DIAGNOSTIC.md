# V18-LC Post-hoc Locality-Constrained Diagnostic

The frozen V18 formal gate failed only its absolute locality requirement:
retain-all itself reached 93.4375% locality and the unconstrained oracle reached
91.875%, while all seven headroom, update, retention, interval and fold checks
passed. The original result remains failed and does not release Stage B.

This explicitly post-hoc diagnostic reuses the already scored arms and performs
no model calls. It asks whether selective forgetting has causal headroom under
a dataset-level safety budget. For 320 episodes, the primary budget permits at
most nine locality failures, equivalent to at least 97% locality. Pareto points
at 100%, 99%, 95% and 90% are reported without selecting among them.

For each episode, feasible arms are all frozen fixed, single-deletion and
key-deletion states. Dynamic programming maximizes total joint success subject
to the failure budget, then update accuracy, retained-key accuracy, locality,
free-generation joint success and negative joint NLL in that fixed order.

The primary diagnostic checks the original gain, clustered interval, update,
retention, locality and every-fold conditions. Passing is exploratory evidence
for a safety-constrained objective and requires a new, independently frozen V19
confirmation corpus. It cannot retroactively pass V18 or release controller
training on V18.
