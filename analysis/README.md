# EDA parameter-distribution study — *bypass* influence diagram

Interactive, self-contained visualization of how the parameters of the *bypass*
influence diagram (the CPTs and the utility table) are recovered by Estimation-of-
Distribution Algorithms (EDAs), shown as the **distribution of each parameter over
the final population** of the search.

Open **[edas_bypass_distributions.html](edas_bypass_distributions.html)** in any
browser — all data is embedded, so it works offline.

## What it shows

- The influence diagram (chance = ellipse, decision = rectangle, utility = diamond),
  each optimizable node coloured by the mean uncertainty of its parameters
  (light = recovered, dark = degenerate).
- Click a node → a histogram of every CPT / utility cell over the 400-individual
  final population, with the **true value** overlaid (green line) and μ / σ.
- For the utility node, a **ranking (low → high)** comparing the true order with the
  order obtained from the **mode** of each cell's population — the meaningful test,
  since utilities are only identifiable up to a positive affine transformation.
- A **Mean** view that pools the 5 repetitions (so σ also captures the spread
  between initializations).

## Grid

Every configuration runs the same 5 repetitions (identical rule set, different
random initialization). Selectors let you pick any combination:

| Axis | Values |
|------|--------|
| Optimizer | EMNA · UMDA |
| Fitness | binary · regret · margin · softmax · entropy |
| Rules elicited | 30% · 70% (of the 20 available) |
| Temperature (utility, chance) | (1, 1) · (4, 1) · (8, 1)|

Fixed for all runs: population = 400, stopping criterion = **top90**, min/max
utilities anchored to 0 / 10 (`min_max_ut = True`), evaluated on the
`example/bypass2` network.
