# Verified simulation improvement and recorded computational cost

The controlled fully 3D cube experiment shows a substantial short-run drag
improvement and smaller velocity improvements. A computational speedup and
the requested developed-wake agreement have not been demonstrated.

All three cases below use identical near-body cells, nominal spacing
`h=0.0625`, and a small FVM box spanning three body widths. The comparison
covers physical time `0.5–1.5`, with twenty exchanges. The full-FVM reference
advances independently, with identical reference drag observations in all
three cases. This spacing differs from the tutorial's original medium mesh.
The baseline is the qualified uniterated experiment, not an archived pristine
version of the project at the start of the investigation.

| Measurement | Uniterated baseline | Iterated interface | Iteration plus body transport |
| --- | ---: | ---: | ---: |
| Relative drag-history RMS error (%) | 2.082312 | 0.555068 | 0.521061 |
| Final near-body velocity RMS error (% U∞) | 0.200498 | 0.195493 | 0.183527 |
| Final whole-small-FVM velocity RMS error (% U∞) | 0.768630 | 0.766375 | 0.746246 |
| Recorded comparison-run wall time (minutes) | 8.10 | 14.31 | 17.26 |

The combined experimental changes reduce drag-history RMS error by `74.98%`,
final near-body velocity error by `8.46%`, and final whole-FVM velocity error
by `2.91%` relative to the uniterated baseline. The curves retain the startup
discrepancy and times when velocity error increases.

![Verified accuracy comparison](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/verified-progress-evidence-short-cube/verified-progress.png)

The figure was visually inspected. The new [evidence generator](show_verified_progress_3d.py)
rechecks the two existing independent qualifications and joins their histories
only after checking identical initial observations and full-reference drag.
The [evidence record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/verified-progress-evidence-short-cube/verified-progress-evidence-3d.json)
retains the source records, computed reductions and exact wall times.

The times are **end-to-end comparison-run measurements**, including both the
reference and hybrid simulations, setup, checks and output. Observer workloads
and background load differ. They are not controlled solver-only timings and
do not establish a hybrid-versus-full-FVM speedup. Interface iteration adds
work; no performance benefit is claimed from these results.

The latest accuracy changes remain experimental. Their body-transport benefit
has also been [verified through physical time 2.5](body-query-and-transport-3d.md),
but the developed-wake comparison and the requested tutorial-level acceptance
remain unfinished.
