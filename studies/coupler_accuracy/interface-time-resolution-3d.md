# Exchange time resolution in the fully 3D cube

The preceding [precision-corrected interface experiment](panel-derivative-precision-3d.md)
converges all 20 intervals while retaining a final drag error of `+0.75578%`
and whole-small-FVM velocity error of `0.00766375 U∞`. This experiment asks how
much of that remaining error changes when FVM and VPM exchange data more often.

The completed comparison finds a tradeoff. At the same 20 physical sample
times, the smaller interval reduces the iterated drag-history RMS error from
`0.55507%` to `0.48815%`, but increases final whole-small-FVM velocity error by
`13.49%` and final near-body error by `5.62%`. All 100 smaller intervals
converge. A shorter exchange interval is therefore not a general accuracy
improvement; the larger interval remains the selected case for the next
benchmark comparison.

The comparison retains the unit cube, its small approximately `[-1.5,1.5]^3`
FVM domain, all 16,936 inherited cells, matched medium `h=0.0625`, the same
initial physical state at `t=0.5`, and the same independent 53,752-cell full
FVM reference. The FVM time step stays `0.01`. Three-dimensional induction,
stretching and body-complete target queries are retained. Auxiliary body queries use the
qualified double-precision evaluation; the particle fields remain single
precision. Production defaults are unchanged.

| Input | Completed larger-interval pair | Completed smaller-interval pair |
| --- | ---: | ---: |
| FVM time step | 0.01 | 0.01 |
| FVM substeps per exchange | 5 | 1 |
| Exchange and VPM step | 0.05 | 0.01 |
| Coupling intervals to physical time 1.5 | 20 | 100 |
| Accepted FVM steps | 100 | 100 |
| Maximum interface sweeps | 0 / 12 | 0 / 12 |
| Normal and derivative convergence thresholds | 1e-6 | 1e-6 |

Changing the exchange interval also changes the VPM RK2 step, GBD diffusion
and remapping schedule, and particle renewal frequency. This is a combined
hybrid time-resolution comparison; it does not separate those effects or
refine the FVM discretization.

The completed [inviscid time-isolation follow-up](inviscid-time-isolation-3d.md)
holds diffusion, exchange and renewal at `0.05` while reducing only RK2 to
`0.01`. All 20 intervals converge, but drag-history RMS error changes only
from 0.555068% to 0.555463%, and final whole-FVM velocity error increases
0.0313%. Thus the larger combined-cadence response is not reproduced by
inviscid refinement alone. Diffusion/remapping, splitting and exchange/renewal
still require separate comparisons. The selected panel mode also excludes
the body contribution from particle stages while retaining it in target
queries; the [body audit](body-query-and-transport-3d.md) measures that distinction.

## Isolated runner and exact control

The [new runner](cube_interface_cadence_3d.py) invokes the existing precision
and interface-iteration studies, overriding only the trial's existing
`substeps` input. It restores the original entry point on exit and records the
actual child time steps, invocation count, source files and artifact hashes.
All runs use the same
[frozen source workspace](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/frozen-workspace.json).
New study files were added under new names; none of the 799 original source
and input files was changed.

At exchange step `0.01`, the zero-sweep control and one-sweep replay produce
bitwise identical comparison histories and complete numeric checkpoints:
17 FVM entries, 11 boundary-history entries and 11 numeric VPM datasets. The
[completed qualification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/cadence-one-substep-replay-verification.json)
checks 109 source/artifact records, every original frozen file for each run,
and ten independently recomputed metrics with zero difference. This verifies
the wrapper and replay at the new interval; it does not establish accuracy
improvement or a separate 100-interval zero/one-sweep trajectory identity.

Both 100-interval trajectories finish at physical time `1.5`. The iterated
run converges all 100 intervals: 89 need two sweeps and eleven need three,
for 211 logical sweeps plus 100 first-map replays. The largest accepted normal
and derivative residuals are `5.52e-7 U∞` and `9.89e-7 U∞/D`, below the
unchanged `1e-6` thresholds. The larger interval uses 60 logical sweeps plus
20 replays. These are operation counts; wall-time ratios are not treated as
a controlled performance comparison.

## Completed accuracy comparison

The four cases use exactly the same initial comparison record, source files
and fixed physical/numerical inputs. Final reference velocities match bitwise,
as do reference forces at all common FVM time indices. Statistics below use
those 20 common endpoints and exclude the initial state.

| Metric | Original, exchange 0.05 | Iterated, exchange 0.05 | Original, exchange 0.01 | Iterated, exchange 0.01 |
| --- | ---: | ---: | ---: | ---: |
| Common-time drag RMS relative error | 2.08231% | 0.55507% | 1.59239% | 0.48815% |
| Final signed drag error | −2.09446% | +0.75578% | +0.73337% | +0.66242% |
| Final whole-small-FVM velocity RMS / U∞ | 0.00768630 | 0.00766375 | 0.00874636 | 0.00869738 |
| Final near-body FVM velocity RMS / U∞ | 0.00200498 | 0.00195493 | 0.00211691 | 0.00206487 |
| Final sampled VPM velocity RMS / U∞ | 0.03124529 | 0.03124260 | 0.03144909 | 0.03143859 |

Over the common times, reducing the interval increases the iterated whole-FVM
velocity RMS statistic by `4.04%`, decreases the near-body statistic by `2.34%`,
and increases sampled VPM error by `0.21%`. The final near-body change has the
opposite sign, so a uniformly better velocity trajectory is not established.

The dense smaller-interval history also resolves a first-step relative drag
error of `11.009%` at physical time `0.51`: the full FVM drag is `2.2491673`
and the iterated hybrid drag is `2.4967818`. The `0.05` coupling-endpoint
history does not sample this time. Across all 100 fine endpoints, its drag
RMS relative error is `1.20507%`, rather than the common-time `0.48815%`.
Both statistics are retained, and the complete fine history is plotted below.
The reference itself has a startup force excursion; no cause for that
transient is inferred from this time-resolution comparison alone.

![Verified exchange interval comparison](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/interface-cadence-comparison-3d.png)

The figure was visually inspected. The
[100-interval verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/cadence-one-substep-hundred-verification.json)
checks 426 source/artifact records, all 799 original frozen files per run,
and 440 independent metrics with zero difference. The
[cross-interval comparison](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/interface-cadence-comparison-3d.json)
retains the common time indices, fixed-input checks and complete metric table.

## Comparison requirements

The [cadence verifier](verify_interface_cadence_3d.py) derives the expected
FVM step count from recorded, independently checked time steps. It retains
the exact replay checks, first-map fingerprints in every interval, convergence
flags, archive hashes and independent endpoint wall-force/velocity metrics.

The [comparison script](compare_interface_cadence_3d.py) then checks that both
pairs have identical solver source records, all fixed physical/numerical
inputs, initial comparison records and final reference fields. Reference
forces must match bitwise at the common accepted FVM time indices. Statistics
use those identical 20 endpoints and exclude the initial state; no phase shift,
interpolation of forces, or force fitting is permitted. The denser fine-step
history remains visible in the figure. The sampled VPM velocity metric is at
matched FVM cell centres, not an exterior-wake profile measurement.

The completed commands below run from the frozen workspace. Choose new output
names when repeating the analyses; existing results are not overwritten.

```sh
env PYTHONPATH=. TI_CPU_MAX_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python studies/coupler_accuracy/verify_interface_cadence_3d.py \
  --control studies/coupler_accuracy/results/cadence-one-substep-hundred-control \
  --replay-control studies/coupler_accuracy/results/cadence-one-substep-one-control \
  --replay studies/coupler_accuracy/results/cadence-one-substep-one-replay \
  --candidate studies/coupler_accuracy/results/cadence-one-substep-hundred-twelve \
  --output studies/coupler_accuracy/results/cadence-one-substep-hundred-verification.json
env PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python studies/coupler_accuracy/compare_interface_cadence_3d.py \
  --coarse studies/coupler_accuracy/results/precision-twenty-step-verification.json \
  --fine studies/coupler_accuracy/results/cadence-one-substep-hundred-verification.json \
  --output studies/coupler_accuracy/results/interface-cadence-comparison-3d.json
```

The original requirements remain: developed three-dimensional flow and forces,
matching mesh resolution, a small FVM domain and close agreement with the
fully meshed solution. Completing this time-resolution pair alone will not
establish those requirements.

The later [RK/GBD isolation](split-time-isolation-3d.md) completes the four
time-schedule contrasts. Refining RK alone barely changes the result; refining
RK and GBD together at fixed exchange also fails to improve overall accuracy.
The remaining complete-exchange contrast still changes renewal, boundary
updates and outer stabilization frequency together. These results narrow the
diagnosis without establishing that a smaller coupled time step is a general fix.
