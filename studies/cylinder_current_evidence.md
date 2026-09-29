# Cylinder evidence snapshot — 2026-09-29

The measured evidence supports provisional mesh/rank choices and short transient
sensitivity results. It does **not** yet select a grid-independent production
mesh or establish completion within twelve hours. This read-only assessment
uses the [execution report](cylinder_execution_report.md),
[original plan](cylinder_3d_accuracy_performance_plan.md), saved pilot JSONs and
the existing `phase-20260929` logs. No simulation was started or modified.

## Measured meshes and parallel choices

The matched-span reference pilots used span 0.96D. At h=0.08D, the native mesh
contained 42,624 cells. Warm late median step times on 2/4/6/8 MPI ranks were
0.440/0.312/0.244/0.481 s; aggregate peak RSS was
1.19/2.14/3.02/3.93 GiB. These trials reached only t=1D/U. Six ranks was best
among those measured reference trials. A six-rank h=0.0565685D pilot contained
114,104 cells, reached t=0.16D/U and measured 0.708 s median steps with 3.83 GiB
RSS. Its mesh reused a historical XY section; it did not measure fresh source
meshing. [Raw reference pilots](cylinder_machine_pilots.json).

The coupled hxy=dz=0.064D, hp=0.08D, span=0.96D mesh contained 36,555 cells
and 114,052 faces. Four ranks measured 16.312/17.031 s warm intervals versus
16.384/17.366 s on six ranks; whole invocations took 225.39/248.68 s and peak
RSS was 5.54/6.45 GiB. Four ranks therefore remains the measured coupled
startup choice. These three-interval trials preceded later numerical fixes.
The parity-corrected four-rank trial measured 16.418/17.078 s warm intervals,
but also preceded the sparse wall-image correction.
[Initial finest pilot](cylinder_finest_cpu_pilot.json),
[parity-corrected pilot](cylinder_finest_cpu_parity_fixed_pilot.json).

For the corrected hxy=hp=dz=0.08D transient, default optimizations reduced
mean warm interval time from 12.206 to 10.416 s and whole invocation from
249.11 to 222.73 s. Both ended at t=0.4D/U with 4,228 particles and only 2/10
interface-converged intervals. These are single matched trials, not developed
wake forecasts. Experimental Aitken converged 10/10 intervals but measured
10.451 s warm intervals, so it demonstrated convergence improvement rather
than speedup. [Raw optimization comparison](cylinder_optimization_comparison_2026-09-22.json).

## Completed sensitivity evidence

Five corrected four-rank cases reached the common transient t=0.4D/U:

| Variant | Whole invocation [s] | Final particles | Endpoint Cd | Converged intervals |
| --- | ---: | ---: | ---: | ---: |
| Exchange dt=0.04, hp=0.08D | 300.1 | 4,228 | 1.5382 | 2/10 |
| Exchange dt=0.02 | 418.1 | 4,887 | 1.7642 | 17/20 |
| Exchange dt=0.08 | 239.7 | 3,660 | 1.4584 | 0/5 |
| Particle-spacing ratio 1.25, realized hp=0.096D | 264.5 | 2,745 | 1.5366 | 4/10 |
| Particle-spacing ratio 1.50, realized hp=0.120D | 242.0 | 1,592 | 1.5860 | 5/10 |

Peak RSS was 5.34–5.36 GiB. The larger-spacing cases also enlarged physical
core, blend and release widths, so they do not isolate particle spacing.
Exchange timestep changes do not independently test skipped renewal cadence.
No stationary drag, lift RMS, Strouhal or uncertainty qualification follows
from these endpoints. [Raw corrected cohort](cylinder_final_short_cohort.json).

## Existing phase benchmark snapshot

The frozen launch selects h=0.04D, span=0.96D, 24 span layers, FVM dt=0.008 s,
exchange dt=0.04 s and horizon 100D/U. It explicitly labels this mesh a
provisional practical choice. The reference domain is
[-8,24]D × [-10,10]D × [-0.48,0.48]D. Its console reports **302,832 global
solver cells on six MPI ranks**; earlier meshing-stage counts in the same log
are not the final solver count.

At the inspected log position, step 1,690 had reached t=13.52D/U with solver
elapsed time 02:21:27.8. The last ten reported steps ranged from 6.64 to
13.40 s. At dt=0.008, a 100D/U trajectory has 12,500 steps and a gross
twelve-hour allowance of **3.456 s per step before startup and output costs**.
The observed late steps exceed that allowance; this snapshot cannot certify
the target. Host load and future cost changes prevent treating it as a final
runtime estimate. No coupled-child start or completed pair appears in the
inspected event journal. This is a moving log snapshot read at approximately
2026-09-29 14:03 UTC, not a completion claim.

Local provenance: `tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/`
`study_results/phase-20260929/{launch.json,events.jsonl,logs/reference/console.log}`.
This independent reference experiment was left untouched; its frozen source
identity must be compared with any later solver changes before reusing results.

## Concrete remaining original requirements

Finish matched reference/coupled histories and measure full wall cost including
meshing, cold initialization, sampling and backups. Obtain developed-wake
timing and memory, independent hxy/dz/span/domain/timestep qualification,
common stationary windows with at least ten shedding periods, and uncertainty
for drag, lift RMS, Strouhal and profiles. Complete controlled renewal cadence,
support/buffer/core and spacing sensitivity with conservation and ownership
budgets, then select and reject configurations from an accuracy–time–memory
comparison. The twelve-hour limit is per complete finest case; it does not
bound the whole sensitivity campaign. Neither a wall timeout nor the current
provisional h=0.04D choice satisfies these scientific requirements.
