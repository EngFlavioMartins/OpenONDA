# Cylinder reference flow

This body-fitted calculation resolves an extruded circular cylinder at
Reynolds number 150. The diameter is D = 1 m, freestream speed is U = 1 m/s,
and kinematic viscosity is UD/Re = 1/150 m²/s. Slip span boundaries represent
a span-invariant flow without physical endcaps. Forces use D times span as
the reference area.

## Mesh and time resolution

The domain is x/D = [-8, 24], y/D = [-10, 10]. Four uniform layers span one
diameter. The spatial family refines only the XY mesh with ratio
r = sqrt(2); separate cases test span resolution, span width and time step.

| Case | Requested spacing/D | Nominal body lattice/D | Span layers | Span/D | Maximum time step [s] |
|---|---:|---:|---:|---:|---:|
| xy_coarse | 0.080000000000 | 0.060000000000 | 4 | 1 | 0.004 |
| xy_medium | 0.056568542495 | 0.042426406871 | 4 | 1 | 0.004 |
| xy_fine | 0.040000000000 | 0.030000000000 | 4 | 1 | 0.004 |
| xy_finer | 0.028284271247 | 0.021213203436 | 4 | 1 | 0.004 |
| z_eight | 0.040000000000 | 0.030000000000 | 8 | 1 | 0.004 |
| span_two | 0.040000000000 | 0.030000000000 | 8 | 2 | 0.004 |
| dt_half | 0.040000000000 | 0.030000000000 | 4 | 1 | 0.002 |

The mesher subdivides a background spacing of 12 times the requested spacing
by powers of two. The nominal near-body lattice is therefore 0.75 times the
requested spacing; surface projection changes individual edge lengths. Use
the saved realized spacing for refinement ratios. A three-dimensional
cell-count estimate of spacing is inappropriate for this XY refinement study.
The span and time controls are excluded from Richardson triplets.

The FVM uses limited-linear convection, least-squares gradients, implicit Euler,
two PIMPLE outer correctors and two pressure corrections. The adaptive time step
limits the maximum Courant number to 0.9 and respects the listed time-step cap.
Every case in the new campaign runs to 100 s. Force samples are spaced by
0.02 s and line samples by 0.1 s. Only `xy_fine` writes volume visualization snapshots and restart
checkpoints every 0.25 s. Its explicit `--output-interval 0.25` enables actual
volume output even with `--lean`; `--backup-interval 0.25` sets the independent
restart cadence. The other six cases retain lean output: 10 s checkpoints and
no volume or surface visualization series. All force/profile cadences and
physical solver settings are unchanged.

## Run and continue

Use an installed OpenONDA environment. From this directory:

```sh
./allrun.sh
```

The script lists the seven cases explicitly and writes below
`campaigns/geometric_xy_100s`. The already-running `geometric_xy_v1` campaign
retains its original 160 s commands and output cadence; its existing 80–160 s
comparison stage is separate and unchanged. Historical `samples/` and
`solution/` records are also separate. Fresh runs reject nonempty output;
continue an individual case from its own checkpoint with the same physical
and numerical options:

```sh
python setup.py --name xy_fine --dx 0.04 --output-root campaigns/geometric_xy_100s/spatial --cores 2 --end-time 100 --span 1 --span-layers 4 --maximum-time-step 0.004 --output-interval 0.25 --backup-interval 0.25 --lean --restart-from campaigns/geometric_xy_100s/spatial/solution/xy_fine/backup
```

`allcontinue.sh` continues the historical `dense` case to 100 s. `allclean.sh`
removes only the tutorial's `solution/` and `samples/` outputs; it does not
remove the separate campaign directory.

## Assess convergence

After the required cases complete, run the analysis explicitly:

```sh
python postprocess_grid_study.py --samples-root campaigns/geometric_xy_100s/spatial/samples --solution-root campaigns/geometric_xy_100s/spatial/solution --output-dir campaigns/geometric_xy_100s/figures --statistics-start 50 --statistics-end 100 --format png
python postprocess_campaign.py campaigns/geometric_xy_100s --start 50 --end 100
```

Use `--format pdf` for PDF figures. Derived tables and reports are placed under
`figures/auxiliary/`. The analysis uses native completion records and a common
50–100 s statistics window. Inspect mean drag, force RMS, Strouhal number,
mean velocity profiles and averaging drift. A 100 s end time does not guarantee
stationarity or enough shedding cycles: extend the run and averaging interval
if the existing qualification checks fail. The coupled comparison still requires
at least eight complete qualified cycles; neither that requirement nor any
convergence tolerance is relaxed for this shorter campaign. Geometric refinement
alone does not establish grid independence.

Following [NASA's spatial convergence guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html)
and [Celik et al.](https://doi.org/10.1115/1.2960953), use at least three
systematically refined meshes and an observed-order Richardson estimate with
GCI safety factor 1.25. The chosen ratio exceeds 1.3. Compare both overlapping
triplets; their fitted asymptotic ratios are not independent convergence proof.
Nonmonotone changes, orders above four, or drag differences below the four-batch
sampling interval prevent a formal GCI estimate. The batch interval is only a
stationarity diagnostic until each batch contains enough shedding cycles.

Assessment targets are 1% for mean drag and 2% for force RMS, mean-profile L2
and Strouhal number, with span/time errors smaller than the spatial changes.
The FFT-bin Strouhal estimate has finite-window resolution and cannot establish
agreement finer than that resolution. If the finest family is not converging,
the next requested spacing is 0.020 m; a longer averaging window may also be
needed. These are criteria to test, not established results.

The [historical reference assessment](../../../../studies/panel_removal/CYLINDER_REFERENCE_ASSESSMENT.md)
records measurements from the earlier unequal-ratio family. The
[planar coupling study](../../../../studies/panel_removal/CYLINDER_EXPERIMENT.md)
defines the matched coupled-flow comparison.
