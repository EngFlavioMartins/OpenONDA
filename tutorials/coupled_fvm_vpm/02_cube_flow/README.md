# Coupled flow around a cube

A body-fitted FVM region resolves the viscous wall of a unit cube at Re=1000.
The surrounding VPM transports vorticity, using RK2, grid diffusion, Gaussian
particles and FMM induction. Both solvers use equilibrium Smagorinsky closure.
Freestream speed is 1 m/s and kinematic viscosity is 0.001 m²/s.

| Quantity | Current setup |
| --- | --- |
| Cube | [−.5,.5]³ m |
| FVM domain | [−1.5,1.5]³ m |
| Transfer region | [−1.45,1.45]³ m |
| FVM near-body spacing | .045 m |
| VPM particle spacing | .045 m |
| FVM step | .01 s |
| VPM/coupling step | .05 s |
| End time | 30 s |
| Force/profile sampling | .05 s |
| Retained fields and coupled checkpoint | .25 s |
| Maximum interface sweeps | 3 |

The mixed vorticity boundary and buffered M4 renewal iterate from the same FVM
start and VPM predictor. Convergence status is recorded in
`solution/coupler_diagnostics.jsonl`; provisional sweeps do not publish samples.
The native mesh cache is `constant/mesh.npz`. The inspected historical cache has
303,264 cells and an actual .045 m cube-patch lattice. The new reference fine
level matches that near-body spacing but covers a larger domain; it is not the
same mesh, and matching spacing alone does not establish grid independence.

## Run and continue

The ordinary `setup.py` attaches the panel solver. From an installed OpenONDA
environment, `python setup.py` runs this configuration. `./allrun.sh` first invokes
`allclean.sh`, which removes this tutorial's generated solution, samples, figures
and mesh cache. Preserve results elsewhere before using that fresh-run launcher.

`./allcontinue.sh` restores `solution/backups/` without cleaning. The constructed
setup must match the checkpoint's numerical and coupled-output identity; an old
checkpoint does not silently adopt the new .25 s backup cadence. Running processes
retain the settings with which they were constructed.

The coupled factory owns MPI construction and solver lifetime. No manual MPI or
thread-environment setup is required in tutorial scripts.

## Panel-free assessment

The [separate study](../../../studies/panel_removal/EXECUTION.md) has passed the
short force/profile comparison through 4 s without panels, using the unchanged
historical native mesh and six permitted interface sweeps. It has not established
mature-flow agreement, mesh independence or an end-to-end speedup. To reproduce
that formulation in a fresh output directory, run from the repository root:

```bash
python -m studies.panel_removal.run_cube --output /path/to/new/cube_run --end-time 30 --snapshot-interval 0.25
```

This command excludes panels by default; the ordinary tutorial entry point does
not. The panel implementation remains needed by other formulations. See the
[portable-output guide](../../../studies/panel_removal/PORTABLE_RUNS.md) for external
storage, complete visualization trees and restart requirements.

## Reference and visualization

The [reference campaign](reference_flow/GRID_CAMPAIGN.md) defines four geometric
grids and a separate temporal control. Its fine case ends at 30 s; any common
comparison window must end by that time. Stationarity and GCI checks remain
necessary before declaring grid independence.

`./allplot.sh` uses PNG by default; `./allplot.sh pdf` selects the same thesis-sized
figures in PDF. The existing tutorial plotters use historical registered fine
results under `reference_flow/samples/fine/` and `reference_flow/solution/fine/`.
They do not automatically select a newly named campaign. Use the campaign's
explicit analysis commands for those results.

Open the FVM/VPM PVD collections with their complete referenced directories in
ParaView. Retained frames support animation; rolling coupled backups preserve
restart state. Compare only common saved physical times, retaining source coverage
and excluded solid regions in any reported error.
