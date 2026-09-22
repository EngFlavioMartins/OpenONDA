# Coupled flow around a cube

A body-fitted FVM region resolves the viscous wall of a unit cube at Re=1000.
The surrounding VPM transports vorticity, using RK2, grid diffusion, Gaussian
particles and FMM induction. Both solvers use equilibrium Smagorinsky closure.
Freestream speed is 1 m/s and kinematic viscosity is 0.001 m²/s.

| Quantity | Current setup |
| --- | --- |
| Cube | [−.5,.5]³ m |
| FVM domain | [−1.485,1.485]³ m |
| Transfer region | [−1.45,1.45]³ m |
| Uniform FVM spacing | .045 m |
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
The native mesh cache is `constant/mesh.npz`. The FVM box uses the reference
fine spacing everywhere, with no refinement or coarsening regions. Its bounds
are snapped inward from ±1.5 m so the uniform lattice does not enlarge the
compact coupled domain.

## Run and continue

From an installed OpenONDA environment, `python setup.py` runs this configuration.
`./allrun.sh` first invokes `allclean.sh`, which removes this tutorial's generated
solution, samples, figures and mesh cache. Preserve results elsewhere before using
that fresh-run launcher.

`./allcontinue.sh` calls `setup.py` without cleaning. It automatically restores
`solution/backups/`, or starts at zero when no backup exists. The constructed
setup must match the checkpoint's numerical and coupled-output identity; an old
checkpoint does not silently adopt the new .25 s backup cadence. Running processes
retain the settings with which they were constructed.

The coupled factory owns MPI construction and solver lifetime. No manual MPI or
thread-environment setup is required in tutorial scripts.

## Reference and visualization

The [body-fitted reference](reference_flow/README.md) is an explicit Re = 1000
four-grid study. Its physics live in `reference_flow/setup.py`, while its grid
names and baseline spacings are visible directly in `reference_flow/allrun.sh`.

`./allplot.sh` uses PNG by default; `./allplot.sh pdf` selects the same thesis-sized
figures in PDF. It uses a complete historical `fine` reference when present,
otherwise the finest completed `grid_h*` reference run. The selected grid is
recorded in the comparison report. Only common saved physical times are plotted.

Open the FVM/VPM PVD collections with their complete referenced directories in
ParaView. Retained frames support animation; rolling coupled backups preserve
restart state. Compare only common saved physical times, retaining source coverage
and excluded solid regions in any reported error.
