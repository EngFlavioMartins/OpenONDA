# Remaining FVM plot inputs — 29 September 2026

The airfoil, standalone cube and cylinder-IBM tutorials still need completed
default runs and portable result archives. Boundary-layer and step now have
genuine completed default runs, committed archives, and successful plotting
from a fresh Git export. Existing coupled
cube and VPM archives describe different cases and cannot replace them.
Taylor–Green now has a genuine archived default 24×24, ten-step history through
t=0.05; its completion and clone plotting evidence are tracked separately.
Coupled NACA and the full cylinder campaign also remain outstanding.

Five shared FVM plotting helpers already apply the thesis theme and validate
canvas geometry. Their CSV readers now reject header-only histories, which
previously allowed a launcher to return success without producing figures.
Six pure reader regressions cover empty rejection and nonempty numeric data.
Taylor–Green's legend uses “total enstrophy” rather than a literal underscore
under LaTeX. No missing scientific data were synthesized or substituted.

## Default computation sizes

Counts below come from the coordinate/mesh formulas, without constructing a
solver or advancing a case. Adaptive stepping can take **more** steps than
the optimistic count obtained at the configured maximum timestep.

| Case | Default cells | Horizon [s] | Initial dt [s] | Effective maximum dt [s] | Optimistic minimum steps | Scheduled field states including initial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Boundary layer | 2,460 | 8 | .005 | .020 | 400 | 5 |
| Step profile | 2,112 | 12 | .020 | .050 | 240 | 7 |
| Cylinder IBM | 12,642 | 60 | .010 | .01171875 | 5,120 | 13 |
| Standalone cube | 25,822 | 120 | .020 | .050 | 2,400 | 25 |
| Airfoil | Requires native surface meshing; not measured | 25 | .005 | .020 | 1,250 | 6 |

Boundary layer has 82×30 cells. Step has 24×8 upstream plus 120×16
downstream cells. IBM has 129×98 cells, including the immersed solid region;
its forcing Fourier cap is 0.1h²/nu=0.01171875 s. Cube has 221×118 cells minus
the 16×16 square hole. Airfoil uses a finite-span STL and graded Cartesian
refinement down to .03125, so assigning it a rectangular 2D cell count would
be misleading.

Boundary layer completed in 232.7 s including invocation overhead; step completed
in 57.7 s on this loaded host. IBM and cube have thousands of steps; airfoil's actual mesh size
must be measured before quoting runtime or memory. Current cylinder benchmark
timings belong to different meshes and physical models and do not predict
these cases. The five cases require 56 scheduled field states in total plus
meshes, one current native backup per case, histories, diagnostics and figures.
For scale only, cube's 25 fields contain about 646,000 cell states; at 100 bytes
of scientific arrays per cell this is about 65 MB before geometry, additional
arrays and encoding. This is a storage illustration, not a measured archive
size or upper bound.

## Isolated execution

The ignored `build/fvm_plot_completion/run_defaults.py` has been executed with
the installed Python environment.
It materializes each unchanged default tutorial into a new isolated workspace,
executes `allrun.sh` then `allplot.sh` serially, and records setup hashes,
commands, return codes, elapsed wall time and resulting bytes in `summary.json`.
All command output goes to disk. A failed simulation skips that case's plots
and retains evidence; the script continues to the next case and returns failure
if any case is incomplete. It changes no physical parameters or health limits.

The first IBM execution exposed inconsistent mixed-boundary branch selection
during pressure correction. The shared FVM fix passed the unchanged 12,642-cell
case through t=0.6 s, beyond its original t=0.15546 s failure. This is a bounded
regression, not completion of its 60 s horizon. See the
[projection audit](../../studies/fvm_ibm_projection_audit.md).

A subsequent full IBM/cube/airfoil batch was deliberately interrupted during IBM
startup to avoid overlapping memory peaks with existing user simulations and
the CUDA quadcopter qualification. It produced no completed result archive.
The user simulation trees were untouched. Remaining full cases must run
sequentially with enough memory available.

Boundary-layer and step were rerun unchanged using the wheel exported from
`91b668b5`, containing the shared FVM correction. Both `allrun.sh` and
`allplot.sh` returned zero. Their lossless bundles were committed in `6851cc89`.
A fresh Git export with no samples or solution directories restored each bundle
through ordinary `allplot.sh`; both launchers returned zero and all four image
hashes matched the original runs. See the
[execution and archive record](../../studies/fvm_small_cases_archive_verification_2026-09-29.json).

## Subsequent real boundary-layer and step execution

The isolated default boundary-layer and step runs completed. Boundary layer
took 865 accepted steps and logged 202.7 s total solver wall time. Their original
plots exposed 8 pt font overrides, missing vertical layout and an 18.034 cm
step-comparison canvas. The five FVM families now inherit the fixed thesis font;
their helpers fit vertical layout before validating symmetric margins. Step
comparison uses the shared 12.5 cm width and a horizontal colorbar. Both real
`allplot.sh` launchers subsequently exited 0 with the installed package. All four
figures passed strict thesis validation and visual inspection. Source/data
copies and `plot_cleanup_evidence.json` are under
`build/fvm_plot_completion/runs-20260929T150300195325Z/`.

Boundary-layer results remain **out of band**, not scientifically validated.
The profile plot had used requested stations .25/.5/.75 instead of recorded
sample columns .243056/.493056/.743056. Correcting the similarity coordinate
and labels to recorded x gives maximum profile errors .0416/.0565/.0759; the
overall .0759 still exceeds .05. Skin-friction error over .2<x/L<.95 remains
15.34% mean and 25.90% maximum against the fixed-freestream Blasius reference,
with a 5% mean target. Step ends at reattachment x/h=3.59 at t=12; this does not
establish stationary or mesh-converged reattachment.

Saved boundary-layer fields show top velocity 1.017–1.041 over the skin-friction
comparison interval and profile peak velocities 1.030–1.045. A diagnostic
rescaling of the Blasius skin friction by the local top-speed ratio to the power
1.5 reduces mean disagreement to 10.07% and maximum to 18.52%, but still misses
the target. It is not a replacement acceptance reference: local acceleration
violates the uniform-edge-speed assumption. Finite domain height/pressure
gradient and discretization effects are plausible contributors; these data do
not isolate a solver defect or establish that coarse resolution alone explains
the discrepancy. The first wall-cell height is .0015 and the plate has 72 cells
along x.

A minimal next qualification would separately double plate resolution to 144,
halve first wall-cell height to .00075 at the same domain and stretch ratio,
then test height .70 with the original central resolution. Keep physical Re,
duration, solver tolerances and timestep rules fixed and compare actual sampled
x coordinates, edge speed, pressure gradient, profiles and wall shear. A paired
smaller-timestep check is needed before attributing changes solely to mesh or
domain. These are proposed isolated studies; none was launched or substituted
for the current default results.
