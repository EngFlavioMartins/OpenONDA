# Remaining FVM plot inputs — 29 September 2026

The local airfoil, boundary-layer, standalone cube, cylinder-IBM and step
tutorials have no `solution/` or `samples/` trees. None has a results archive.
Their plot inputs must come from genuine runs of those cases. Existing coupled
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

No matching completed default-case runtime measurements were found in the
existing verification reports. Boundary layer and step are the smallest first
candidates. IBM and cube have thousands of steps; airfoil's actual mesh size
must be measured before quoting runtime or memory. Current cylinder benchmark
timings belong to different meshes and physical models and do not predict
these cases. The five cases require 56 scheduled field states in total plus
meshes, one current native backup per case, histories, diagnostics and figures.
For scale only, cube's 25 fields contain about 646,000 cell states; at 100 bytes
of scientific arrays per cell this is about 65 MB before geometry, additional
arrays and encoding. This is a storage illustration, not a measured archive
size or upper bound.

## Prepared isolated execution

The ignored `build/fvm_plot_completion/run_defaults.py` is prepared but has
**not been launched**. Run it with the final installed Python environment.
It materializes each unchanged default tutorial into a new isolated workspace,
executes `allrun.sh` then `allplot.sh` serially, and records setup hashes,
commands, return codes, elapsed wall time and resulting bytes in `summary.json`.
All command output goes to disk. A failed simulation skips that case's plots
and retains evidence; the script continues to the next case and returns failure
if any case is incomplete. It changes no physical parameters or health limits.

Successful isolated execution would supply genuine inputs and launcher evidence.
Portable archival, fresh-clone restoration and plotting, and visual review of
each resulting figure are still required to close the original every-allplot
request. Existing active workloads must be coordinated before launching this
sequential batch.
