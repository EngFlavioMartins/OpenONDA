# Cylinder mesh-convergence study

`allrun.sh` builds/checks three study meshes plus two physical-control meshes, executes an eleven-run campaign,
computes Cd/Cl/Strouhal statistics, and finishes with reports and plots. It selects
**medium or fine only if every error budget passes**. Coarse is the three-grid
convergence anchor, not automatically a qualified selection.

**Current blocker:** the laptop `fine_domain` trial generated 266,429 cells,
passed independent OpenFOAM `checkMesh`, and peaked at 1.74 GiB RSS during
mesh construction/checking. OpenONDA rejects its LSQ condition of 11.67 against
the production limit of 9. The other four final-profile meshes and full-cylinder
FVM memory/time are not yet measured. This is a provisional, resource-sized
configuration, not a runnable-study certificate. No quality limit is waived. See the
[qualification plan](../../../../REFERENCE_FLOW_QUALIFICATION_PLAN.md).
Setting up the campaign is not completion of those mesh repairs.

## Commands

Run inside `reference_flow`, in the OpenONDA Python environment:

```bash
./allrun.sh --dry-run       # Print configurations and estimates; launch nothing.
./allrun.sh --mesh-only     # Build and qualify study + physical-control meshes.
./allrun.sh                 # Run/resume campaign, then report and plot.
./allplot.sh                # Reanalyse/replot without solving.
```

On this Mac an explicit interpreter is available:

```bash
PYTHON=/opt/anaconda3/envs/OpenONDA/bin/python ./allrun.sh --dry-run
```

Use `--config /absolute/path/config.json --output /absolute/path/new-study` for a
different campaign. Copy and edit `study_config.json`; all fields are required.
Pass the same `--output` to `allplot.sh`. Changes to source or configuration
require a new output directory. Unrelated files and old `solution/` and
`samples/` directories are preserved. The old standalone postprocessor supplies
legacy spatial estimates only and cannot claim overall grid independence.

Exit codes: 0 = passed study, successful dry-run or successful mesh-only;
1 = operational/mesh/resource failure; 2 = inconclusive analysis.
Reports and plots are still written for an owned campaign after a failure.
Inspect `study_laptop/REPORT.md` and the named log rather than assuming every nonzero
exit means solver divergence.

## Campaign

Re=150 and D=U=1 are retained. The default is a **span-limited laminar cylinder**:
x/D in [-8,24], y/D in [-10,10], z/D in [-0.5,0.5]. The inlet is uniform U=1,
the outlet fixes pressure with reverse-flow-safe velocity, the cylinder is no-slip,
and lateral/span planes use slip. There are no physical cylinder end caps.
This approximates an infinite-cylinder laminar wake, not a finite-length 3D cylinder.
The unchanged in-plane near-body/wake boxes use the same cfMesh-style wrapper;
all target sizes scale together. Local transitions are dyadic but the inter-grid
refinement ratio is **2**, which the GCI calculation uses explicitly.

| Mesh | Wall spacing | Runs |
|---|---:|---|
| coarse | D/8 | fixed base dt |
| medium | D/16 | base dt, dt/2, dt/4, base dt with tighter solvers |
| fine | D/32 | base dt, dt/2, dt/4, base dt with tighter solvers |
| fine_domain | D/32 | x=[-12,28], y=[-12,12], same span, base dt |
| fine_span | D/32 | same in-plane domain, 25% reduced span z=[-0.375,0.375], base dt |

The domain and span controls must also produce stationary statistics; their
Cd/Cl/St changes enter the error budget. Force area is D times the actual span,
including 0.75D for `fine_span`. Span-limited modeling is motivated by Re=150 being
below the classical 3D wake-instability threshold near Re=188.5; see
[Barkley and Henderson](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/threedimensional-floquet-stability-analysis-of-the-wake-of-a-circular-cylinder/61575FBF0BC45054592D46382DEF30BB).
That result motivates the approximation, not a guarantee that these discrete
meshes are span-independent. See [the sizing decision](LAPTOP_SIZING.md).

D/4 is optional preflight, not part of production completion or statistics.
It remains available through `mesh.py --case very_coarse --output-dir <new-path>`.

Defaults: base dt=0.001, end time=250, earliest statistics time=100, minimum
20 whole shedding cycles, force interval=0.01. All times are convective because
D=U=1. A smooth transverse wake perturbation of amplitude 0.001U is identical in
physical coordinates on every mesh. The solver setter initializes its associated
time history and flux consistently. Force reference area is D times span.

The campaign uses serial execution, not an implicit MPI relaunch. Full fields
are written every 50 units; complete restart checkpoints every 5 units.
Rerunning reuses completed runs and resumes incomplete runs from the last
atomically published checkpoint/health pair. The solver rewinds force samples to
the checkpoint time. Only the latest two checkpoints per run are retained;
older campaign-generated checkpoints are removed after publication succeeds.

Every mesh source (generated, dictionary or loaded NPZ/MSH) is backed up by
default before FVM admission: `study_laptop/solution/<case>/mesh.vtu` is directly
readable in ParaView, with a lossless `mesh.npz` alongside it. Force samples go
to `study_laptop/samples/<case>/`. This applies separately to every timestep and
iteration-control case as well. A repeated startup preserves earlier mesh pairs
in `mesh-backup-*` folders before publishing the new pair. Mesh generation also
exports before quality rejection; `mesh_backup.json` distinguishes that raw
backup from an accepted mesh. A backup is not a qualification certificate.

### Resource gate

The target Mac has **16 GiB total RAM**, initially about 4.1 GiB available, and
about 20.7 GiB free disk. The new default working budget is **3 GiB**, with a
300,000-cell cap and an **8 GiB free-disk reserve**. Current available RAM is also
checked before each new build/run. These controls protect the laptop; raising a
budget does not supply memory. Dry-run prints conservative case estimates;
actual build/solver capacity evidence is recorded in the sizing decision.

Eleven long serial runs can still take substantial time. The smaller family is
not a promise of a particular accuracy: if D/32 fails the registered tolerances,
the report remains inconclusive rather than launching an unaffordable finer mesh.

OpenFOAM `checkMesh` must be available. The verified Mac runtime is detected
first, then PATH elsewhere. Every mesh requires an explicit `Mesh OK.` from
`-allTopology -allGeometry`; zero exit status alone is insufficient.

## Statistics and verdict

Each run uses its last N complete lift cycles after the discard time. Endpoints
may differ between grids as the frequency changes; cycle count and acceptance
rules are identical. Mean/RMS Cd and Cl are time weighted. Cd peak-to-peak and Cl
amplitude are averaged per cycle. St is D/U divided by mean cycle period, with
a spectral cross-check. Mean Cl is reported with an absolute tolerance because
its expected symmetric value is near zero.

Four cycle blocks provide an engineering sampling-uncertainty estimate, inflated
for positive adjacent-block correlation. Block drift, period stability, at least
100 samples/cycle and decimated-history sensitivity must pass. These estimates
are not rigorous confidence bounds. Short, drifting or absent-shedding histories
remain inconclusive; extend the end/discard times in a new campaign when needed.

Three-grid Richardson/GCI uses observed order and safety factor 1.25.
Positive fitted order is not presented as independent proof of an asymptotic
regime. Unresolved, non-monotone or implausible-order spatial differences do not
select a mesh. Additional refinement may be needed. Temporal error is estimated
for the **base dt** using dt, dt/2, dt/4; iterative sensitivity uses tighter linear
tolerances and extra PIMPLE corrections. Even roundoff-equal time results retain
their sampling uncertainty.

Default tolerances: mean Cd/St 1%; Cd RMS/peak-to-peak and Cl RMS/amplitude 2%;
mean Cl absolute 0.002. Spatial/temporal/iterative/sampling contributions must fit
40/20/20/20 percent of each metric's tolerance and their conservative sum must
fit the total. Domain/span sensitivity must each fit within 20% of each metric's
tolerance and are added to the same total; they are not free extra error allowances.
All required runs must qualify. Stored health requires finite
fields, converged linear solves, max Courant <=0.9 and continuity/equation residual
<=1e-4, alongside the solver's existing abort limits.

The method separates [spatial convergence](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html)
and [temporal convergence](https://www.grc.nasa.gov/www/wind/valid/tutorial/tempconv.html)
as in NASA's guidance. The budgets/cycle minimum are project choices, not
NASA-prescribed constants. Results concern only these observables and this fixed
quasi-2D model and the tested domain/span variations, not general physical validation.

## Outputs

Under `study_laptop/` (or `--output`); old `study/` results remain untouched:

- `REPORT.md`: Cd/Cl/St table, selected mesh, unmet requirements.
- `status.json`: compact phase/progress/failure record for economical monitoring.
- `grid_study.json`: cycle statistics, mesh identities, health and error budgets.
- `figures/grid_study.png` / `.svg`: size trends and sampling error bars.
- `figures/force_histories.png` / `.svg`: Cd/Cl histories and selected windows.
- `figures/error_budgets.png` / `.svg`: candidate error contributions when available.
- `meshes/<grid>/`: NPZ/VTU, provenance, mesh report and independent check.
- `solution/<run>/`: initial VTU/NPZ mesh backups, fields and solver run manifest.
- `samples/<run>/`: force histories.
- `runs/<run>/`: checkpoints and progress metadata.
- `mesh-*.log`, `flow-*.log`: execution logs.

Medium is selected when all its gates pass, otherwise fine if it passes,
otherwise **none**. Incomplete plots explicitly show that no mesh qualifies.

## Verification

`tests/fvm/test_cylinder_study_campaign.py` covers known synthetic convergence,
stationary/drifting/absent signals, uncertainty/time/iteration rejection,
resource/provenance protection, incomplete reports, plot rendering, and a tiny
actual FVM checkpoint/resume integration. Synthetic figures test the reporting
code, not the physical cylinder. Full-cylinder qualification remains separate.
