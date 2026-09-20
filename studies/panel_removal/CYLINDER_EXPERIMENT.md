# Panel-free cylinder experiment

This study compares a coupled FVM–VPM solution with the fully meshed Re=150
cylinder reference. The VPM represents an infinite span; the FVM resolves the
viscous wall and wake formation region. It does not model finite-cylinder ends.
See [the method](planar_model.md) for equations and supported configurations,
and [measured results](EXECUTION.md) for the current qualification limits.

## Run and restart

Use the installed OpenONDA environment from the repository root. The launcher
requires the passing 4 s cube identity/force/profile gate in `cube_gate.json`.
Use a new output directory for each independent experiment:

```bash
python -m studies.panel_removal.run_cylinder \
  --output studies/panel_removal/runs/cylinder_assessment --cores 2 --end-time 100
```

`--describe` validates and prints the configuration without meshing or solving.
`--mesh /path/to/native/mesh.npz` reuses a saved mesh for a fresh replay; actual
bounds, wall spacing and span layers must match. `--max-coupling-steps 50`
checkpoints after 2 s while retaining the configured 100 s horizon. Continue
that output with the same physical settings and omit the short-run limit:

```bash
python -m studies.panel_removal.run_cylinder \
  --output studies/panel_removal/runs/cylinder_assessment --cores 2 \
  --end-time 100 --restart
```

Restart requires the run's own native mesh and coupled checkpoint. Fresh output
is protected against overwriting. Forces and lines are saved. Retained FVM/VPM volume frames and rolling coupled
checkpoints use a 0.24 s interval, six accepted coupling steps. An exact 0.25 s
interval is rejected because it falls between those steps. The qualified trajectory
is stored in `runs/cylinder_no_panel_converged`. Its exact mesh identity is
recorded in [native_mesh_identity.json](runs/cylinder_no_panel_converged/native_mesh_identity.json).

## Physical and numerical configuration

Lengths are in m, time in s, D=1 m and U∞=1 m/s.

| Quantity | Baseline |
| --- | --- |
| FVM domain | x=[−2.5,6.5], y=[−3.5,3.5], z=[−0.5,0.5] |
| Transfer region | x=[−2,6], y=[−3,3], z=[−0.375,0.375] |
| FVM wall spacing | Requested .04; nominal realized .03 |
| FVM span | 1D, four layers, slip end boundaries |
| FVM time step | .004 |
| Coupling interval | .04, at most six sweeps, 1e-5 interface tolerances |
| VPM representation | Gaussian filaments, z=0, represented span 1D |
| Particle spacing | .05 |
| Interior transfer vorticity floor | .05 s⁻¹; strength cutoff .000125 m³/s |
| GBD/release vorticity floor | .01 s⁻¹; strength cutoff .000025 m³/s |
| Particle retention bounds | x=[−8,24], y=[−10,10] |

The historical reference's mean centreline over 80–100 s remains negative
through x/D=1.52. A transfer edge at x/D=1.25 would cut the recirculation region.
The study domain contains that region and the transverse profiles at x/D=1,2,4;
a separate domain-size check is still needed before reducing cost.

The FVM shares the reference's convection, gradients, implicit Euler integration,
corrector counts, tolerances, relaxation and viscosity. Its step is fixed at
.004 s; the reference adapts its step below a .004 s ceiling. Particle strengths
and volumes are omega_z*h²*span and h²*span. Continuous planar renewal preserves
circulation and first moments. Cutoff sensitivity is explicit through
`--transfer-cutoff` and `--gbd-vorticity-floor`.

The native FVM wall provides solid exclusion. No panel bodies or panel solver
are attached. Pressure remains FVM-owned; VPM profile queries use compatible
velocity/derivative operators. Direct filament induction costs O(N²): the
100,000-particle cap is a capacity limit, not a performance or convergence claim.
Reference slip outer boundaries differ from unbounded VPM induction, so far-domain
sensitivity remains a distinct assessment.

`--particle-spacing`, `--coupling-dt`, `--downstream` and `--half-height` support
independent sensitivity runs. Change one parameter at a time in a fresh directory.

The requested 100 s horizon is a run limit, not a guarantee of stationary
shedding. A 50–100 s candidate window must pass the same cycle-count and drift
checks. See [portable runs](PORTABLE_RUNS.md) for output storage and older-run
restart requirements.

## Comparison and acceptance

For a fresh 100 s assessment, compare completed matched-reference results with:

```bash
python -m studies.panel_removal.compare_cylinder_run \
  studies/panel_removal/runs/cylinder_assessment \
  --reference tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/campaigns/geometric_xy_100s/spatial/samples/xy_fine \
  --start 50 --end 100 --output studies/panel_removal/cylinder_comparison.json
```

The comparator uses identical saved physical times, without phase fitting or
force scaling. Mean profiles use trapezoidal time integration. Centreline fluid
segments are clipped to reference support; wall-adjacent values are not
extrapolated. Reports retain covered extents, excluded points and missing times.
Only consumed force fields are numeric inputs; the sampler's textual patch field
is retained in the source CSV.

| Mature-flow criterion | Required result |
| --- | --- |
| Mean Cd | Difference ≤2% |
| Lift RMS | Difference ≤5% |
| Strouhal number | Difference plus uncertainty ≤3% |
| Mean vector profiles | RMS error <3% U∞ for every FVM/VPM line |
| Exterior VPM centreline, x/D>6 | RMS error <3% U∞ |
| Cd half-window drift | <1% |
| Spanwise velocity variation | <.001 U∞ |
| Shedding coverage | At least eight complete periods |
| Interface iterations | Every step in the window converged |

Every force/interface timestamp must cover the window at the declared coupling
cadence. Profiles and span probes must cover every five coupling steps. FVM and
VPM lines require all three finite velocity components on their declared .08D
lattices. The VPM centreline covers x/D=.6–12; transverse lines cover y/D=−3–3.
Every span frame must contain nine finite probes at x/D=1.5, y/D=0 and z/D=−.45–.45.
Full FVM donor-stack invariance is checked before transfer, without projecting
away a three-dimensional donor field. Missing data or identity prevent admission.

Frequency is estimated independently from upward crossings of mean-subtracted
lift. Crossings are interpolated only between adjacent saved samples. The
estimate uses complete periods, requires at least three in each window half,
at least twenty samples per period, and agreement with the dominant FFT band.
Period coefficient of variation and half-window frequency drift must each be ≤3%.

If the first crossing lies in [a0,b0] and the last in [aN,bN], the N-period
frequency lies in [N/(bN−a0), N/(aN−b0)]. The uncertainty bound is the largest of
this sampling interval's deviation from the estimate, the converted Student-t
period-mean uncertainty, and half the frequency drift. Correlated fluctuations
may require longer records; this screening bound does not assert independence.
Both runs' uncertainties enter the 3% comparison. Equal FFT peak bins alone do
not prove frequency agreement. The reference-grid FFT estimates retain their
separate bin-resolution limitation.

Startup records remain provisional. Historical reference data ending at 100 s
do not generally provide enough cycles in 80–100 s for mature admission.
`--plot`, `--plot-vpm` and `--snapshot-time` produce force/profile figures from
actual saved times; they do not change the numerical gate. PNG is the default;
`--format pdf` selects the same fixed-size figures in PDF.

## Completion stage

`complete_cylinder_comparison.py` remains the one-shot local process attached
to the existing 160 s experiment/reference pair. It does not target the new
100 s assessment; use the explicit comparator command above for that run. Attach it after any intentional short-run stop
has been resumed continuously and `process.json` records that launcher:

```bash
python -m studies.panel_removal.complete_cylinder_comparison
```

It pins launcher PID, start time and command; a kernel lock prevents duplicates.
It requires access to process inspection. Status is written to
`runs/cylinder_no_panel_converged/mature_comparison/status.json`. Restarted or
reused PIDs, explicit failures, or dead processes with incomplete output require
investigation rather than silent adoption.

Experiment completion requires FVM/VPM terminal lifecycle and time 160 s,
committed coupled backup step 4000/time 160 s, final force/interface records and
launcher exit. The reference requires `xy_fine` registration plus native terminal
lifecycle and samples at 160 s. Registration or a checkpoint alone is insufficient;
later reference-grid/control cases may continue while comparison runs.

The stage evaluates 80–160 s once, writing `comparison.json`, force/profile
figures and `comparator.log` under `mature_comparison`. Exit 0 means the mature
gate passed; exit 3 means postprocessing completed but scientific criteria failed;
exit 2 means operational failure. It does not launch simulations or establish
separate grid, coupling-step or particle-spacing independence.

## Span-probe evidence

The retained 0–2 s startup CSV used an affine stencil whose XY neighbours changed
with z. The [checkpoint audit](cylinder_span_probe_audit.json) reproduces its
.002401 variation even on exactly planar input, while actual cell-stack deviation
is 1.37e-5. Nearest-cell sampling observes all four layers of one XY stack and
retains the .001 threshold. The first resumed sample at 2.20 s passes, with
maximum component range 6.81e-6 U∞. Historical values remain unchanged.
Reference force/profile convergence calculations do not consume this span probe.

Reproduce the original diagnostic from its exact checkpoint with:

```bash
python -m studies.panel_removal.audit_cylinder_span_probe \
  studies/panel_removal/runs/cylinder_no_panel_converged \
  --checkpoint solution/backups/fvm_000050 --time 2 \
  --expected-checkpoint-sha256 3b79b432f28909ca4e2cfd286b422a5c7728613e4b0920b83829e23760c75877 \
  --output /path/to/new_span_audit.json
```

The output must be new. Reproduction requires the original mesh, span samples
and all checkpoint files with the stated hash. Rolling retention can remove that
checkpoint; another time is not a substitute. The saved audit retains measured
values, hashes, raw stack velocities and method if the checkpoint is unavailable.
