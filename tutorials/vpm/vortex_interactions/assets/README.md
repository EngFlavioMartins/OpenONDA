# Vortex-ring stabilization experiments

Start with the [core-transport physics controls](../../../../docs/reviews/2026-09-vpm-core-transport.md). No stabilization has qualified against the LBM trajectory yet. `allrun.sh` now compares corrected Gaussian initial conditions with CS, GBD/M4' and experimental GBD/Lagrange6, without LES or a seed.

The earlier LES + transposed stretching + SSPRK3 campaign remains available.
The audit corrected a stale-vorticity health check that stopped the original
leapfrog control at step 22: with that correction, the same evolution completed
1,200 steps (`t Gamma/R0^2 = 29.4`). Added stabilization has not demonstrated a
physically validated extension through collision breakdown at these spacings.

The [implementation audit and measured results](../../../../docs/reviews/2026-09-vpm-stabilization.md)
explain the publication correspondence, fixes, rejected trials, mode behavior,
resolution controls and remaining limitations. No rVPM formulation is used.

## Running

From this tutorial directory:

```sh
./allrun.sh                                # molecular CS / GBD / higher-order GBD controls
./allrun.sh --campaign recommended         # earlier LES baseline control
./allrun.sh --dt .0075 --steps 800          # same physics horizon, smaller timestep
./allrun.sh --campaign strategies --quick  # baseline, splitting, remeshing, weak P + remeshing
./allrun.sh --campaign screen --quick      # broader single-method and combined screening
./allrun.sh --campaign legacy              # original six setup.py configurations
```

`--steps`, `--dt`, `--spacing`, `--support` and `--scenario` select controls.
The physics campaign fixes amplitude=0, Cs=0 and a retained initial Gaussian tail; its default horizon is 400 steps at dt=0.015 and spacing 0.04. It reserves 500,000 particles and saves fields every 20 steps. Capacity contact still needs checking for each run.
For seeded experiments use `study.py --seed-direction axial --support disturbed --amplitude .05`; this is a different benchmark from the unperturbed LBM trajectory.
Runs are serial to avoid competing for one GPU. `OPENONDA_PYTHON` selects the
Python executable. A stopped or rejected case does not prevent later cases
from running; read each recorded termination status rather than interpreting a
zero launcher exit code as physical success.

Compatible completed results are reused. Changed parameters or source create
an archived previous run; `--clean` forces new study runs and archives their
previous directories, while removing the legacy outputs. `--no-plot` skips
summary generation. Large remeshing branches can consume much more time than
the baseline: the particle cap is an accuracy/capacity gate, not a wall-time cap.

## Physical and numerical controls

Both interactions use `R0=1`, `Gamma0=pi`, `a0=0.1`, initial separation `1 R0`,
`Re_Gamma=3000` and Gaussian particles. Collision reverses the second ring's
circulation. The earlier stabilization campaigns use core spreading and
Smagorinsky `Cs=0.20`; their prescribed disturbance is radial, mode eight,
amplitude `0.05 R0`. It is not automatically an unstable
eigenmode, and it is not the unperturbed Cheng et al. Fig. 5 reference problem.
See [reference provenance](references/README.md).

The stabilization campaigns use support following the disturbed centreline. This avoids
azimuth-dependent truncation of the seeded Gaussian tube and keeps the initial
mode more consistent under refinement. `--support circular` reproduces the
original layout. The legacy runner keeps that original support. Compare runs
with the same support, amplitude, spacing, time step and LES parameters.

With the corrected Gaussian tail at spacing 0.04, all three physics controls
start with the same 68,952 particles and `sigma=0.04`. The additional wider-blob
CS diagnostic used 27,448 particles and `sigma=0.08`.
Saved older runs used a more severely truncated Gaussian and
fewer particles. The production health limits remain
CFL 1.0, relative divergence 0.12, and current-curl misalignment 25 degrees.
They have not been relaxed to obtain longer trajectories.

## Profiles and interpretation

- **Baseline:** preferred until an added method demonstrates both useful
  survival and acceptable resolved-field behavior.
- **Splitting:** checks every five steps for a factor-two lineage or absolute
  strength increase, with symmetric quarter-length displacement. It refines
  sampling along a filament; unchanged Gaussian cores mean it cannot restore
  lost transverse/core resolution.
- **Remeshing:** Gaussian variance redistribution, without another viscous
  step, with moment restoration and event limits of 2% energy and 5% enstrophy
  loss. The comparison profile checks every 50 steps, resets to the initial
  particle core, and uses a 1% strength-tail budget. This profile is a recorded
  comparator, not a demonstrated collision remedy.
- **Weak realignment plus remeshing:** the same remeshing with unnormalized,
  moment-corrected P-relaxation at `f=0.384684814725 /s`; its blend is `f*dt`
  (0.003 at the default step). Holding the blend fixed under dt refinement
  would change the physical relaxation frequency.
- **Screen-only methods:** residual viscosity, strong normalized realignment,
  constrained divergence relaxation, weak realignment alone, combined
  splitting/remeshing and direct solenoidal grid remeshing. The latter is a
  classical Helmholtz-projection adaptation; tested capacity failures prevent
  recommending it for this laptop-sized problem.

Particle-volume changes also change the current LES filter `Delta=V^(1/3)`.
A spacing study at fixed Cs therefore changes both discretization and the LES
model. The study's `--smagorinsky` option supports nominal `Cs*h`-matched
controls; that is not an exact fixed-filter model on nonuniform particles.

## Inspecting results

Each `study_results/<tag>/result.json` records the parameters, source fingerprint,
stabilization settings, termination and total wall time. CSV diagnostics and
compressed particle snapshots are stored alongside it. Results stopped by
health limits, rejected stabilization proposals, numerical errors and manually
terminated cost screens must remain distinct. The `qualified_` run prefix
identifies the corrected implementation campaign, not physical certification.

```sh
python -m tutorials.vpm.vortex_interactions.assets.analyze_study --runs RUN_NAME
python -m tutorials.vpm.vortex_interactions.assets.render_study RUN_NAME
python -m tutorials.vpm.vortex_interactions.assets.render_vorticity RUN_NAME --steps 0 100 200
python -m tutorials.vpm.vortex_interactions.assets.audit_fields RUN_NAME --steps 0 100 200
```

These module commands run from the repository root. `render_study` shows actual
particle snapshots; `render_vorticity` reconstructs the Gaussian field in a
meridional plane. Particle colors cease to be material labels after remeshing.
The common-box field audit avoids runtime energy-definition changes. Its grid
spacing and padding can be varied independently of the simulated field.
Pre-audit FFT energies are superseded by remeasurement of saved fields.

Long survival, conserved global moments, low misalignment and a visually smooth
ring are separate observations. None alone validates the growth of the seeded
mode, reconnection, or late turbulent breakdown.
