# Wind-turbine wake

This study has not passed a full 10 s run or a grid/time-step convergence
check. An earlier, unstabilized run stopped on a strain-based time-step check
at 7.68 s. The current settings add two existing wake-stabilization methods;
that historical stop cannot establish their late-time behavior. Rotor loads
and induction still need spatial, temporal, and particle-core convergence checks. See the [VPM guide](../../../docs/vpm.md#vlm-coupling)
for the lifting-surface model and its limitations.

The setup now also samples four streamwise lines at design `r/R = 0, 0.25,
0.65, 1.1`, from `x/D = -1` to `3`, every `0.06 s`. Their native
`streamwise_r*.csv` files retain `time`, `step`, positions, and all three signed
velocity components. The output owner appends these records across accepted
steps and restarts. An older single-snapshot CSV cannot be resumed into this
schema; use a fresh output namespace to preserve the original data.

The wake uses selective eddy viscosity at coefficient 0.5 and conservative
filament splitting every five steps when a particle's strength doubles from
its lineage reference. `max_n_particles` sizes the solver arrays.
An older unstabilized checkpoint is incompatible with this setup; start a
fresh run.

`assets/plot_rotor_streamwise.py` plots axial deficit `1 - ux/U` and signed
`uy/U`, `uz/U` versus `x/D`, using an exactly bracketed five-revolution time
mean. It refuses missing or incomplete histories. `allplot.sh` includes this
figure; old retained results without the new lines will need a fresh run.

Run `./allrun.sh` for a clean simulation, `./allcontinue.sh` to resume the
latest backup, and `./allplot.sh` for figures. Use `./allplot.sh pdf` for vector
figures. `python setup.py` also continues automatically. To use a separate
case namespace, invoke `python setup.py --output-tag <name>` and select the
same namespace when plotting.

The ordinary `setup.py` starts and continues compatible runs. An explicit, bounded smaller-step
restart pilot is available for a validated native checkpoint:

```bash
python assets/run_restart_pilot.py \
  --resume solution/vpm_001152.h5 \
  --resume-dt 0.001 \
  --attempt pilot
```

The pilot performs an eight-step sampler-free preflight and then at most 120 accepted
steps. It recomputes
the force and wake-plane cadences from the requested step, preserves the native VLM
geometry and plane definitions, and writes an isolated identity-derived
`solution/<attempt>_from_step_...` plus `samples/rotor/<same-token>` namespace. A
native start backup is written before the pilot; the solver records source and
requested step sizes in `vpm_metadata.json`. Default restart identity remains strict
for every numerical setting except the explicit changed-step override. The pilot is
not supported for DVH because its checkpoint does not retain enough physical-time
diffusion history to remap the accepted-step counter. Diagnostic grid histories are
not serialized in native numerical checkpoints, so pilot samples start fresh and
must not be concatenated with the source run without explicit reconciliation.

The three-bladed turbine has a 6 m radius, 7 m/s wind and tip-speed ratio 7. Its
prescribed rotation ramps smoothly over one nominal revolution. Attached VLM force,
power and loading tables are recorded on every accepted VPM step. Full VPM/VLM
numerical backups follow blade motion at `0.024 s` (four authored `0.006 s`
steps); only those accepted steps with a numerical backup receive a VLM geometry
companion. Each coupled HDF5 backup contains the restartable VPM particle state
and VLM surface state used by the animation. There is no independent VLM geometry
sampler. Native VLM companion surface files,
metadata and checkpoints are under `solution/`, while mandatory blade
force/power/loading records and VPM velocity-field samples remain under
`samples/rotor/`. Plotters read the run's own geometry, motion, density and sample
times; they never regenerate design inputs or reconstruct a solver from a backup.

Blade chord Reynolds numbers are approximately 0.47–1.15 million, using the recorded chord, molecular viscosity and nominal relative speed `sqrt(U² + (Omega r)²)`. The wake LES resolves a different part of the flow from the inviscid blade-loading model.

The performance figure uses wind-turbine coefficient definitions:

- `CT = T / (0.5 rho U^2 A)`;
- `CP = P / (0.5 rho U^3 A)`, with positive `P` denoting extracted shaft power.

The native console also reports generic lift/drag coefficients using its displayed reference pressure and blade area. Use the disk-based `CT` and `CP` in these figures for the rotor theory comparison.

Power is the native fluid-on-blade rotational power, evaluated using actual angular speed, including the ramp. BEM uses the same recorded blade geometry, a thin-plate lift polar, Prandtl hub/tip losses and Buhl's high-induction relation. Loading profiles compare time-averaged sampled circulation and sectional lift with BEM. Operating-point means and wake profiles use the final five nominal rotor revolutions; the impulse balance uses a separate final three-revolution clock window. The ideal far-wake reference includes streamtube expansion. Wake stationarity requires a complete five-revolution native window and compares its first two whole revolutions with its final three using exact bracketed time-weighted means, so every portion of the published mean is evaluated. Subtracting the freestream and retaining local velocity vectors prevents opposite spatial changes from cancelling. Curves with more than 1% drift, or incomplete native windows, are dotted. The validator also reports persistent induced-velocity signal onset at each plane using a predeclared 1%-of-freestream RMS threshold for three consecutive native frames; this is not a physical convected-front proof. A five-revolution window is refused as stationary unless onset precedes it, and 7.5 s may therefore remain insufficient. No unrequested 3D/5D context is emitted by the completion setup.

The current 10 s run writes numerical backups every four steps and compact
wake fields every 0.06 s. Filament splitting changes particle counts, so
storage estimates from the older unstabilized run do not describe this case.

`python assets/validate_results.py --pre-plot` checks
completion, force/power stationarity, BEM agreement, coupled bound-plus-wake
impulse, complete mean-field windows and reports finite-distance axial/azimuthal
induction errors at both required stations. It also reports conservative
particle-front brackets from native numerical checkpoints when available; these
brackets are evidence only and do not certify physical convected-wake arrival.

The latter uses the analytical right-vortex-cylinder equations of [Li et al. (2025), Eqs. 30–45](https://wes.copernicus.org/articles/10/2515/2025/) and the planar superposition/closure described by [Li et al. (2022), Sect. 3](https://wes.copernicus.org/articles/7/75/2022/). It uses the matched BEM bound-circulation table to form piecewise-constant cylinder sheets; it does not fit a curve to the VPM samples. The predeclared scaled-RMS screens are 25% for axial induction and 35% for azimuthal induction, with reference floors of 5% and 2% of the relevant velocity scale. These screens are explicitly diagnostic flags, not uncertainty-based acceptance gates: no defensible uncertainty budget is claimed for the infinite-blade, inviscid, non-expanding reference against the finite-blade LES wake. The report includes dimensional RMS errors and finite-bin coverage so weak or zero swirl cannot be hidden by a floor or profile average. A flag keeps theory agreement unqualified; do not tune the screens or silently replace the model with the ideal far-wake curve.

`allplot.sh` additionally writes `figures/rotor_induction_validation.{png,pdf}`, showing native and finite-distance reference axial and azimuthal induction profiles at 1D/2D. It also writes `assets/animation/rotor_30fps.gif` from the recorded coupled VPM+VLM backups and nearest recorded 1D wake fields. The sidecar JSON records the exact source backups and wake-plane files. The GIF title reports accepted physical time and its accelerated playback multiplier; no solver state is interpolated or fabricated. The authored 7.5 s horizon precedes the older run's t=7.68 s CFL stop but is not a selected healthy endpoint or before-failure proof: report any missing signal onset, late growth or stationary window rather than relabeling startup as steady validation.

This is an inviscid blade-loading model coupled to a viscous/LES particle wake. It does not resolve blade boundary layers, transition, stall or airfoil profile drag. BEM agreement validates the corresponding attached-flow approximation, not every effect in a high-Reynolds-number turbine.

Reference: [CCBlade theory](https://wisdem.readthedocs.io/en/master/wisdem/ccblade/theory.html), citing Ning's bracketed inflow-angle method, Prandtl losses and Buhl's correction. The shared reference implementation lives in `openonda.rotor_theory`; independent momentum/scaling tests accompany it.

The wake uses the VPM solver's default Smagorinsky coefficient, core spreading,
and Gaussian particles.
Selective viscosity damps local positive stretching (`C=0.5`); filament
refinement checks every five steps and splits particles whose strength has
doubled since their lineage reference. Both settings come from the vortex
interaction study. Core spreading changes particle width through viscosity;
filament refinement addresses stretching. The previous `7.68 s` failure is
historical evidence, not proof that these settings complete the 10 s case.
Loads include native unsteady pressure. Pedrizzetti relaxation is disabled for
momentum validation: rotating particle strengths can introduce substantial
momentum even when their individual magnitudes are preserved. If enabled for a
separate numerical study, the native flow-integrals sampler records its
cumulative vector-strength, linear-impulse and angular-impulse transfers. Those
transfers are numerical sources, not turbine forces.

The non-remeshing late-growth diagnostic is staged separately from qualification.
The changed-step restart route is intentionally model-strict: it permits only
an explicit `time_step_size` change, so it must reject applying stretching
viscosity to the baseline checkpoint. Use the public fresh-pair runner instead:

```bash
python assets/run_matched_stabilization_pair.py \
  --variant baseline --output-tag matched_baseline_7p5 --endpoint 7.5
python assets/run_matched_stabilization_pair.py \
  --variant selective_eddy_viscosity --output-tag matched_stabilized_7p5 \
  --endpoint 7.5
```

This comparison isolates selective viscosity: one case disables stabilization,
and the other uses `C=0.5`. The main tutorial now combines that viscosity
with filament refinement, so the pair does not validate the main case. It
requires two independent runs and output directories.

If both members reach 7.5 s without an onset or native guard failure, extend
each member from its own unchanged-model `vpm_001250.h5` to `t=9.0 s` in a
fresh namespace, for example:

```bash
python assets/run_matched_stabilization_pair.py \
  --variant baseline --resume solution/matched_baseline_7p5/vpm_001250.h5 \
  --output-tag matched_baseline_9p0 --endpoint 9.0
```

Repeat with the stabilized source and matching variant. This adds 250 accepted
steps per member, passes the old `7.68 s` failure while remaining below the
`10 s` horizon, and keeps the unchanged physics identity. It remains bounded
diagnostic evidence, not a qualified endpoint. Every trial uses a fresh
namespace, dense `0.024 s` full backups, mandatory accepted-step loading CSVs,
and native guards. No remeshing, clipping, particle deletion or guard
relaxation is permitted; longer survival alone is not qualification.

Run the full case before judging late wake planes, loads, or stationarity.
The short startup check cannot establish stability after 7.5 s.
