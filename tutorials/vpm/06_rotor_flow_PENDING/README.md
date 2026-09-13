# Wind-turbine wake

Production status: **unqualified**. The September 2026 VLM audit found concrete
backup/sampling defects and repaired them, but the retained rotor run still
fails its timestep health check and does not establish converged induction.
See [the production audit](../../../docs/development/vlm_production_audit.md)
for the measured blockers and test evidence.
The [implementation follow-up](../../../docs/development/vlm_production_repair_results.md)
records the repaired row closure, kernel precision, observer performance,
CPU/Metal agreement and downstream-wake diagnosis. These repairs do not yet
qualify the full rotor's stability, CT/CP or developed induction fields.

The setup now also samples four streamwise lines at design `r/R = 0, 0.25,
0.65, 1.1`, from `x/D = -1` to `3`, every `0.06 s`. Their native
`streamwise_r*.csv` files retain `time`, `step`, positions, and all three signed
velocity components. The output owner appends these records across accepted
steps and restarts. An older single-snapshot CSV cannot be resumed into this
schema; use a fresh output namespace to preserve the original data.

`assets/plot_rotor_streamwise.py` plots axial deficit `1 - ux/U` and signed
`uy/U`, `uz/U` versus `x/D`, using an exactly bracketed five-revolution time
mean. It refuses missing or incomplete histories. `allplot.sh` includes this
figure; old retained results without the new lines will need a fresh run.

From this directory, run `./allrun.sh` (which uses the canonical fresh
`completion` namespace) and then `./allplot.sh`; use `./allplot.sh pdf` for
vector figures. To choose another preserved namespace, invoke
`python setup.py --output-tag <name>` and set `ROTOR_OUTPUT_TAG=<name>` when
plotting. Direct `python setup.py` runs require `--output-tag` when an existing
native result is present. The installed OpenONDA package supplies the solver
and plotting dependencies.

The ordinary fresh run also declares a restartable lifecycle guard: a 570000-particle
soft ceiling, 12 GiB process-RSS ceiling and 2 GiB available-memory floor, below the
separate 600000-particle numerical container capacity.

The ordinary `setup.py` owns fresh runs only. An explicit, bounded smaller-step
restart pilot is available for a validated native checkpoint:

```bash
python assets/run_restart_pilot.py \
  --resume solution/vpm_001152.h5 \
  --resume-dt 0.001 \
  --attempt pilot
```

The pilot performs an eight-step sampler-free preflight and then at most 120 accepted
steps, with wall-time, particle-count, RSS and available-memory bounds. It recomputes
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

The authored 7.5 s horizon samples force/loading histories every accepted step
(0.006 s at the authored step) and compact fields every 0.06 s. Coupled restart
backups occur every four accepted steps (`0.024 s`), giving 312 scheduled
frames plus the final state over the authored horizon. At full speed, adjacent
backups differ by about 11.23 degrees, providing approximately 32 states per
revolution for smooth 30 fps playback. Based on the retained native
checkpoints (17,280 added particles per 128 accepted steps), a linear size fit
to HDF5 checkpoints is approximately `47.1 bytes * particles + 128 KiB`:
scaling the previous 250-frame estimate to 313 frames gives about 1.2 GiB HDF5,
18.3 MiB VLM companions and 0.9 MiB XDMF indexes, excluding logs and metadata.
This estimate assumes the old particle-growth history; the qualified rerun's
timestep and horizon will determine its actual storage cost. It is not a
wall-clock write-throughput measurement: no retained native record times the
backup writes, so an elapsed-I/O forecast is unavailable and must be measured
in a bounded future pilot if it is needed. The dense policy is therefore an
explicit write/storage cost, not a free visualization setting. The mandatory
CSV histories remain dense independently.

For a bounded restart diagnostic from `solution/vpm_000896.h5` (`t=5.376 s`,
`120,960` active particles) to the `t=7.5 s` planning endpoint, the measured
`135` new particles per accepted step make the storage cost depend on `dt`:

| requested `dt` | accepted steps | added particles | final particles | full backups* | serialized budget |
|---:|---:|---:|---:|---:|---:|
| `0.006 s` | 354 | 47,790 | 168,750 | 71 | ~0.465 GiB |
| `0.003 s` | 708 | 95,580 | 216,540 | 71 | ~0.541 GiB |
| `0.001 s` | 2,124 | 286,740 | 407,700 | 71 | ~0.844 GiB |

\* Seventy periodic backups plus one final endpoint backup; the table excludes
the source checkpoint. These are serialized bytes per simulated horizon, not
elapsed I/O. Wall-clock write time remains unforecast until a permitted pilot
records it.
The horizon is bounded legacy planning evidence, not a qualified healthy endpoint.
`python assets/validate_results.py --pre-plot` checks
completion, force/power stationarity, BEM agreement, coupled bound-plus-wake
impulse, complete mean-field windows and reports finite-distance axial/azimuthal
induction errors at both required stations. It also reports conservative
particle-front brackets from native numerical checkpoints when available; these
brackets are evidence only and do not certify physical convected-wake arrival.

The latter uses the analytical right-vortex-cylinder equations of [Li et al. (2025), Eqs. 30–45](https://wes.copernicus.org/articles/10/2515/2025/) and the planar superposition/closure described by [Li et al. (2022), Sect. 3](https://wes.copernicus.org/articles/7/75/2022/). It uses the matched BEM bound-circulation table to form piecewise-constant cylinder sheets; it does not fit a curve to the VPM samples. The predeclared scaled-RMS screens are 25% for axial induction and 35% for azimuthal induction, with reference floors of 5% and 2% of the relevant velocity scale. These screens are explicitly diagnostic flags, not uncertainty-based acceptance gates: no defensible uncertainty budget is claimed for the infinite-blade, inviscid, non-expanding reference against the finite-blade LES wake. The report includes dimensional RMS errors and finite-bin coverage so weak or zero swirl cannot be hidden by a floor or profile average. A flag keeps theory agreement unqualified; do not tune the screens or silently replace the model with the ideal far-wake curve.

`allplot.sh` additionally writes `figures/rotor_induction_validation.{png,pdf}`, showing native and finite-distance reference axial and azimuthal induction profiles at 1D/2D. It also writes `assets/animation/rotor_30fps.gif` from the recorded coupled VPM+VLM backups and nearest recorded 1D wake fields. The sidecar JSON records the exact source backups and wake-plane files. The GIF title reports accepted physical time and its accelerated playback multiplier; no solver state is interpolated or fabricated. The authored 7.5 s horizon precedes the older run's t=7.68 s CFL stop but is not a selected healthy endpoint or before-failure proof: report any missing signal onset, late growth or stationary window rather than relabeling startup as steady validation.

The old generated GIF has been removed; a new dense animation awaits the
qualified rerun. The retained rotor data is failure-diagnosis evidence. Its
nine native backups and VLM companions share the same clock; unpaired legacy
surface frames and old VLM exports in samples have been deleted. Cleaning the
output files does not resolve the observed late vorticity growth.

This is an inviscid blade-loading model coupled to a viscous/LES particle wake. It does not resolve blade boundary layers, transition, stall or airfoil profile drag. BEM agreement validates the corresponding attached-flow approximation, not every effect in a high-Reynolds-number turbine.

Reference: [CCBlade theory](https://wisdem.readthedocs.io/en/master/wisdem/ccblade/theory.html), citing Ning's bracketed inflow-angle method, Prandtl losses and Buhl's correction. The shared reference implementation lives in `openonda.rotor_theory`; independent momentum/scaling tests accompany it.

The LES control uses Smagorinsky Cs=.17, CS diffusion, Gaussian particles and
wake-core overlap2.5. The native LES filter is `Delta=particle_volume^(1/3)`,
distinct from the Gaussian core radius. The VPM stabilization study's
vortex-ring settings cannot be transferred without checking these distinct
scales. CS is the prescribed diffusion method for this rotor qualification.
Keep the LES coefficient and core overlap fixed when isolating other changes.
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
  --variant stretching_viscosity --output-tag matched_stabilized_7p5 \
  --endpoint 7.5
```

Both fresh members use the same `dt`, rotor setup, owner cadence, CPU runtime
override and native health/resource guards. The baseline has stabilization
disabled; the matched diagnostic sets only the existing local stretching
viscosity coefficient `C=0.5`. This is a paired model comparison, not a
continuation claim. Each member costs the full approximately 1.05 GB serialized
7.5 s output budget and its own bounded compute; the pair therefore plans for
approximately 2.1 GB of serialized output and at least twice the baseline
accepted-step work, with additional stretching-kernel wall time unmeasured.

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

The saved older run is incomplete and uses preceding solver settings. It is not
qualification evidence for the corrected setup. Performance/loading figures can
be made before completion, but late wake-plane figures require the corresponding
native samples. A completed long run with the corrected source is still needed
to assess stability and developed turbine performance.
