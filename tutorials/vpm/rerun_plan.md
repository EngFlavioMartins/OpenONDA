# VPM tutorial rerun, calibration, backup, and certification plan

## Objective and definition of done

Re-run every canonical case under `tutorials/vpm`, calibrate only where the
evidence requires it, generate both PNG and PDF figures without changing the
established plot style, and preserve reproducible backups of the old and newly
accepted results.

This work is complete only when all of the following are true:

- [ ] All 45 canonical simulations listed below have an accounted-for final
      status and every case that is required to complete reaches its configured
      physical end time.
- [ ] The three-level Lamb--Oseen CS grid study passes its existing
      self-convergence gate.
- [ ] Every numerical and physics validator passes. A zero process exit by
      itself is not evidence that a result is stable or useful.
- [ ] Every figure is made from the accepted run that its labels and caption
      describe, contains all expected curves, and uses physically correct
      quantities, signs, units, normalizations, and reference data.
- [ ] The 26 expected figures exist as both PNG and PDF (52 files total), are
      non-empty, and pass visual inspection.
- [ ] A pre-run backup and a final accepted-results backup exist with manifests
      and checksums.
- [ ] All relevant tests pass and the final diff contains no plot-style change.
- [ ] The acceptance ledger records the evidence, configuration, backend,
      commit, validator output, plots, and backup path for every case.

Do **not** call the campaign complete while any required item is failed,
missing, partial, visually misleading, or merely assumed.

## Scope

| Tutorial | Canonical simulations | Expected figures per format |
|---|---:|---|
| `flat_plate` | 20: moving and static at -10, -5, -2, 0, 2, 5, 8, 10, 12, 15 degrees | `plate_polar`, `plate_staticvsmoving`, `plate_spanwise`, `flat_plate_kelvin` |
| `delta_wing` | 1: `delta_wing` | `delta_wing_forces`, `delta_wing_circulation_history` |
| `lamb_oseen_vortex` | 12: vortex, dipole, and merging, each with CS/RWM/DVH/GBD | `vortex_comparison`, `dipole_comparison`, `merging_comparison`, `vortex_surface_fields`, `lamboseen_energy` |
| `vortex_ring` | 4: `dns_direct`, `dns_transposed`, `dns_mixed`, `les_transposed` | `vortex_ring_motion`, `vortex_ring_energy`, `vortex_ring_circulation` |
| `vortex_interactions` | 6: leapfrog and collide, each DNS/LES/stabilized LES | `rings_circulation`, `rings_energy_budget`, `rings_energy`, `rings_conservation`, `rings_resolution`, `rings_stability`, `rings_trajectory` |
| `rotor_flow` | 1: `rotor` | `rotor_performance`, `rotor_wake_planes`, `rotor_loading_validation` |
| `quadcopter` | 1: `quadcopter` | `quadcopter_particle_count`, `quadcopter_vorticity_history` |

The three CS levels run by
`lamb_oseen_vortex/assets/grid_independence_cs.py` are validation runs outside
the 45-case canonical count, but are required evidence for accepting the
Lamb--Oseen spatial resolution.

Coupled FVM--VPM tutorials are outside this plan. Do not alter or remove their
current generated outputs.

## Non-negotiable rules

- Preserve user data. Never run an `allclean.sh` until the directories it will
  remove have been copied to the pre-run backup and that copy has been checked.
- Do not mix samples, checkpoints, logs, or figures from different commits,
  configurations, backends, or calibration trials.
- Do not hide an instability by clipping vortex strength, deleting offending
  samples, truncating a curve, shortening the final physical time, widening a
  plot axis, or weakening a validator.
- Only the interacting-ring DNS and plain-LES baselines may be accepted with
  `status: resolution_lost`, because that early loss of resolution is an
  explicit part of that tutorial's comparison. Every other canonical case,
  including both stabilized interacting-ring cases, must complete its full
  requested horizon.
- Tune the model, time step, resolution, domain, tree accuracy, subcycling, or
  a genuinely conservative stabilization mechanism. Do not change the
  underlying physical problem merely to obtain a pass.
- A higher Smagorinsky coefficient is a hypothesis, not an automatic fix. It
  must improve stability without erasing the physical interaction, corrupting
  force/loading validation, or causing excessive modeled dissipation.
- Do not transfer a calibrated `C_s` from one flow to another without checking
  it independently; the filter width and resolved strain differ by case.
- Do not edit acceptance thresholds after looking at a failed result merely to
  make it pass. Any threshold correction must be supported by an analytic,
  published, symmetry, conservation, convergence, or discretization argument
  recorded in the ledger.
- Plot styling is frozen. Do not change colors, line styles, markers, fonts,
  dimensions, legends, layout, DPI defaults, or the shared theme. It is allowed
  to fix file discovery, renamed columns/files, wrong axes, wrong signs, wrong
  units, wrong normalizations, missing-data handling, and output orchestration.
  A new diagnostic may use the existing shared style, but must not redefine it.
- Plot scripts must fail on missing/incomplete certification data instead of
  silently producing a reference-only or partial figure for the final campaign.
- Keep failed calibration evidence (configuration, log, status, and summary
  metrics). Large rejected checkpoint series may be pruned only after their
  manifest, log, failure state, and checksums have been retained.

## Phase 0 -- establish provenance and back up current state

- [ ] Choose a UTC run identifier such as
      `20260824T120000Z_<short-git-sha>` and create
      `artifacts/vpm_tutorial_runs/<run-id>/`.
- [ ] Record `git rev-parse HEAD`, `git status --short`, branch name, Python
      executable/version, platform, package versions, Taichi version and
      available backends, CPU/GPU information, and relevant environment
      variables in `environment.txt` and `campaign_manifest.json`.
- [ ] Record checksums of `docs/themes/matplotlib_setup.py` and every VPM plot
      script. Compare these at handoff and explain any non-style compatibility
      edit.
- [ ] Copy every existing VPM `solution/`, `samples/`, `figures/`, root-level
      log, and generated geometry needed to reproduce the old state into
      `pre_run/`, preserving paths, timestamps, symlinks, and permissions.
- [ ] Generate `pre_run/SHA256SUMS` and verify it against the copy before any
      cleanup. Record absent source directories as absent rather than creating
      misleading empty backups.
- [ ] Explicitly note the current read-only audit evidence in the ledger:
      existing delta-wing, rotor, and quadcopter logs show the Taichi Metal
      `host_to_device` mapping failure; current Lamb--Oseen data contains only
      the four single-vortex schemes, with DVH and GBD marked incomplete.
      These results are not accepted campaign results.

Recommended artifact layout:

```text
artifacts/vpm_tutorial_runs/<run-id>/
  campaign_manifest.json
  environment.txt
  acceptance.md
  calibration.csv
  pre_run/
  trials/<tutorial>/<trial-id>/
  accepted/<tutorial>/
  logs/
  SHA256SUMS
```

The `artifacts/` tree is ignored by Git and is not removed by tutorial cleanup
scripts. The accepted backup must contain full `solution/`, `samples/`, and
`figures/` trees, not only plots.

## Phase 1 -- repair orchestration and certification contracts before long runs

The repository audit found the following issues. Fix them with narrowly scoped
changes and tests before starting the production campaign:

- [ ] `lamb_oseen_vortex/allrun.sh` passes obsolete `--gamma1/--gamma2`
      options; the parser now requires `--circulation1/--circulation2`.
- [ ] Add `set -euo pipefail` and durable per-subcase log capture to the
      Lamb--Oseen orchestration so a failed scheme cannot be followed by
      partial plotting and an apparent success.
- [ ] `rotor_flow/allrun.sh` validates before plotting even though
      `validate_results.py` requires the figures. Use the order
      simulate -> data/physics validation -> plot -> figure/completeness
      validation, or split the validator into pre-plot and post-plot stages.
- [ ] Invoke the existing flat-plate and vortex-ring validators from their
      final orchestration. They currently exist but are not called.
- [ ] Correct the vortex-ring validator to count only numbered scheduled
      checkpoints, not the separately named final state, and extend it beyond
      only two of the four canonical variants.
- [ ] Add strict certification validators for delta wing and quadcopter.
- [ ] Add a strict Lamb--Oseen completeness/physics validator. The existing
      post-processing manifest intentionally tolerates partial runs and cannot
      certify the final campaign.
- [ ] Ensure every plotting command returns non-zero when a required input is
      missing, malformed, from the wrong configuration, or not at the required
      end/common comparison time.
- [ ] Fix the interacting-ring subset workflow: it currently performs a full
      cleanup and later requires all six plot inputs even when only selected
      cases were requested. Calibration runs should clean/run only their named
      case and should not overwrite full-matrix figures.
- [ ] Make run manifests capture the actual explicit Smagorinsky coefficient,
      time step, horizon, backend, precision, tree/direct method, resolution,
      stabilization settings, output cadence, git SHA, and a configuration
      fingerprint. Add missing fields rather than inferring defaults later.
- [ ] Make all scripts use canonical renamed field/file names. Compatibility
      fixes may change readers and paths, but not the figure style.

Run these gates after the repairs and again at final handoff:

```sh
python -m compileall -q source tests tutorials openonda scripts
ruff check source tests tutorials scripts openonda
ruff format --check source tests tutorials scripts openonda
pytest -q tests/vpm tests/test_tutorial_plot_contracts.py
pytest -q tests
```

Also run `git diff --check`. Do not start long production runs while a relevant
test or static check is failing.

## Phase 2 -- compute-backend preflight

- [ ] Verify the installed package and dependency set using the repository's
      documented install checks (`openonda-verify-install` and
      `python -m pip check`). Do not use `PYTHONPATH` to mask a broken install.
- [ ] Because the existing logs show `AUTO` selecting a failing Metal backend,
      first certify a tiny isolated VPM smoke run with direct Metal access.
      A sandbox-only Metal initialization failure is an execution-environment
      failure, not evidence about the flow physics.
- [ ] The campaign owner explicitly requires GPU execution for every tutorial.
      Therefore use one explicitly recorded `OPENONDA_COMPUTE_DEVICE=METAL`
      backend for all final runs and calibration series, including interacting
      rings. Metal does not provide f64 Taichi kernels; those cases must use
      the explicit GPU-compatible f32 path, record that precision in the
      manifest, and be judged against GPU/f32 diagnostics rather than being
      silently rerun on CPU.
- [ ] If a faster backend is later enabled, repeat a deterministic short case
      on CPU and that backend and compare particle count and core physical
      diagnostics within declared floating-point tolerances before using it.
- [ ] Run `rotor_flow/assets/verify_shedding.py` before any rotor calibration;
      the configured blade must advance leading-edge first and shed from the
      trailing edge at every radial station.

## Phase 3 -- disciplined calibration protocol

Use individual setup commands and unique output labels/directories for trials.
Do not use a destructive full `allrun.sh` during calibration.

For every trial, append one row to `calibration.csv` containing tutorial/case,
trial ID, parent trial, git SHA, configuration fingerprint, backend, resolution,
`dt`, horizon, `C_s`, stabilization settings, completion status, wall time,
particle extrema, stability metrics, physics metrics, validator result, log,
and retained-output path.

Apply changes in this order:

1. [ ] Correct setup errors first: geometry orientation, motion sign, force
       convention, sampling plane, renamed field/file, stale data, output
       mixing, or backend initialization.
2. [ ] Establish a deterministic baseline and reproduce the failure. Save the
       first rejected state and identify the first bad step/time and diagnostic.
3. [ ] Check temporal and spatial resolution, tree/direct accuracy, domain
       bounds, output cadence, particle capacity, and coupled substep limits.
       A model coefficient cannot compensate for an under-resolved or wrongly
       oriented problem.
4. [ ] Calibrate LES with a bracketed, one-variable-at-a-time sweep. Start at
       the existing coefficient, increase in small documented increments, and
       retain the lowest coefficient that satisfies the full stability and
       physics gates. If the failure is over-dissipation, also test downward.
5. [ ] Introduce or tune conservative stabilization only if LES and adequate
       resolution cannot preserve the particle cloud. Its energy transfer and
       invariant corrections must be exported and bounded; no hidden clipping
       or unreported remeshing is acceptable.
6. [ ] Use shortened runs only to reject configurations quickly. A promising
       configuration must be rerun from a clean initial condition through the
       complete canonical horizon and revalidated.
7. [ ] Perform at least one time-step or spatial-resolution sensitivity check
       for every VLM family. Accept a default only if the reported quantities
       change by no more than 5%, or document and fix the unresolved trend.

Do not change a DNS case into LES while continuing to label it DNS. If the
physics model changes, update the setup description, manifest, labels, and
validator consistently while preserving the existing style.

## Phase 4 -- case-by-case execution and physics gates

### 4.1 Flat plate (20 VLM--VPM runs)

Run representative 0, 5, and 8 degree pilots first; 5 degrees exercises the
moving/static and spanwise comparisons, and static 8 degrees exercises the
Kelvin/wake-plane path. After calibration, run the complete two-frame sweep
from clean initial conditions.

- [ ] All 20 force histories are finite and reach at least 23.5 chord lengths.
- [ ] Retain the existing tail-stationarity gate: lift range over the final
      five chords is no more than 0.2% of mean lift (handle the zero-lift case
      with a physically meaningful absolute scale rather than division noise).
- [ ] Kelvin closure for static 8 degrees is at most `1e-4`, as required by
      `assets/validate_results.py`.
- [ ] Fit the low-angle lift slope over `|alpha| <= 10 deg` and compare with
      the existing finite-aspect-ratio Prandtl/lifting-line reference. Require
      no more than 10% relative slope error.
- [ ] Check odd lift symmetry and even/non-negative induced drag. Opposite-angle
      lift magnitudes should agree within 5% of their pair scale; investigate
      sign, force-axis, and geometry errors before tuning.
- [ ] Compare moving and static tail means. Require lift agreement within 2%;
      treat near-zero drag with an absolute reference scale rather than an
      unstable relative error.
- [ ] The 5-degree spanwise loading must have the correct sign, approach zero
      at both physical tips, remain symmetric, and have normalized RMSE no more
      than 10% against the existing lifting-line curve.
- [ ] Confirm the polar, transient, spanwise, and Kelvin figures use the
      intended case files and physical axes; run the strict validator after
      both figure formats are generated.

If these gates fail, calibrate VLM panel resolution, wake core size, time step,
tree accuracy, and then LES. Do not tune plot limits or discard angle cases.

### 4.2 Delta wing (one two-wing VLM--VPM run)

- [ ] Complete all 3,520 steps (`t = 8.8 s`) with finite particles, core radii,
      volumes, per-wing force histories, flow integrals, and the three wake
      sampler planes at 1, 5, and 10 half-spans.
- [ ] Validate from metadata and sampled motion that the wings remain exactly
      pi out of phase and that force rows are assigned to the correct front and
      rear surface after the renaming.
- [ ] Over the final three heave cycles, require successive-cycle mean lift and
      oscillation amplitude to drift by no more than 5%. A periodic response
      may oscillate; exponential or secular growth is not stationarity.
- [ ] Verify the rear-wing force disturbance occurs at physically consistent
      wake-crossing phases rather than being a surface-label or time offset.
- [ ] Require finite, bounded circulation history with no late exponential
      growth. Check combined bound-plus-wake circulation/Kelvin closure if the
      available VLM diagnostics expose it; add the diagnostic rather than
      inferring conservation from `sum(abs(vortex_strength))`.
- [ ] Perform a time-step/panel sensitivity run and require final-cycle force
      means and amplitudes to change by no more than 5%.
- [ ] The strict validator must fail on missing wing curves, missing common
      times, incomplete horizon, non-finite values, or missing figures.

This tutorial has no external absolute-force reference, so describe its
wake-crossing force result as a qualitative/relative validation unless a
traceable reference is added. Do not overstate agreement.

### 4.3 Lamb--Oseen vortex, dipole, and merging pair (12 runs + grid study)

Run each physics family one scheme at a time and certify completion before
moving on. DVH and GBD particle/node guards need active monitoring.

- [ ] All 12 `run_metadata.json` files report `status: complete`,
      `completed: true`, and final time `29.973 s` within one time step.
- [ ] Every case has a complete, strictly increasing sample cadence, finite
      flow integrals, the expected field planes, and a final sample at the
      configured end time. The final plots must compare all schemes at one
      common physical time.
- [ ] For the single vortex, compute relative L2 errors in azimuthal velocity,
      z-vorticity, and velocity gradient against the analytic Lamb--Oseen
      solution using the run metadata. Confirm the velocity/vorticity signs,
      core centre, and the diffusion law `a^2(t) = a0^2 + 4 nu t`.
- [ ] Run
      `python assets/grid_independence_cs.py --compute-device <backend> --require-converged`.
      Require all three levels to use one backend, all successive differences
      to decrease, and each medium-to-fine difference to be at most the
      existing 0.5% tolerance. Preserve its CSV and JSON report.
- [ ] For the counter-rotating dipole, verify equal/opposite circulation,
      symmetric cores, translation in the Biot--Savart direction, physically
      increasing core radius, and no false merger or boundary-limited radius.
- [ ] For the co-rotating pair, verify rotation direction, decreasing
      separation through merger, continuous unwrapped orientation, increasing
      core size, and scheme agreement with the valid Cerretelli--Williamson
      orientation and core-radius curves. Do not compare the retained
      dimensional separation curve until its `nu/b0^2` conversion is supported
      by the primary source.
- [ ] In every viscous case, kinetic energy must not inject and sampled
      `dE/dt` must agree with the solver's viscous `-nu*Omega` budget to a
      declared discretization tolerance. Circulation drift and symmetry errors
      must be reported, not inferred from a visually smooth curve.
- [ ] RWM is one stochastic realization, not an ensemble validation. Record
      its seed and do not claim statistical convergence.
- [ ] The strict certification manifest must reject missing/partial cases and
      distinguish numerical grid independence from model-form agreement with
      the infinite two-dimensional analytic solution.

### 4.4 Single vortex ring (four stretching/turbulence variants)

- [ ] All four canonical variants complete all 600 steps; a
      `resolution_lost` manifest is not acceptable here.
- [ ] Repair/check the checkpoint contract: 24 numbered states at the 25-step
      cadence, plus the intentional initial and final states, with matching
      XDMF descriptors. Do not count `_final.h5` as a numbered checkpoint.
- [ ] No checkpoint or sampler row may contain non-finite values, non-positive
      core radii/volumes, missing groups, or the setup's strength blow-up.
- [ ] Apply the existing Saffman speed gates: relative RMSE at most 10% for
      `dns_transposed` and 12% for `les_transposed`. Report all four variants
      and define equally explicit tolerances for direct and mixed before the
      production run.
- [ ] Retain the existing impulse drift <= 5% and tube-circulation drift <= 8%
      gates, and gate major-radius drift rather than merely printing it.
- [ ] Verify energy decays consistently with viscosity, circulation has the
      correct physical definition (tube circulation, not just
      `sum(abs(vortex_strength))`), and the ring moves in the induced-velocity
      direction.
- [ ] Inspect `ring_modes.csv`: the seeded Widnall band must be represented at
      initialization within the manifest's seed/noise limits, modal evolution
      must remain resolved, and an unstable numerical mode must not be passed
      off as physical Widnall growth.
- [ ] If the LES coefficient is adjusted, require improved resolution/stability
      while retaining the speed, radius, impulse, circulation, and energy
      gates at full horizon.

### 4.5 Interacting vortex rings (six-case DNS/LES/stabilized-LES matrix)

Use the existing `assets/check_run.py` as the minimum acceptance contract; do
not weaken it. Calibrate leapfrog and collision separately because the current
documented coefficients are `C_s = 0.16` and `0.32`, respectively.

- [ ] Confirm all six cases share the same intended initial state within each
      family, use `f64` direct interactions on CPU, and retain the same
      molecular diffusion and discretization across compared models.
- [ ] DNS has zero eddy/stabilization viscosity; plain LES has eddy viscosity
      but no residual viscosity or regularization; stabilized LES activates
      all its declared mechanisms and records their budgets.
- [ ] Only DNS/plain LES may end early at the documented resolution limits,
      with `status: resolution_lost`, a reason, a final diagnostic, and final
      restart state. Both stabilized cases must reach all 1,140 steps.
- [ ] Plain LES must outlive DNS by at least 5% and improve at least two of
      strength growth, divergence, and vortex-line misalignment by 5% over the
      common DNS lifetime.
- [ ] Stabilized LES must improve at least two of those indicators by 10% over
      the common plain-LES lifetime.
- [ ] Enforce every existing absolute gate: monotonically non-injecting energy,
      <= 2% integrated energy-budget error, <= 20% RMS modeled-rate mismatch,
      circulation/linear/angular impulse limits, overlap/divergence/alignment
      limits, and core-spreading/regularization correction limits.
- [ ] Every accepted regularization event must remove rather than inject
      kinetic energy, stay inside the 20% energy and 15% enstrophy dissipation
      limits, preserve declared moments, and record any adaptive core change.
- [ ] If the current `C_s` values fail the comparative gate, run bracketed
      sweeps with identical initial conditions. Keep the lowest coefficient
      that passes the full comparative and absolute physics gates; do not use
      the coefficient that merely runs longest.
- [ ] Validate all scheduled samples/states and all seven figures. Inspect that
      leapfrogging and collision trajectories actually depict the named
      interaction and that plots do not reconstruct data from logs.

### 4.6 Rotor flow (one three-bladed VLM--VPM rotor)

First fix backend startup and orchestration. Preserve rejected states from
divergent trials.

- [ ] `assets/verify_shedding.py` passes at every radial station before the
      first run.
- [ ] Complete all 2,400 steps with the actual `C_s`, time-step, subcycling,
      particle spacing, tree settings, bounds, and VLM resolution recorded in
      the manifest.
- [ ] Preserve the existing admissibility guard: finite particles, positive
      cores/volumes, and maximum particle vortex-strength magnitude <= 10.
- [ ] The wake linear-impulse budget satisfies
      `rho*abs(dI_x/dt)/T = 1 +/- 0.10` over the final two rotations.
- [ ] Every wake plane used by the plot is present and has disc-mean drift <=
      1% over the averaging window. If not, extend the physical horizon rather
      than plotting an in-transit wake as a converged profile.
- [ ] Tail-mean coefficients remain in the existing plausibility ranges
      `0.4 < C_T < 1.1` and `0.2 < C_P < 0.62`, and each differs from the
      matched BEM reference by no more than 25%.
- [ ] Check signs explicitly: thrust opposes/acts consistently with the chosen
      axial convention, power extraction has the expected sign, circulation
      and spanwise loading correspond to trailing-edge shedding, and all three
      blades have equivalent phase-averaged loading.
- [ ] Perform at least one refined VLM/time-step run. Require tail `C_T`, `C_P`,
      and normalized spanwise loading to change by no more than 5% before
      calling the default calibrated.
- [ ] Calibrate LES with the protocol above. A coefficient is accepted only if
      the wake survives the full horizon **and** the impulse, stationarity,
      BEM, loading, and resolution-sensitivity gates pass. Do not use bounding
      removal to conceal a near-rotor instability.
- [ ] Generate plots before the validator's figure-completeness stage and
      inspect that momentum-theory references are evaluated at the simulation's
      actual operating point.

### 4.7 Quadcopter (one four-rotor VLM--VPM run)

The current setup uses DNS and has no strict validator. Add force sampling and
certification data if necessary without changing the existing plot style.

- [ ] Complete all 288 steps/six revolutions with finite particles, positive
      cores/volumes, complete flow-integral history, wake plane, checkpoints,
      and a final restart state.
- [ ] Verify the CW/CCW blade files and angular-velocity signs make each rotor
      advance leading-edge first and that adjacent rotors counter-rotate as
      declared.
- [ ] Record per-rotor/blade forces and moments. Phase-averaged thrust among
      the four symmetric rotors must agree within 5%, and residual vehicle yaw
      moment must be <= 5% of the sum of absolute rotor torques.
- [ ] Over the final two revolutions, require successive-revolution mean thrust,
      particle count, and enstrophy statistics to drift by no more than 5%.
      Confirm that particle-count changes correspond to shedding and declared
      out-of-bounds removal, not silent loss near a rotor.
- [ ] Check force/thrust and climb-direction signs, wake convection direction,
      fourfold/counter-rotating symmetry, finite enstrophy, and a bounded
      circulation history. `sum(abs(vortex_strength))` alone is not a Kelvin
      or force validation.
- [ ] Compare total thrust/power with an independently evaluated rotor/BEM or
      momentum-theory estimate at the actual climb operating point. State the
      assumptions and tolerance before accepting the result.
- [ ] Perform a time-step/panel sensitivity run and require phase-averaged
      thrust/power to change by no more than 5%.
- [ ] If DNS loses resolution, calibrate an explicitly labeled LES model and,
      only if needed, conservative stabilization. Re-run the full six
      revolutions and all physics gates; do not keep the DNS label.
- [ ] The two existing diagnostic plots must cover the full accepted horizon,
      have correct enstrophy units/meaning, and contain no stale data. Add any
      force validation as a separate shared-style diagnostic or validator
      report rather than restyling these figures.

## Phase 5 -- final clean production campaign

After calibration defaults and validators are fixed and tested:

- [ ] Freeze the configuration fingerprints and acceptance thresholds in the
      campaign manifest. No further threshold changes are allowed during the
      final campaign.
- [ ] Back up any last calibration output, then clean only the intended
      tutorial's generated directories.
- [ ] Run the deterministic/analytic cases first, then simple VLM, then the
      difficult long cases. Recommended order:
      Lamb--Oseen -> flat plate -> vortex ring -> delta wing -> interacting
      rings -> rotor -> quadcopter.
- [ ] Capture stdout/stderr for every subcase with timestamps and exit status.
      Monitor diagnostic samples while long runs execute; stop and preserve a
      rejected state on the first non-finite value, invalid core/volume,
      capacity violation, or documented resolution failure.
- [ ] After each simulation, run the data/physics validator before plotting.
      Do not wait until the end of all 45 runs to discover a corrupt cadence.
- [ ] Generate both formats only from accepted data:
      `./plot_all.sh png` and `./plot_all.sh pdf`.
- [ ] Run the figure/completeness validator, then visually inspect every PNG
      (and spot-check PDF rendering) for blank panels, missing cases, duplicate
      or stale curves, wrong time alignment, clipped data, impossible signs,
      incorrect units/normalizations, and misleading reference comparisons.
- [ ] Copy each accepted tutorial's complete generated state immediately to
      `accepted/<tutorial>/` and update checksums before cleaning or proceeding.

If an `allrun.sh` still performs cleanup, remember that invoking it is a
destructive operation against generated results. Use it only after the backup
gate and only when its repaired orchestration is known to be atomic and strict.

## Phase 6 -- final backup, regression checks, and handoff

- [ ] Confirm the accepted backup contains every tutorial's `solution/`,
      `samples/`, `figures/`, manifests, logs, validation output, and calibration
      report. Generate and verify the final top-level `SHA256SUMS`.
- [ ] Confirm exactly 26 expected PNG and 26 expected PDF plot basenames, with
      no zero-byte files and no figure newer than its source samples in a way
      that indicates stale/mixed provenance.
- [ ] Re-run all static checks and the full test suite from Phase 1.
- [ ] Review `git diff` and the stored plot-style checksums. Revert any
      accidental style change while preserving legitimate data-contract fixes.
- [ ] Run every tutorial validator once more against the restored accepted
      directories, proving the backup is usable rather than merely present.
- [ ] Complete `acceptance.md` with one row per canonical case plus the grid
      study. Include: status/end time, stability evidence, physics/reference
      comparison, convergence result, validator command/result, figure list,
      backend/config fingerprint, backup path, and any scientifically important
      caveat.
- [ ] Summarize all code/config changes, final calibrated coefficients, failed
      trial lessons, tests, output locations, backup checksums, and remaining
      limitations. Do not report unresolved validation as success.

## Required acceptance-ledger template

Use this table for every canonical case (expand metrics in linked per-case
reports where needed):

| Case | SHA/config | Backend | Full horizon? | Stable? | Physics/reference gate | Convergence gate | Validator | PNG/PDF | Accepted backup | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| example | `<sha>/<fingerprint>` | CPU | yes | pass | pass: metric + value | pass: delta + value | command + exit 0 | pass | path + checksum | caveat or none |

No cell may be left blank. Use `not applicable` only with a reason. “Looks
reasonable” is not an acceptable physics/reference or convergence result.
