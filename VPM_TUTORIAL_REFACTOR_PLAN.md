# VPM tutorial refactor and publication-figure master plan

Last updated: 2026-08-12 (numerically gated production campaign active)

This file is the persistent source of truth for the requested VPM tutorial work.
It records the complete scope, non-negotiable constraints, completed work,
current execution state, validation criteria, and remaining work so that no goal
is lost while simulations are running.

Status notation:

- `[x]` completed and checked
- `[~]` implemented or running, but still awaiting final validation
- `[ ]` not yet completed

## 1. Non-negotiable scope and scientific constraints

- [x] Apply the refactor and audit to every case under `tutorials/VPM/`, except
  `tutorials/VPM/vortexInteractions/`.
- [x] Treat the tutorial cases as physics examples: expose only inputs that
  distinguish the physics or numerical method, and hide common infrastructure
  choices behind clear defaults.
- [x] Preserve the established Matplotlib publication design. Do not redesign
  the palette, colormaps, fonts, ticks, markers, line styles, labels, or overall
  visual language merely to simplify code.
- [x] Use old results only to verify expected diagnostics, physical quantities,
  historical timing, and figure content. Do not restore, copy, or publish old
  numerical results as current results.
- [ ] Produce all final numerical results with the current solver by rerunning
  the complete Lamb--Oseen matrix from scratch: four viscous schemes (`cs`,
  `rwm`, `dvh`, and `gbd`) for each of the three physics cases (`vortex`,
  `dipole`, and `merging`). The matrix is configured, preflighted, and currently
  running through the numerical acceptance gate.
- [x] Use GPU execution for production VPM runs. CPU execution is not an
  acceptable substitute for generating the requested results.
- [x] Make the Lamb--Oseen execution backend explicitly GPU-only (`VULKAN`) so
  no production case can silently fall back to the CPU.
- [x] Keep processed samples at the immutable case-root location
  `tutorials/VPM/<tutorial>/samples/<case_name>/...`.
- [x] Never place samples inside `solution/`; solution directories contain raw
  solver output and are unsuitable for Git-synced processed data.
- [x] Use the solver's built-in timestepper `sampler`. Do not introduce or use a
  separate `record_diagnostics` mechanism.

## 2. Minimal Lamb--Oseen setup and run interface

- [x] Make `tutorials/VPM/lambOseenVortex/vortex_setup.py` small, readable, and
  aesthetically formatted, with intentional spacing and parameter grouping.
- [x] Limit its command-line interface to the information needed to select the
  physics/method and cadence: case, diffusion scheme, sample period in seconds,
  and raw-backup period in seconds.
- [x] Remove command-line switches for memory fraction, processing unit, backup
  internals, and other infrastructure settings common to every case.
- [x] Convert cadence expressed in physical seconds to timestep counts inside
  Python, so cases with different timesteps retain comparable physical output
  intervals.
- [x] Keep solver configuration close to the corresponding physics, without a
  large forest of conditional setters.
- [x] Make `allrun.sh` the small, readable controller for the complete case
  matrix and the few cadence values shared by the runs.
- [x] Ensure `allrun.sh` controls all twelve Lamb--Oseen cases:
  `vortex`, `dipole`, and `merging`, each with `cs`, `rwm`, `dvh`, and `gbd`.
- [x] Audit whether `cd "$(dirname "$0")"` is genuinely needed. It is retained
  because the shell scripts invoke sibling `allclean.sh`, setup, asset, and plot
  files by relative name; Python resolves its own data paths but cannot change
  the calling shell's working directory. Removing this one line would either
  break invocation from outside the tutorial directory or require several more
  distracting path locators.
- [x] Remove unnecessary environment setters, pass-through flags, repeated
  branches, obsolete helpers, distracting comments, and AI-like explanatory
  prose from the run/setup scripts.
- [x] Preserve genuinely useful progress output and error propagation.
- [x] Keep the tutorial setup fresh-run-only. Do not add checkpoint recovery or
  restore old results through an environment variable or command-line option.
- [x] Use a vortex-column length of 50 initial core radii in all twelve cases,
  matching the historical long-column intent closely enough to suppress the
  severe finite-column error of the discarded 16-radius setup.
- [x] Configure the single-vortex cases for 20 seconds and the translating dipole and
  merging-pair cases for 40 seconds, so the pair-history plots cover their
  intended nondimensional time range.

## 3. Samplers, raw backups, and storage policy

- [x] Verify that all requested diagnostics are generated through built-in
  samplers and written beneath the root `samples/` tree.
- [x] Express sampling and backup periods in seconds in the run controller, with
  conversion to integer timesteps in the Python setup only where required.
- [x] Reduce raw VPM checkpoint frequency enough to save substantial storage
  while retaining enough physical states to diagnose a run and follow its
  evolution.
- [x] Keep processed time histories dense; storage savings must come from sparse
  raw checkpoints, not from discarding inexpensive sampled diagnostics.
- [x] Avoid repeatedly appending large line/surface records where a final
  publication snapshot suffices. Retain scheduled time diagnostics needed for
  trajectories, energies, and evolving physics.
- [x] Sample compact single-vortex histories every 1.0 physical second and pair
  histories every 2.5 seconds and pre-coalescence merger histories every 1.25
  seconds while writing one final raw restart checkpoint. This leaves 15--20
  usable points in every plotted history and avoids making
  expensive flow-integral evaluations dominate the DVH runtime.
- [x] Store a final mid-plane field and final single-vortex line profiles for
  publication comparisons, instead of storing a large field at every sampled
  time.
- [x] Use a compact timestepper sampler for dipole and merging core histories;
  do not reconstruct those dense histories from sparse raw checkpoints.
- [x] Confirm on fresh solver output that every sampler's files, columns, time
  stamps, case names, and units match the plotting code's expectations.
- [x] Give VLM tutorials denser raw output than general VPM tutorials because the
  raw states are needed for animations.
- [x] Give especially dense animation output to the moving delta-wing and rotor
  cases; retain appropriate animation cadence for the quadrotor and flat-plate
  cases as well.
- [x] Re-audit every non-excluded VPM tutorial so its raw-backup cadence reflects
  its purpose: sparse for diagnostics-only VPM cases and denser for VLM motion
  or wake animations.
- [x] Replace vortex-ring motion and circulation histories reconstructed from
  sparse raw HDF5 checkpoints with a compact built-in-timestepper sampler CSV at
  `samples/<variant>/ring_diagnostics.csv`; keep HDF5 cadence sparse.

## 4. Samples and Git history

- [x] Change `.gitignore` so processed `samples/` directories are eligible for
  Git synchronization.
- [x] Continue excluding raw checkpoints and other large solver formats from
  Git, even if they occur under a sample-related path.
- [x] Keep raw solution output out of version control.
- [x] Audit the currently syncable sample files for individual file size and
  total footprint. The present compact tracked sample payload is about 18.8 MiB;
  its largest file is 9.39 MiB and remains below the 10 MiB pre-commit limit.
- [x] Keep the repository's large-file check compatible with the existing
  approximately 9.4 MiB reference dataset while preventing newly generated
  oversized files from entering ordinary Git history.
- [ ] Repeat the size audit after all fresh runs finish and before declaring the
  result set ready to sync.
- [x] Distinguish root processed samples from legacy nested
  `solution/**/samples` data. No nested legacy sample is tracked; those old
  ignored trees are not a result source and will disappear during fresh cleans.
- [x] Confirm that no generated raw `.h5`, `.hdf5`, `.xdmf`, or `.npz`
  sequence, or equivalent checkpoint has become trackable accidentally.
  Processed VTS/PVD fields remain trackable except for the explicitly ignored
  636 MiB cube reference-flow volume sequence; its compact CSV histories remain
  syncable.
- [x] Stop the invalid short-column process. Its generated sample and solution
  files are not final results and will be removed by the clean start of the
  valid campaign; nothing from that run will be recovered.

## 5. Plot discovery and incremental plotting behavior

- [x] Make plotting scripts read processed data from
  `tutorials/VPM/lambOseenVortex/samples/<case_name>/`, never from a solution
  directory or a nested samples directory.
- [x] Keep output figure names and scientific content compatible with the
  previous publication workflow where the underlying current-solver diagnostic
  still exists.
- [x] Make `allplot.sh` call every plot family on every invocation, without
  waiting for the full twelve-case run matrix to finish.
- [x] Render each figure as soon as any scientifically usable current result is
  available. Missing future cases must not block already available figures.
- [x] Refresh every currently possible figure automatically after each case
  completes in future `allrun.sh` invocations; `allplot.sh` also remains safe to
  call manually at any point during a case.
- [x] Refresh a figure with the most current available samples whenever
  `allplot.sh` is called.
- [x] For instantaneous cross-method profile and surface comparisons, use the
  latest physical time common to the currently available methods so that the
  panels remain scientifically comparable while runs are in progress.
- [x] For time-history figures, draw each available method through its own most
  recent sample; naturally incomplete curves are acceptable during a run.
- [x] Clearly omit methods that have no fresh current-solver data yet, rather
  than recovering old output or silently mixing incompatible times.
- [x] Do not treat a partial result set as a plotting error. A plot command
  should fail only for a genuine parsing, numerical, or rendering failure.

## 6. Complete Lamb--Oseen publication figure set

- [x] Restore the velocity-profile comparison for the Lamb--Oseen vortex.
- [x] Restore the vorticity-profile comparison.
- [x] Restore the radial vorticity-gradient comparison.
- [x] Restore velocity and vorticity surface/contour fields for each available
  diffusion method.
- [x] Restore the dipole vortex-core trajectory over time.
- [x] Restore the dipole core-radius evolution.
- [x] Restore the merging-vortex angle, core-radius, and separation histories,
  including the intended reference comparison where current diagnostics allow.
- [x] Restore energy/circulation-style diagnostic panels across vortex, dipole,
  and merging cases where those current solver diagnostics are available.
- [x] Fix the dipole plot's sample-path lookup so trajectory data can actually
  be found under the new root sample structure.
- [x] Plot every sampled time point in time histories instead of visually
  thinning markers with `markevery`.
- [x] Ensure surface comparisons include every method that currently has a
  valid fresh field, including an in-progress method's most recent comparable
  field.
- [ ] Confirm that all four methods appear in every applicable final comparison
  after all twelve runs complete.
- [~] Record any old figure quantity that the current solver no longer produces,
  and document the scientifically closest current replacement rather than
  fabricating or recovering a result.

## 7. Publication sizing and visual quality

- [x] Set every requested figure to exactly 12.5 cm physical width at save time.
- [x] Avoid `bbox_inches="tight"` or other cropping that changes the requested
  physical output dimensions.
- [x] Verify generated PDF media boxes are 354.331 points wide (12.5 cm).
- [x] Keep subplot gaps, outer margins, legend clearance, shared labels, and
  colorbar spacing intentional rather than relying on unstable automatic
  cropping.
- [x] Make time-series plots visually dense by showing the available samples,
  without increasing expensive raw backup frequency merely for appearance.
- [x] Keep labels readable and prevent awkwardly large or cramped spaces between
  subplots, tick labels, axes labels, legends, and colorbars.
- [x] Preserve the user's Matplotlib theme and established design choices.
- [x] Visually inspect every incrementally available PNG after the latest-data
  plotting changes.
- [ ] Visually inspect the complete final PNG/PDF set after all runs finish,
  checking every panel, curve, method, label, legend, contour, and margin.

## 8. Audit of all other VPM tutorials (except vortexInteractions)

- [x] Apply the root-samples rule to the other VPM tutorials in scope.
- [x] Refactor plot readers that still expected processed data in a solution
  directory.
- [x] Repair the remaining generated ParaView sample readers so their PVD input
  comes from each tutorial's root `samples/<case>/` directory rather than an old
  absolute `solution/**/samples` path.
- [x] Simplify Python setup aesthetics and remove distracting configuration
  plumbing where it does not distinguish the physics.
- [x] Make Vulkan GPU selection a fixed common setup value in delta-wing,
  flat-plate, quadcopter, rotor, vortex-ring, and Lamb--Oseen tutorials.
- [x] Reduce the rotor setup interface to sample and backup cadence only; keep
  coupled-integration, treecode, guard, and resolution values as readable
  physical/numerical constants rather than hidden parser defaults.
- [x] Reduce each flat-plate invocation to mode, angle of attack, sample period,
  and backup period; derive redundant case names, frames, kinematics, and the
  one wake-plane diagnostic consistently in Python.
- [x] Reduce each vortex-ring invocation to variant name, sample period, and
  backup period; derive DNS/LES and stretching from the named case.
- [x] Simplify `allrun.sh`, `allplot.sh`, and adjacent helper scripts where
  repeated setters, arguments, or directory assumptions obscure the example.
- [x] Remove hallucinated, redundant, stale, and strangely conversational
  comments while retaining comments that explain actual physics or a necessary
  numerical choice.
- [x] Re-scan every in-scope setup, run script, plotting script, sample path, and
  backup cadence after the Lamb--Oseen final fixes, so no inconsistent pattern
  remains.
- [x] Run final syntax/style/shell checks on all files changed by this audit.

## 9. Fresh simulation matrix and current execution state

- [x] Stop and discard the incomplete high-density `vortex_cs` attempt; it
  exceeded the 20--30 minute target and is not a source of current results.
- [x] Complete short fresh GPU preflights with the current solver and verify
  that Taichi reports the Vulkan backend.
- [x] Start the clean production campaign only after the short tests and
  numerical acceptance gates passed.
- [x] Complete fresh `vortex_cs` output.
- [x] Complete fresh `vortex_rwm` output.
- [x] Complete fresh `vortex_dvh` output.
- [x] Complete fresh `vortex_gbd` output.
- [x] Complete fresh `dipole_cs` output.
- [x] Complete fresh `dipole_rwm` output with the corrected 0.20 s timestep;
  it passes circulation, monotone translation, symmetry, and core-growth gates.
- [~] Complete fresh `dipole_dvh` output. Two 20,000-particle candidates and a
  coarse-lattice timing trial were rejected from the result set; the accepted
  0.06 m-resolution, 22,000-particle Vulkan run is active with 2.5 s diagnostics
  and only a final raw checkpoint.
- [ ] Complete fresh `dipole_gbd` output.
- [ ] Complete fresh `merging_cs` output.
- [ ] Complete fresh `merging_rwm` output.
- [ ] Complete fresh `merging_dvh` output.
- [ ] Complete fresh `merging_gbd` output.
- [ ] Verify every completed case has its expected final raw checkpoint cadence
  and processed sampler files, without duplicated or stale old-run records.
- [ ] Confirm the run controller exits successfully only after the entire fresh
  matrix is complete.
- [x] Make surface-sampler PVD indexes restart-safe by loading existing entries,
  replacing an already-present step instead of duplicating it, and preserving
  chronological order.

### Other explicitly requested production runs

- [x] Keep obsolete rVPM/`LES_rvpm` variants removed. They do not exist in the
  current campaign and must not be recreated.
- [ ] Leave all other current tutorial cases and their run definitions intact;
  defer their production runs while the complete Lamb--Oseen matrix is the
  active focus.
- [ ] When VLM production runs resume, retain denser raw backups for animation,
  especially for the moving delta wing and rotor.
- [ ] Do not run, clean, refactor, or regenerate
  `tutorials/VPM/vortexInteractions/`; preserve any unrelated existing work in
  that excluded directory.

## 10. Validation and completion gates

- [x] Run Python syntax compilation on the edited plotting/setup modules during
  the refactor.
- [x] Run Ruff checks/format validation on the edited Python files during the
  refactor, fixing issues introduced by these changes.
- [x] Run shell syntax checks on edited `allrun.sh` and `allplot.sh` scripts.
- [x] Compare generated publication PDF widths against the exact 12.5 cm target.
- [x] Run the sampler, Lamb--Oseen pair-diagnostic, plotting, and vortex-ring
  regression tests; all 25 selected tests pass with the compact root-sample
  histories.
- [x] Re-run `allplot.sh` immediately after incremental plotting is repaired and
  inspect every figure available from the current CS/RWM/DVH data.
- [x] Re-run all relevant syntax, lint, focused tests, diff-whitespace, and shell
  checks after the last edit.
- [ ] Run the final all-figure plotting command after the twelve fresh solver
  cases finish.
- [ ] Verify final figures use only fresh current-solver sample files.
- [ ] Verify every applicable figure contains all four methods and sufficiently
  dense data.
- [ ] Verify the vortex trajectory, vortex profiles, velocity/vorticity fields,
  merging histories, and energy diagnostics are all present when supported.
- [ ] Verify no output figure has clipped labels, overlapping annotations,
  inconsistent whitespace, empty panels caused by a path bug, or altered theme.
- [ ] Repeat the Git size/ignore audit on the final processed dataset.
- [ ] Provide a concise final handoff listing the completed fresh cases, figure
  files, validations, any unavoidable current-solver diagnostic gaps, and the
  exact location of the synced processed samples.

## 11. Completed reduced-particle preflight

- [x] Correct the persistent scope to the full 4-scheme by 3-physics matrix and
  mark every case pending because the earlier 16-radius results are invalid.
- [x] Reduce the 50-radius initial populations to 1,197 particles for the
  single vortex and 2,520 for either pair by using 0.06 m in-plane spacing and
  0.10 m axial quadrature spacing. Keep final publication field sampling at the
  denser 0.0375 m spacing.
- [x] Disable vortex stretching for these deliberately pseudo-two-dimensional
  columns, and skip the corresponding unused velocity-gradient evaluation in
  the solver.
- [x] Limit regenerated particles independently by method: 9,000 for GBD and
  20,000 for DVH. A 10,000-node DVH limit was rejected because it retained only
  89.2% of the heat-kernel circulation magnitude after three short steps.
- [x] Make the regeneration cap group-aware and propagate vortex group IDs to
  newly occupied diffusion nodes, preventing one vortex of a symmetric pair
  from being preferentially removed.
- [x] Validate the final 9,000-particle GBD configuration on Vulkan with a
  two-vortex run. The sampled core stayed at y=+0.5 with circulation 1.020,
  the solver retained 99.969%, 99.886%, and 99.738% of candidate circulation
  magnitude in the first three steps, and steady steps cost 2.3--2.9 seconds.
  At 667 steps, with production diagnostics four times less frequent than the
  timing test, this provides margin at the 30-minute full-pair target.
- [x] Restrict nearest-group propagation to actual surviving diffusion nodes
  instead of traversing the full 2.8-million-node grid every step.
- [x] Use a 0.30 s GBD timestep, below its reported 0.318 s diffusion limit.
  The larger valid step avoids hundreds of unnecessary remesh cycles and gives
  133 steps for a 40 s pair case.
- [x] Validate DVH on Vulkan at 20,000 particles: its first three cap-retention
  values were 99.999%, 99.877%, and 99.261%; both core centroids remained at
  y=+0.5 and y=-0.5; per-vortex circulation remained 0.982 after 0.993 s; and
  its 121-step 40 s run projects comfortably below 20 minutes.
- [x] Repair the final CSV sampler dispatcher after a real short run exposed
  that legacy line samplers do not accept the pair sampler's `step` keyword.
- [x] Confirm on fresh Vulkan output that the single-vortex run writes both
  line-profile CSVs and a final VTS containing velocity, vorticity, strain, and
  velocity-gradient fields; confirm pair runs write dense trajectory CSVs and
  final surface fields.
- [x] Run the focused lint, shell syntax, Python compilation, and 54-test VPM
  regression suite after the performance and sampler fixes.
- [x] Exercise the new compact pair sampler and pair plotting families on
  controlled data, including safe reads while a CSV is still being appended.
- [x] Confirm that both pair figures render all four methods on a 12.5 cm-wide
  canvas with the established publication style and intentional spacing.
- [x] Confirm that the run controller enumerates exactly the twelve intended
  cases, cleans stale output once, and refreshes all currently possible figures
  after every completed case.
- [~] Run the clean twelve-case Vulkan campaign, monitor it, and refresh and
  inspect figures as methods become available.

## 12. Allrun interruption and single-format plotting

- [x] Add `-png` and `-pdf` selectors to Lamb--Oseen `allplot.sh`; each call now
  renders exactly one format, with PNG as the backward-compatible default.
- [x] Make `allrun.sh` request PNG explicitly after each completed solver case,
  avoiding duplicate publication rendering during the simulation matrix.
- [x] Identify the interrupted-run cause from the fresh output state. The
  `vortex_cs` solver reached step 400 and wrote its final samples successfully;
  the following incremental plot failed with `KeyError: 'Uy'`, and `set -e`
  correctly stopped `allrun.sh` before `vortex_rwm` began.
- [x] Repair the vortex-profile reader for the current final-sampler format,
  where physical time is stored in a leading `# flow_time=...` line and the
  next line is the field header.
- [x] Verify incremental plotting with only the completed CS case. Both
  `allplot.sh -png` and `allplot.sh -pdf` exit successfully, missing future
  schemes are skipped, and a PDF-only call leaves existing PNG modification
  times unchanged.
- [x] Add a regression test for final profile CSV parsing and normalize sampler
  CSV output to LF endings so freshly Git-synced samples pass whitespace checks.
- [x] Preserve the completed fresh `vortex_cs` result while diagnosing the
  failure; do not clean or restart the production matrix during this repair.

## 13. Current samples-only plotting

- [x] Remove support for older time-column vortex profiles. A profile is valid
  only when it uses the current final-sampler metadata line and field header in
  `samples/vortex_<scheme>/vortex_<scheme>_x.csv`.
- [x] Remove manual time-step selection and nearest-time matching. Final line
  and surface samplers now plot their latest saved result directly; trajectory
  and flow-integral histories plot every valid row through their latest time.
- [x] Derive history-plot time limits from the latest sampled row, so an
  interrupted or extended run is neither shown with a mostly empty axis nor
  clipped at an earlier hard-coded duration.
- [x] Keep all plot-data discovery below the tutorial's root `samples/`
  directory, with one subdirectory per physics/scheme case.

## 14. Reference visibility and physics-scale audit

- [x] Render analytical and experimental reference curves above numerical
  markers in every applicable comparison.
- [x] Replace clipped or overly broad plot windows with fixed limits chosen
  from the complete twelve-case sample ranges; preserve the established theme.
- [x] Quantify and reject the earlier mismatched single-vortex end states rather
  than attributing them to plotting. The replacement configuration now reaches
  the expected 20 s diffusion scale for all four schemes.
- [x] Correct the numerical causes of those rejected states: average independent
  replicated RWM realizations, preserve the proposed diffused second moment in
  grid regeneration, and raise only the isolated-vortex DVH/GBD caps.

## 15. Numerical correctness gate and accepted production configuration

- [x] Add a samples-only validator that makes `allrun.sh` stop immediately when
  a completed case is incomplete, non-finite, too sparse, outside its physical
  circulation budget, above the 30-minute limit, or (for the single vortex)
  outside the analytic profile tolerances.
- [x] Require single-vortex relative L2 errors below 10% for velocity and 15%
  for vorticity over `|x| <= 5.5 a_c,0`.
- [x] Treat the isolated Lamb--Oseen vortex as an Eulerian-stationary viscous
  benchmark by disabling particle self-advection. Pair cases retain RK3
  advection because translation and rotation are part of their physics.
- [x] Correct capped grid regeneration so removed circulation is first assigned
  locally, then corrected to preserve total circulation, linear impulse, and
  the diffused angular impulse. Do not project the physical second-moment
  growth back to its pre-diffusion value.
- [x] Raise only the isolated-vortex DVH/GBD caps to 30,000. Keep the cheaper
  pair caps at 22,000 for DVH and 9,000 for GBD.
- [x] Validate the accepted full single-vortex endpoints: CS 1.7%/0.9%, RWM
  4.2%/10.1%, DVH 5.3%/6.3%, and GBD 5.0%/7.0% velocity/vorticity error.
- [x] Use four independent RWM seeds with 16 divided-strength replicas per
  lattice particle, averaging only the final profile and surface fields. Keep
  the primary realization's dense integral history and raw checkpoints.
- [x] Reject exact-position divided-strength replicas for advecting RWM pairs.
  A four-replica timing test projected beyond 30 min, while a two-replica run
  halved the initialized per-vortex circulation and failed every physics gate.
  Pair noise must be reduced across independent valid realizations instead.
- [x] Avoid evaluating self-induced particle velocity on frozen-advection
  steps unless a diagnostic consumes it; when due, evaluate it after diffusion
  so the energy diagnostic describes the end-of-step state.
- [x] Validate 3 s advecting GBD preflights for both pair types. The dipole
  reached `x=0.402` with per-vortex circulation 0.991; the merging pair retained
  total surface circulation 1.982 and symmetric, finite geometry.
- [x] Pass focused VPM regression tests, Ruff, Python compilation, shell syntax,
  and diff-whitespace checks before starting production.
- [~] Run all twelve cases from a clean samples/solution state on Vulkan. Each
  completed case refreshes PNG figures immediately and must pass the validator
  before the next case begins.
- [x] Extend the production gate beyond file completion: a dipole must
  translate monotonically, remain on its symmetry plane, and show sustained
  diffusive core growth; a merging pair must rotate, grow, and approach
  coalescence.
- [x] Reject and delete the first full/partial pair candidates instead of
  publishing them: the old RWM pair used a coarser 0.05 s configuration with
  excessive stochastic spread, and the first capped DVH pair stopped growing
  after about 10 s despite conserving total circulation.
- [x] Preserve capped DVH/GBD moments separately for each vortex group. The
  pure-NumPy regression verifies circulation, linear impulse, and angular
  impulse for both opposite-signed groups; the fresh 10 s Vulkan DVH preflight
  grew monotonically and stayed within 7% of its isolated diffusive-radius
  scale while preserving per-vortex circulation to 0.1%.
- [x] Reject a 0.09 m pair-regeneration lattice after its first sampled core was
  visibly over-diffused. Keep the 0.06 m lattice that passed the isolated-vortex
  analytic field gate, increase the pair DVH cap only to 22,000, and reduce the
  expensive pair diagnostic frequency instead of coarsening the physics.
# Current execution blocker (2026-08-12)

- [ ] **NOT RELEASE-READY:** the final eight pair cases still require fresh runs
  after fixing pair diagnostics to use a mid-plane slab and enlarging the
  diffusion domain for the analytically predicted final core size.
- [ ] Delete the rejected partial `dipole_gbd` sample/checkpoint, then rerun it
  with the final 0.075 m grid and 25,000-node cap. The 20,000-node trial matched
  early diffusion but plateaued late and was correctly rejected.
- [ ] Rerun the other seven pair cases from clean current-solver state, validate
  all twelve cases, render all five figures, visually inspect every panel, and
  audit Git-synced sample sizes.
- [x] Pair sampling now includes the exact built-in-sampler t=0 state and uses a
  z=0 slab for geometry while retaining full-column circulation.
- [x] Validation now rejects late-time dipole core sizes more than 35% from the
  analytical diffusive scale, in addition to circulation, symmetry, completion,
  monotonicity, finiteness, and runtime gates.
- [x] RWM pair samples use four full independent GPU realizations; same-position
  replicas were rejected because they corrupted group diagnostics.
- [ ] Execution paused only because the Codex workspace-operation allowance was
  exhausted at 2026-08-12 07:xx; do not report completion on resume until every
  unchecked gate above passes.
