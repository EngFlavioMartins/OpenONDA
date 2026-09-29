# Original-request completion ledger

This ledger covers the complete conversation, including requests predating the
archive work. An implemented fix, a regression test, a real tutorial execution,
and a completed scientific study are different kinds of evidence. Unchecked
items must not be described as complete.

## Solver portability and the reported failures

- [x] Canonical and legacy VPM backup discovery, numeric ordering, and RWM member reuse implemented.
- [ ] Repeat the original completed Lamb–Oseen CS/DVH/GBD/RWM continuation paths with the installed final package.
- [x] Restore native FVM, VPM and coupled state from latest backups; initial mode ignores prior state; launchers distinguish cleanup from continuation.
- [x] Repeat real coupled initial/latest/completed-run execution in serial and on two MPI ranks, exercising both native solver states and sampled output clocks. Standalone installed-package verification is also repeated below.
- [x] Remove stale particle-capacity values from restart identity where they do not define the numerical model.
- [x] Exercise the original DVH/treecode allocation failure on the actual CUDA device and check repeated evaluation memory.
- [x] Implement transactional subdivision for excessive coupled wall corrections, preserving the accepted macro step.
- [x] Exercise excessive wall correction using native particle motion, the real RK2 integrator and a rotated wall; subdivision preserves circulation, one accepted clock and projection diagnostics.
- [x] Correct ParaView view/layout screenshot targets; include the leapfrogging reference table; correct thesis-layout clearance.
- [ ] Finish real plotting executions that exercise each reported plotting failure.

## Minimal configuration and readable tutorials

- [x] Remove remaining public GPU-memory, diffusion-node and remeshing soft-capacity controls and associated truncation paths.
- [x] Verify old backups remain readable after retiring operational settings; reject genuinely incompatible physical state.
- [x] Audit all tracked tutorial setup files and comments for redundant constants, machine assumptions, incident-specific prose, and unnecessary orchestration.
- [x] Remove avoidable nonphysical setup arguments and package-bootstrap clutter without breaking editable local assets.
- [ ] Confirm particle splitting handles core growth/strength conservatively and qualify rotor stability past the reported 7.5 s failure.
- [ ] Investigate quadcopter strain failure; preserve intentionally unstable comparison cases and label their health stops accurately.

## Cylinder geometry, accuracy and performance

- [x] Preserve Re=150 and implement resolved three-dimensional span configurations.
- [x] Shared solid geometry excludes injected particles and constrains shallow wall crossing; test curved, rotated, thin and concave surfaces.
- [x] Implement consistent slip-span induction/diffusion and physical interpolation on anisotropic donors.
- [x] Implement measured trace/interpolation reuse, image accumulation and PETSc workspace optimizations; retain accuracy/iteration limits.
- [x] Record short transient coupling sensitivity and exact optimization evidence in the cylinder execution report.
- [ ] Complete independent span/z, domain, grid and temporal qualification and select coupled/reference production meshes.
- [ ] Complete injection/renewal cadence, transfer support, particle-spacing and exchange-time sensitivity with uncertainty and paired physics.
- [ ] Establish developed-wake parallel timings and a conservative finest-case runtime within 12 hours on this machine.
- [ ] Verify ordinary cylinder launchers produce the requested comparisons and grid-independence report.

The detailed numerical obligations remain in
[the original cylinder plan](../../studies/cylinder_3d_accuracy_performance_plan.md),
The [current evidence table](../../studies/cylinder_current_evidence.md) records
measured mesh/rank costs and the running reference timing. Requirements include
conservation, spanwise diagnostics, statistical windows and rejected
candidates. They are not replaced by this shorter ledger.

## Data, every plot, installation and delivery

- [x] Inventory large/numerous samples and preserve approved current data losslessly; leave unreferenced superseded histories local.
- [x] Commit nine checksum manifests and LFS archives; verify automatic local-clone hydration and every archive hash.
- [x] Measure temporal reconstruction/spectral sensitivity for all archived delta/rotor panel-loading streams and aggregate forces; report qualified/rejected candidates without deleting data. Evidence: [cadence study](../../studies/loading_cadence_report.md). Final-run cadence certification remains dependent on completed histories.
- [ ] Ensure every expected tutorial plot/scene/animation has genuine portable inputs and execute every allplot launcher from a clone.
- [ ] Visually inspect all figure families against the thesis palette, typography and geometry.
- [x] Refresh VPM completion status and distinguish continue-ready cases, active runs, completed runs and physical health stops.
- [x] Recover the original README animation into its tutorial assets and verify its frames/hash.
- [x] Standardize supported CPython minor version to 3.11 throughout packaging, installation, Conda, documentation and CI.
- [x] Verify an installed wheel outside the checkout, without source-path overrides, including meshing/FVM/VPM/restart and plotting.
- [ ] Repeat final wheel installation, package tests and direct tutorial entrypoints after cleanup.
- [ ] Obtain real macOS CI evidence; Linux execution alone does not establish it.
- [x] Commit completed solver/package/archive work locally; retain unrelated work and generated figures.
- [x] Commit this final cleanup and its measured evidence. Publishing remains separate from the explicitly local-only archive approval.

## Execution constraints

The host suffered an OOM during earlier verification. Subsequent work uses
disk-backed scratch and one numerical/rendering workload at a time. Existing
delta-wing, rotor and cylinder benchmark processes are user workloads and must
not be killed, restarted, or have their input/output trees rewritten by checks.
Run verification in isolated copied cases. Never fabricate missing production
data or relax numerical health checks to make a launcher appear successful.

## Final cleanup evidence

The public memory-pool fraction, per-diffusion node caps, secondary remeshing
capacity triggers, late-step refinement schedules and energy-feedback switches
were removed. GBD/DVH now reject insufficient declared particle capacity before
discarding retained nodes. Real diffusion overflow tests verify unchanged
positions, strengths, radii and volumes. Legacy inactive settings and operational
backend choices do not block restart; changed physical algorithms remain checked.

All 19 tracked setups and 73 launchers were audited. Reusable geometry and
diagnostic helpers live in the installed `openonda.tutorial_support` package;
setups import them normally. No executable tutorial code needs a home-directory
path or source-path manipulation. The wheel exported from the reviewed Git tree
contains all 26 support modules and no local phase-benchmark scripts or archives.

The copied native DVH checkpoint restored 80,958 particles on CUDA. Eight repeated
induction evaluations retained one tree workspace and 690 MiB device memory with
matching velocities. See [the CUDA record](../../studies/cuda_dvh_restart_2026-09-29.json).
The clean-clone flat-plate launcher also completed all seven analytical figures
and its ParaView scene using the installed package.

The native rotated-wall regression passed after three rejected trials and four
accepted RK intervals. Its endpoint remains outside the solid within the
geometry's declared tolerance. The original 126-test VPM sweep had one fixture
that depended on the removed node truncation; the corrected uninterrupted versus
restarted GBD comparison passes with sufficient declared particle capacity.
The 56 installation/archive/style/cadence contracts also pass.

The coupled manifest now uses the same canonical VPM compatibility rules as
standalone restoration, after authenticating the original saved configuration.
The original cube step-440 manifest validates against current VPM settings.
Thirteen coupled backup regressions and fifteen standalone compatibility checks
pass. Full initial/latest/completed-run coupled lifecycles pass both serially
and on two MPI ranks, including particle fields and CSV/PVD sample clocks.
The serial comparison uses tolerances scaled to float32 particle traces and
physical flow units; solver tolerances were not changed.
