# Original-request completion ledger

This ledger covers the complete conversation, including requests predating the
archive work. An implemented fix, a regression test, a real tutorial execution,
and a completed scientific study are different kinds of evidence. Unchecked
items must not be described as complete.

## Solver portability and the reported failures

- [x] Canonical and legacy VPM backup discovery, numeric ordering, and RWM member reuse implemented.
- [x] Run the original completed Lamb–Oseen CS/DVH/GBD continuation commands with the installed final package; execute real RWM member discovery and aggregation (coverage below).
- [x] Restore native FVM, VPM and coupled state from latest backups; initial mode ignores prior state; launchers distinguish cleanup from continuation.
- [x] Repeat real coupled initial/latest/completed-run execution in serial and on two MPI ranks, exercising both native solver states and sampled output clocks. Standalone installed-package verification is also repeated below.
- [x] Remove stale particle-capacity values from restart identity where they do not define the numerical model.
- [x] Exercise the original DVH/treecode allocation failure on the actual CUDA device and check repeated evaluation memory.
- [x] Implement transactional subdivision for excessive coupled wall corrections, preserving the accepted macro step.
- [x] Exercise excessive wall correction using native particle motion, the real RK2 integrator and a rotated wall; subdivision preserves circulation, one accepted clock and projection diagnostics.
- [x] Correct ParaView view/layout screenshot targets; include the leapfrogging reference table; correct thesis-layout clearance.
- [x] Finish real plotting executions that exercise each reported plotting failure.

## Minimal configuration and readable tutorials

- [x] Remove remaining public GPU-memory, diffusion-node and remeshing soft-capacity controls and associated truncation paths.
- [x] Remove the public VPM target-batch setting; authenticated legacy backups ignore that operational size. Filament refinement now rejects an insufficient hard particle capacity before a partial split. See the [native capacity checks](../../studies/vpm_capacity_restart_qualification_2026-09-29.md).
- [x] Verify old backups remain readable after retiring operational settings; reject genuinely incompatible physical state.
- [x] Audit all tracked tutorial setup files and comments for redundant constants, machine assumptions, incident-specific prose, and unnecessary orchestration.
- [x] Remove avoidable nonphysical setup arguments and package-bootstrap clutter without breaking editable local assets.
- [ ] Confirm particle splitting handles core growth/strength conservatively and qualify rotor stability past the reported 7.5 s failure.
- [ ] Complete the quadcopter strain-failure qualification and verify full-horizon particle capacity; preserve intentionally unstable comparison cases and label their health stops accurately.

## Cylinder geometry, accuracy and performance

- [x] Preserve Re=150 and implement resolved three-dimensional span configurations.
- [x] Shared solid geometry excludes injected particles and constrains shallow wall crossing; test curved, rotated, thin and concave surfaces.
- [x] Implement consistent slip-span induction/diffusion and physical interpolation on anisotropic donors.
- [x] Implement measured trace/interpolation reuse, image accumulation and PETSc workspace optimizations; retain accuracy/iteration limits.
- [x] Record short transient coupling sensitivity and exact optimization evidence in the cylinder execution report.
- [ ] Complete independent span/z, domain, grid and temporal qualification and select coupled/reference production meshes.
- [ ] Complete implemented exchange/renewal-clock, transfer-support and particle-spacing sensitivity with uncertainty and paired physics.
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
- [x] Commit twelve checksum manifests and LFS archives, including the completed IBM default; verify local-clone hydration and fresh-export restoration without reducing the saved histories.
- [x] Measure temporal reconstruction/spectral sensitivity for all archived delta/rotor panel-loading streams and aggregate forces; report qualified/rejected candidates without deleting data. Evidence: [cadence study](../../studies/loading_cadence_report.md). Final-run cadence certification remains dependent on completed histories.
- [ ] Ensure every expected tutorial plot/scene/animation has genuine portable inputs and execute every allplot launcher from a clone.
- [ ] Visually inspect all figure families against the thesis palette, typography and geometry.
- [x] Refresh VPM completion status and distinguish continue-ready cases, active runs, completed runs and physical health stops.
- [x] Recover the original README animation into its tutorial assets and verify its frames/hash.
- [x] Standardize supported CPython minor version to 3.11 throughout packaging, installation, Conda, documentation and CI.
- [x] Verify an installed wheel outside the checkout, without source-path overrides, including meshing/FVM/VPM/restart and plotting.
- [x] Rebuild and repeat outside-checkout native solver, restart, plotting and direct-entrypoint verification after the final allocation, finite-facet and field-plot commits. The [f1deb186 wheel](../../studies/installed_wheel_verification_f1deb186.json) passed the complete verifier and pip check.
- [ ] Obtain real macOS CI evidence; Linux execution alone does not establish it.
- [x] Commit completed solver/package/archive work locally; retain unrelated work and generated figures.
- [ ] Commit the remaining active source changes and final verification evidence. Publishing remains separate from the explicitly local-only archive approval.

## Execution constraints

The host suffered an OOM during earlier verification. Subsequent work uses
disk-backed scratch, isolated runs and memory monitoring. Additional trials wait
for headroom; shared-host timings are not treated as idle-machine benchmarks. Existing
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

Native RWM aggregation was rerun from the original saved states in isolated
output directories: all 16 vortex members across all 53 checkpoints, and all
28 dipole/14 merging members at steps 0, 468 and 927. All rebuilt fields and
jackknife diagnostics completed; sampled maximum relative errors were 6.735%,
6.801% and 7.010%, respectively. The latter two are bounded regression checks,
not full-history error maxima. See [the execution record](../../studies/rwm_aggregation_verification_2026-09-29.json).

The earlier wheel built from commit `048a77cf` passed
`python -I -m openonda.verify_install --require-site-packages`. Native FVM/VPM,
Cartesian meshing, installed tutorial resources, direct commands and rendering
completed. Its SHA-256 is
`41570e4b2e2caa97dc2cee14365d6aac4646fef9baba2d37ffc7315288d7082d`.
Completed CS, DVH and GBD tutorial commands each returned success at their saved
step 927 / time 29.973 s without extending the solution or allocating induction
work for an already completed run.

## Additional execution and archive closure

The shared mixed-boundary FVM correction passed the genuine 12,642-cell IBM
case to t=0.6 s, beyond its original t=0.15546 s failure; serial and two-rank
projection regressions passed. Completed unchanged boundary-layer and step
defaults were archived losslessly. Their ordinary plotting launchers restored
all data from a fresh Git export and reproduced the four original image hashes.
The boundary-layer errors remain above their stated targets; a passing plot is
not an accuracy qualification. See the
[small-case evidence](../../studies/fvm_small_cases_archive_verification_2026-09-29.json).

The complete coupled-cube plotting launcher returned zero; all 353 PNGs decoded
successfully. The four time-dependent families each contain 88 frames through
t=22 s, plus coupling diagnostics. Raw force histories keep their independent
sampling clocks. See the
[cube execution record](../../studies/cube_archive_plot_verification_2026-09-29.json).
The repaired delta archive also passed its complete plotting launcher from a
fresh restoration, with all five figure families inspected.

Native refinement tests exposed and fixed empty-wake lineage initialization,
first-emission moment capture, and restoration from a populated state to an
empty initial checkpoint. These are shared solver fixes, not changes to health
limits. The clean CUDA quadcopter qualification remains in progress; passing
these native lifecycle tests alone does not establish full-run stability.

The originally reported vortex-ring, Lamb–Oseen and interaction plotting failures
were exercised by their complete ordinary launchers using genuine archived data.
Lamb–Oseen returned zero under a retained supervisor; all seven PNGs decoded and
representative analytical and particle figures passed visual inspection. See the
[Lamb execution record](../../studies/lamb_archive_plot_verification_2026-09-29.json).

The corrected rotor full plotting launcher returned zero under a detached
supervisor. Its five analytical PNGs and 274 GIF frames decoded. This closes
plotting execution coverage for the archived accepted history; full-horizon rotor
stability remains unqualified. See the
[rotor execution record](../../studies/rotor_archive_plot_verification_2026-09-29.json).

The wheel from `585d6c48` passed the full installation verifier with `python -I`
from `/tmp`, outside the checkout, and `pip check` found no broken requirements.
The wheel contains all 26 tutorial-support Python files, excludes result archives
and local phase-benchmark scripts, and declares only CPython 3.11. The exact wheel
hash and native results are in the
[installation record](../../studies/installed_wheel_verification_585d6c48.json).

Actual execution exposed a separate IBM cylinder force-normalization error:
the implicit unit reference area was 16 times the cylinder's projected area.
The setup now passes diameter times span to the force sampler. Four regressions
cover two spans and two reference speeds. The corrected default finished at
t=60 s and was archived losslessly as the twelfth result bundle. Its real
`allplot.sh` completed; mean Cd=1.7681 is in the cited band, while the measured
recirculation length 2.137D is outside 1.55–1.70D, so accuracy is not qualified.
The field figures now cover the actual stretched-cell cross-section without
scatter gaps. See the [bundle manifest](../../tutorials/fvm/cylinder_ibm/assets/results/manifest.json).

Cylinder particle-spacing studies now hold the physical core radius and transfer
widths fixed, recording both requested and resolved lattice spacings. Sampling
cadence no longer depends on divisibility of the run horizon: a 2501-exchange run
keeps the same physical cadence as a 2500-exchange run. Configuration, factory
equivalence and cadence regressions passed. The long physical comparisons remain
open.

The original injection-rate request can be assessed through the implemented
combined exchange/renewal clock. A separate held-cloud algorithm is additional
scope, not a prerequisite for reporting that sensitivity honestly. The minimum
paired study and current timing evidence are in the
[scope assessment](../../studies/cylinder_injection_rate_scope_2026-09-29.md).
Historical reference data use a different span and observation window and cannot
replace the current campaign; see the
[historical input audit](../../studies/historical_plot_input_audit_2026-09-29.md).

## Final installed solver and generalized scene checks

The wheel from `37de66ba` passed the complete verifier outside the repository
with isolated Python imports. This now includes native coupled initial/latest
execution (64 FVM cells, four FVM steps and two VPM steps), in addition to the
standalone solvers, mesher, rendering and installed tutorial entrypoints.
`pip check` passed. See the [exact wheel record](../../studies/installed_wheel_verification_37de66ba.json).

The flat-plate scene no longer assumes a particular saved step, particle count,
plate velocity or camera. Genuine steps 160 and 197 both rendered correctly with
the shared vorticity palette; see the [native scene checks](../../studies/flat_plate_native_render_verification_2026-09-29.json).
The ordinary complete launcher then passed from a fresh Git export in a path
containing spaces, restoring its committed lossless archive and producing all
nine PNGs. See the [launcher record](../../studies/flat_plate_archive_plot_verification_2026-09-29.json).

Exact SciPy direct-factorization reuse preserves matrix identity and each solve's
residual checks. Native cached/uncached results agree, and repeated paired
solve-stage measurements show reduced cost; these measurements are not a
whole-case speed claim. See the [factorization audit](../../studies/fvm_exact_factorization_audit.md).

Actual installed coupled execution exposed floating-point curl in uniform flow.
The shared transfer now recognizes a wholly unresolved curl field at a
velocity/spacing-scaled machine-precision bound; any resolved curl preserves the
entire field. Seventy-three transfer regressions, native two-rank continuation,
and all five continuation tests at the unchanged default divergence tolerance
pass. See the [roundoff analysis](../../studies/coupled_uniform_transfer_roundoff.md).

The airfoil setup's finest spacing applies to its surface rather than every
outer domain plane. Its complete 105,508-cell mesh now passes strict native
validation after finite-facet feature recovery. The flow run and real airfoil
plotting remain pending; see the [geometry qualification](../../studies/airfoil_mesh_verification_2026-09-29.md).

The NACA 4412 scalar force history now samples every accepted FVM step while
large field snapshots retain their 0.8 s schedule. Its actual sampler schedule
and CSV writer passed the focused regression; full-case portable input remains
outstanding.

The cfMesh face lookup uses compact integer keys and releases ordering scratch
after its last use. Thirteen exact/native checks cover face order, duplicates,
missing faces, integer widths and array layouts. An isolated 100,000-face
comparison reduced the lookup's peak RSS increment from 29,588 to 10,372 KiB and
time from 0.579 to 0.323 s; this does not establish full-mesher gains. See the
[lookup evidence](../../studies/cfmesh_face_lookup_verification_2026-09-29.json).

The rebuilt `685c2d8a` wheel passed the complete outside-checkout verifier and
`pip check` in a separate CPython 3.11 virtual environment, including native
coupled continuation and the updated mesher. The wheel was installed locally
while dependencies came from the development environment; this complements the
earlier independent dependency environment. See the
[installation record](../../studies/installed_wheel_verification_685c2d8a.json).

Both body-fitted reference flows now have ordinary `allplot.sh` launchers and
shared thesis-sized grid figures. The cube launcher restored genuine archived
samples from the parent bundle in an isolated case and wrote both figure panels.
The cube's 15–30 s histories contain zero complete force cycles and do not
show monotone Richardson differences; its report and plots explicitly mark
statistics unqualified. Historical duplicate replay rows are accepted only
when the repeated values agree, without rewriting the archive. Cylinder
reference plotting remains blocked on genuine completed production samples;
the historical differing-span data cannot substitute for them.

## Latest installed package and IBM archive

The wheel from `fc921ec0` passed the outside-checkout installation verifier
and `pip check` in 229.44 seconds. It includes the source-batching, diffusion
workspace-lifetime and AMG pressure-default fixes. Native meshing, FVM, VPM,
coupled continuation, plotting and direct entrypoints ran successfully.
[Exact wheel and environment evidence](../../studies/installed_wheel_verification_fc921ec0.json).

The completed IBM cylinder archive restores into a fresh Git export and its
ordinary `allplot.sh` regenerates all four expected PNGs using that installed
wheel, including from a directory containing spaces. All four images were
visually inspected. The raw history is preserved losslessly; mean drag is
within its reference band, but wake length remains outside its band.
[Archive, execution and image hashes](../../studies/ibm_archive_plot_verification_2026-09-29.json).

The full airfoil meshing attempt reached 105,508 cells but failed strict final
validation: 114 fluid-cell centres lay inside the surface. The saved raw mesh
already has 76 such centres around the finite wing cap/side seam. Nearest-point
projection mixes the two surfaces on individual wall faces. A geometry fix and
a successful native rerun are required before this case can be called runnable.

The same installed wheel passed a 64-cell coupled lifecycle in serial and on
two MPI ranks, using the new default AMG pressure solver. Initial, latest,
completed no-op and initial-reset behavior all passed with native FVM clocks
and sampled histories checked.
[Native execution record](../../studies/coupled_installed_amg_lifecycle_2026-09-29.json).

## Final geometry, allocation and plotting corrections

The airfoil finite-feature projection uses connected finite STL facets and their
actual shared edge segments. It preserves sharp rims without a triangle-count
criterion or relaxed quality limits. The full retained native airfoil mesh has
zero interior fluid centres, zero unclosed cell edges and no inverted face
pyramids. The native two-disjoint-body fixture and ten focused feature tests
also pass. The raw native meshing stage was reused; this was a complete strict
finalization and independent revalidation, not a second full remesh.

Hierarchical induction now allocates scratch from the active source count and
destroys the old owned Taichi tree before growing. The sole public hard particle
capacity and target-query batching remain intact. Native CPU numerical, target
and restart checks and small CUDA allocation-growth checks pass. These checks
do not enable production CUDA FMM, which remains outside its qualified backend
set. See [allocation evidence](../../studies/vpm_lazy_hierarchical_workspace_2026-09-29.md).

Actual execution of the FVM cube plotting launcher found and corrected a long
force annotation and two plots that treated native cell data as point data. All
three figures now render from genuine partial outputs through 25 s, with force
history through 26.217 s. They were visually inspected; the default 120 s flow
run is still active. See [the plotting record](../../studies/fvm_cube_partial_plot_verification_2026-09-29.json).
The same-class audit changed airfoil velocity figures to native section polygons
and checked the IBM force figure against its complete saved history. The airfoil
field reader has geometric fixture coverage; genuine flow-output verification
remains required. See [the association audit](../../studies/fvm_plot_association_audit_2026-09-29.json).

The quadcopter accepted checkpoints show correct child-reference resets and
continued stretching under the current bisection rule. The capacity
risk is real: the original emission count alone approaches its declared ceiling.
Neither increasing capacity arbitrarily nor suppressing splits has been accepted
as a stability fix. See [the growth audit](../../studies/quadcopter_refinement_growth_audit_2026-09-29.json).

The committed `f1deb186` wheel passed the complete outside-checkout native
installation verifier and `pip check` in 209.28 seconds. Its exact wheel hash,
CPython 3.11 requirement, packaged helpers and genuine README GIF are recorded
in [the final-wheel check](../../studies/installed_wheel_verification_f1deb186.json).

The same installed wheel restored the original 80,958-particle DVH checkpoint
on the actual CUDA device. Eight repeated evaluations retained one tree of
capacity 80,958 and matching velocities. NVIDIA process allocation remained
434 MiB on all eight samples, compared with 690 MiB in the earlier verified
implementation. This directly exercises the originally reported allocation
failure; it does not promise that arbitrarily large physical problems fit any
GPU. See [the native checkpoint record](../../studies/cuda_dvh_restart_f1deb186.json).
