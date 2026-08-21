# OpenONDA — FINAL COMPLETION, SCIENTIFIC DEBUGGING, TUTORIAL QA AND RELEASE AUDIT

## Completion review — 21 August 2026

Release status: **complete on the `development` branch**.

The checked items below were verified through the implementation history,
focused regression tests, full local scientific suites, representative CPU and
Metal tutorial runs, isolated wheel validation, and hosted CI. A checked
conditional investigation means that the branch was either executed or closed
by stronger evidence (for example, the valid fixed-time study proved BDF2 is
second order, so speculative defect-isolation work was not required).

Durable numerical results and command-level evidence are summarized in
`docs/PROJECT_COMPLETION_TODO.md`. The final reconciliation found these genuine
follow-ups; none is a defect in the supported release:

[ ] Merge `development` to the default `main` branch when ready to release.
    GitHub only schedules workflows from the default branch, so the corrected
    `.github/workflows/nightly.yml` cannot run nightly before that merge.
[ ] Establish and retain a post-correction performance baseline, then compare
    representative FVM and coupled profiles against it. Correctness and memory
    telemetry are covered, but no durable before/after timing baseline exists.
[ ] Validate the optional OpenVSP importer when its external Python API is
    available. Native VLM and panel paths are already covered.
[ ] Continue VPM--LES as research: the current variable-coefficient particle
    eddy-viscosity operator is not solenoidal and is not claimed as production
    validated. See `docs/vpm-les-validation.md`.

Repository:
    /Users/flaviomartins/OpenONDA

Remote:
    https://github.com/EngFlavioMartins/OpenONDA

Primary branch:
    development

Current remote development head at handoff:
    18e7c275725aa64bbcdede76dd867661b2115bcf

Final audited development head before this checklist reconciliation:
    6116ce8ee84647c7d4914d451e8a5517da13531b

You are taking over an advanced CFD solver project after a long API/nomenclature
consolidation.

YOUR JOB IS TO FINISH THE PROJECT TO A HIGH STANDARD.

This is not a request for another planning-only pass. You are authorized to inspect,
edit, test, instrument, run tutorial cases, create focused regression tests, make
logical local commits, and continue from one task to the next without repeatedly
asking whether you should proceed.

Do substantial work before reporting back.

Do NOT stop after diagnosing one failing test and ask “want me to continue?”.
Continue investigating and fixing demonstrated root causes.

Only stop and ask the user when:
- an action would destroy or overwrite user data;
- a force-push/reset/history rewrite would be required;
- a system-level/sudo installation is genuinely necessary;
- there are multiple scientifically defensible model changes and evidence cannot
  distinguish them;
- a required external dependency/service is inaccessible and no other useful work
  can proceed.

Otherwise continue autonomously.

===============================================================================
0. CORE OBJECTIVE
===============================================================================

Bring OpenONDA to a state where:

[x] FVM numerical certification is scientifically credible.
[x] VPM core and VLM/Panel coupling remain physically and numerically correct.
[x] FVM↔VPM coupling is scientifically defensible and regression-tested.
[x] Checkpoint/restart naming and paths are clean and predictable.
[x] Public physical-variable names are consistent across solvers.
[x] User-facing APIs contain no legacy aliases.
[x] Tutorials are clean, short, physics-focused and easy to understand.
[x] Representative tutorial cases actually run.
[x] Serial and MPI public APIs are identical.
[x] Output is consistently rooted in the case directory.
[x] CI/package/install checks pass or any environmental limitation is documented.
[x] No tests have been weakened merely to make the repository green.
[x] No temporary agent/debug/process documentation remains in the release-facing tree.

Maintain a live TODO checklist during the work.

Prefer:
    docs/PROJECT_COMPLETION_TODO.md

Keep it concise. It is an engineering checklist/evidence ledger, NOT a diary.

At the end, either:
- convert it into a concise durable verification document; or
- remove it if it has no lasting user/developer value.

Likewise, inspect `CONSOLIDATION_STATUS.md`.
Do not leave a large chronological “AI agent did X” document in the final clean
repository. Preserve only useful release/developer information.

===============================================================================
1. OPERATING RULES — IMPORTANT
===============================================================================

1.1 Work autonomously

You have permission to:
- edit source;
- edit tests;
- edit tutorials;
- create focused diagnostic scripts;
- run expensive tests when justified;
- run MPI;
- run tutorial simulations;
- run format/lint/build checks;
- create logical git commits;
- use the internet to verify numerical equations against primary literature or
  authoritative implementations when required.

Do not ask permission between ordinary tasks.

If one task becomes blocked, record it in the TODO and move to another task.

1.2 Never sacrifice physics for green tests

DO NOT:
- loosen scientific tolerances because the solver misses them;
- change benchmark reference values to match current output;
- reduce mesh resolution to make a failing test disappear;
- shorten a validation time horizon in the committed test merely to pass;
- mark a genuine failure xfail just to make CI green;
- change a slow test marker so CI stops exercising it;
- remove a CI platform merely because its job is failing without establishing
  that the platform is explicitly unsupported;
- add empirical coefficients solely to hit a test value;
- suppress exceptions hiding real bugs.

A test may be corrected only when you prove that THE TEST is mathematically or
physically wrong.

1.3 One scientific cause per commit

Keep scientific changes isolated.

Good:
    Fix BDF2 pressure-projection temporal consistency
    Correct WALE tensor invariant implementation
    Correct IBM interpolation/spreading normalization

Bad:
    Fix FVM tests
    Various solver fixes
    Make validation green

Nomenclature/user-interface cleanup may be grouped separately.

1.4 Do not reset existing work

Never use blindly:
    git reset --hard
    git clean -fd
    git checkout -- .
    git restore .
    force push

Inspect the local tree first.

If dirty:
    git status --short
    git diff
    git diff --check

Determine what the changes are before touching them.

===============================================================================
2. TOKEN / CONTEXT EFFICIENCY RULES
===============================================================================

The project is large. Conserve tokens and context aggressively.

[x] Do NOT dump entire 1,000–3,000 line source files into context.
[x] Locate symbols first with `git grep -n` or a small Python/AST script.
[x] Read only ±50–150 lines around relevant functions.
[x] Use `pytest -q --tb=short --maxfail=1` during diagnosis.
[x] Run the broad suites only at phase boundaries.
[x] Do not rerun a 1,000-test suite after every one-line change.
[x] Store long command output in /tmp or artifacts and inspect summaries/tails.
[x] Keep diagnostic scripts small and delete them when they are no longer useful.
[x] Reuse measured baselines rather than recomputing them unnecessarily.
[x] Do not repeatedly explain the repository architecture in chat.
[x] Update the TODO/evidence file instead.
[x] Report to the user only after completing a meaningful work batch or hitting a
    genuine blocker.
[x] Keep code comments concise. Do not insert essay-like AI comments.
[x] Prefer code clarity over explanatory comments.
[x] Avoid web research except where a numerical/physical equation actually needs
    external verification.
[x] When research is required, prioritize original papers, OpenFOAM source/docs,
    standard CFD literature, or another authoritative primary implementation.

The preferred cycle is:

    inspect -> reproduce -> isolate -> fix -> focused test -> related tests ->
    commit -> continue

not:

    inspect -> write long report -> ask permission -> repeat

===============================================================================
3. CURRENT STATE — DO NOT REDO BLINDLY
===============================================================================

Earlier consolidation work already addressed many items.

VERIFY them, but do not redo them without evidence.

Expected current state includes:

[x] Public modules:
        import openonda.fvm as fvm
        import openonda.vpm as vpm
        import openonda.coupler as coupling

[x] Factories:
        fvm.create_fvm_solver(...)
        vpm.create_vpm_solver(...)
        coupling.create_coupler(...)

[x] Public coupled scripts contain no rank-specific construction.

[x] VPM particle vector is `vortex_strength`, NOT `circulation`.

[x] Genuine scalar VLM Gamma remains `circulation`.

[x] VPM particle fields use:
        position
        velocity
        vortex_strength
        core_radius
        volume
        kinematic_viscosity
        eddy_viscosity
        effective_viscosity
        velocity_gradient
        strain_rate
        vorticity
        group_id
        zone_id

[x] FVM fields use:
        velocity
        kinematic_pressure
        face_flux
        eddy_viscosity

[x] Fluid properties use:
        density
        kinematic_viscosity
        dynamic_viscosity
        eddy_viscosity
        effective_viscosity

[x] Time vocabulary uses:
        time_step_size
        start_time
        end_time
        time
        step
        n_steps
        fvm_time_step_size
        vpm_time_step_size
        fvm_substeps

[x] Cadence uses explicit units:
        *_interval_steps
        *_interval_time

[x] FVM setup loading is canonical-only.

[x] VPM serialized states are canonical-only and forbid unknown fields.

[x] Coupler persistent metadata uses descriptive physical names.

[x] VLM particle absorption uses canonical particle fields.

[x] Panel/VLM public API uses descriptive velocity names.

[x] VPM/VLM molecular viscosity consistency is enforced.

[x] Inviscid wake particles receive exactly nu=0.

[x] Viscous wake particles receive exactly the configured molecular nu.

[x] VPM setup serialization refuses to silently discard a live VLM setup.

[x] Real public-API `mpirun -np 2` coupled smoke test exists and passes.

Re-run regression tests covering each of these after substantive work.

===============================================================================
4. HARD NOMENCLATURE CONTRACT
===============================================================================

These names are the canonical public vocabulary.

GENERAL:
    FVMSetup
    VPMSetup
    CouplerSetup
    subsystem configuration -> *Config
    setup argument -> setup
    solver setup -> self.setup

TIME:
    time_step_size
    time
    step
    start_time
    end_time
    n_steps
    substep_size
    n_substeps
    fvm_time_step_size
    vpm_time_step_size
    fvm_substeps
    *_interval_steps
    *_interval_time

FLUID:
    density
    kinematic_viscosity
    dynamic_viscosity
    eddy_viscosity
    effective_viscosity
    kinematic_pressure

FVM:
    velocity
    kinematic_pressure
    face_flux
    eddy_viscosity
    n_cells
    cell
    face
    centre / neighbour in our public/OpenFOAM-style terminology

VPM:
    vortex_strength
    core_radius
    particle_spacing
    particle_kernel
    compute_device
    max_evaluation_points
    domain_bounds
    regeneration_*
    dvh_support_radius_ratio
    n_particles
    n_sources
    turbulence_model

VLM:
    true bound-vortex Gamma -> circulation
    kinematic_viscosity
    density
    freestream_velocity
    external_velocity
    reference_velocity
    wake_velocity

Do not mechanically rename local mathematical variables merely to make them long.

Internal:
    nu, rho, dt, p, U, Gamma

can be acceptable in compact equations/kernels when the context is mathematically
unambiguous.

They should NOT leak into:
- setup dataclasses;
- public function parameters where a descriptive name is reasonable;
- tutorial constants;
- persistent metadata schemas.

OpenFOAM on-disk names such as U, p, phi, nut remain contractual where relevant.

===============================================================================
5. TRUE VLM CIRCULATION — NON-NEGOTIABLE
===============================================================================

Do not rename true VLM circulation Gamma.

VLM:
    Gamma [L^2/T] -> circulation

VPM particle strength:
    alpha_p ~= omega_p * V_p [L^3/T] -> vortex_strength

These must remain semantically distinct.

Audit every remaining `circulation` use before changing it.

Classification:
- bound VLM Gamma -> KEEP circulation
- circulation theorem / line integral -> KEEP circulation
- particle alpha vector -> vortex_strength
- volume integral of vorticity represented by particles -> total_vortex_strength

Never raw-search-and-replace `circulation`.

===============================================================================
6. CHECKPOINT / “BACKUP” NAMING — HIGH PRIORITY
===============================================================================

This is especially important to the user.

Use `checkpoint`, not `backup`, in the active public API.

TODO:

[x] Search active code, tests, tutorials, README and user-facing docs:

    git grep -n -i "backup" -- \
        source openonda tutorials scripts tests README* pyproject.toml

Classify every hit.

[x] Replace active/public names such as:
        DEFAULT_BACKUP_FILENAME
        COUPLER_BACKUP_PERIOD
        backup_frequency
        backup_directory
        backup_file_name
        “perform backup”
    with canonical checkpoint vocabulary.

Preferred examples:
        DEFAULT_CHECKPOINT_NAME
        checkpoint_name
        checkpoint_directory
        checkpoint_interval_steps
        checkpoint_interval_time

[x] If a value is in seconds, its name MUST end in `_interval_time`.
[x] If a value is a number of steps, its name MUST end in `_interval_steps`.

Known user-facing example needing inspection:
    tutorials/coupled_FVM_VPM/cube_flow/cubeFlow_setup.py

It currently contains names along the lines of:
    CHECKPOINT_INTERVAL
    COUPLER_BACKUP_PERIOD
    VPM_LOG_PERIOD
    TRANSFER_DIAGNOSTIC_INTERVAL

Make these self-describing, e.g.:
    VPM_CHECKPOINT_INTERVAL_TIME
    COUPLER_CHECKPOINT_INTERVAL_STEPS
    VPM_LOGGING_INTERVAL_STEPS
    TRANSFER_DIAGNOSTIC_INTERVAL_STEPS

Do not blindly use those exact suggestions if units differ; verify first.

[x] Rename stale comments/docstrings in SolverIO:
        “backup” -> “checkpoint”

[x] Rename stale constants such as DEFAULT_BACKUP_FILENAME if active.

[x] Audit `SolverIO.__init__`.

IMPORTANT suspected path bug:
`VPMSolver` resolves `self.checkpoint_directory` relative to `case_dir`, but
`SolverIO` currently appears to derive `export_dir` from
`solver.setup.checkpoint_directory`.

If the setup contains relative `"solution"` and cwd != case_dir, that can write
checkpoints to the wrong directory.

Verify this explicitly.

Preferred invariant:
    all solver output paths are rooted in case_dir unless the user deliberately
    supplies an absolute path.

Add a test that:
1. creates a case in a temporary directory;
2. changes cwd somewhere else;
3. runs/writes a checkpoint;
4. verifies the checkpoint is under the case directory;
5. verifies no stray `solution/` appears in cwd.

[x] Audit checkpoint file names.

Good default VPM form:
    vpm_000020.h5
    vpm_000020.xdmf

With a custom checkpoint_name:
    vpm_<name>_000020.h5
    vpm_<name>_000020.xdmf

Avoid:
    backup_20
    checkpoint_final_new
    vpm_vpm_name
    arbitrary timestamp-only names

[x] Coupled checkpoint bundle should remain predictable.

Current desirable pattern is approximately:
    solution/checkpoint/
        manifest.json
        fvm_000020.npz          # or partitioned directory
        vpm_000020.h5
        vpm_000020.xdmf
        vpm_bc_000020.npz

Verify actual behavior.

[x] Decide/document retention policy:
    latest checkpoint bundle only
or
    multiple step bundles

Do not accidentally delete earlier checkpoints unless latest-only retention is
intentional and documented.

[x] Test exact restart behavior:
    continuous run N steps
vs
    run K steps -> save -> restart -> N-K steps

Compare:
- time;
- step;
- velocity;
- pressure;
- face flux;
- BDF history;
- VPM position;
- vortex strength;
- core radius;
- viscosities;
- coupling boundary history;
- subcycle count.

Use mathematically appropriate exact/allclose comparisons.

[x] Preserve checkpoint precision rule:
    live particle fields -> configured f32/f64
    lineage/reference metadata -> float64

Especially:
    filament_reference_vortex_strength
    filament_reference_length
    divergence-relaxation reference moments

===============================================================================
7. USER-FACING TUTORIAL STYLE — HIGH PRIORITY
===============================================================================

Use:
    tutorials/coupled_FVM_VPM/cube_flow/cubeFlow_setup.py

as the stylistic baseline, but clean its remaining naming inconsistencies too.

The user wants tutorial files that are:
- short;
- explicit;
- physics-focused;
- easy to modify;
- minimally commented;
- free of internal implementation detail.

For every `*_setup.py` user-facing tutorial:

[x] Prefer a short top-level docstring:
    what the physical case is
    key Re/Mach/geometry if relevant
    one run command

[x] Imports should use public APIs:
        import openonda.fvm as fvm
        import openonda.vpm as vpm
        import openonda.coupler as coupling

Avoid `source.solvers...` in tutorials.

[x] Organize configuration into compact sections:
        Physical problem
        Domain / mesh
        Numerics
        Coupling (if applicable)
        Output / sampling
        Case files
        Setup objects
        main()

[x] Use uppercase physical constants.

[x] Compute derived physical quantities explicitly:
        nu = U * L / Re
rather than duplicating magic numbers.

[x] Keep `main()` extremely simple.

Ideal coupled main:

    def main() -> None:
        fvm_solver = fvm.create_fvm_solver(
            FVM_SETUP,
            case_dir=CASE_DIR,
            mesh=FVM_MESH,
        )
        vpm_solver = vpm.create_vpm_solver(
            VPM_SETUP,
            case_dir=CASE_DIR,
        )
        coupled_solver = coupling.create_coupler(
            fvm_solver,
            vpm_solver,
            COUPLER_SETUP,
        )
        coupled_solver.run()

[x] Same code serial and MPI.

FORBIDDEN in user tutorials:
    FVMVPMCoupler.is_master_rank()
    vpm_solver = None
    rank-specific imports
    MPI rank branching
    user-side log ownership
    internal proxy classes

[x] Remove lengthy comments explaining historical bugs.
[x] Remove “AI-style” explanation.
[x] No comments such as:
        “This was previously broken because...”
        “Regression fix for...”
in setup files.

Those explanations belong in tests/commit history, not tutorials.

[x] Comments should explain PHYSICS:
        Re
        mesh refinement
        boundary conditions
        sampling plane
        LES model
        coupling zone

[x] Fix stale usage commands.

For example, confirm that a docstring saying:
    python cube_flow_setup.py
actually matches the real filename.

[x] Audit tutorial filenames/directories for obvious inconsistencies, but do not
perform a large directory rename without checking downstream references.

[x] All samples go under:
        <case_dir>/samples/

[x] Solver/checkpoint/output state goes under:
        <case_dir>/solution/

No tutorial should write into repository root or caller cwd unexpectedly.

===============================================================================
8. CREATE AUTOMATED TUTORIAL CONTRACT TESTS
===============================================================================

Do not rely only on visual inspection.

Add/extend lightweight tests that verify:

[x] coupled tutorials import the public module namespaces;
[x] no user-side MPI ownership;
[x] no obsolete public names;
[x] no `source.solvers` imports in user setup scripts;
[x] setup modules import without launching a simulation;
[x] setup construction succeeds;
[x] `case_dir` is explicit;
[x] checkpoint cadence names have explicit units;
[x] sampler paths resolve below case_dir/samples;
[x] serialized metadata uses canonical names.

Do NOT build brittle tests enforcing exact comment count or line count.

Style should be reviewed manually, contracts should be tested mechanically.

===============================================================================
9. FIRST SCIENTIFIC TASK: REASSESS THE BDF2 DIAGNOSIS
===============================================================================

The previous agent stopped with this conclusion:

    ABC spatial order = ~1.780
    temporal order study = ~0.97
    therefore BDF2 is effectively first-order

DO NOT ACCEPT THIS CONCLUSION WITHOUT REPRODUCING IT CORRECTLY.

CRITICAL:

`_run_abc()` defines final time as:
    T = steps * time_step_size

Therefore, a “temporal convergence” study that changes time_step_size while
leaving `steps` fixed changes the physical final time.

THAT IS NOT A VALID TEMPORAL-CONVERGENCE STUDY.

This can easily produce apparent O(dt) behavior from accumulated spatial error.

Before changing any BDF2 production code:

[x] Reproduce the existing ABC certification:
        python -m pytest -q \
          tests/fvm/test_validation_abc_flow.py \
          --tb=short

Record:
    mesh levels
    errors
    fitted order
    continuity

[x] Reproduce the previous temporal experiment and inspect exactly how it was done.

[x] Run a VALID temporal study at fixed:
        mesh
        physical final time T

Example:
    T = 0.04
    dt = 0.01      -> 4 steps
    dt = 0.005     -> 8 steps
    dt = 0.0025    -> 16 steps
    dt = 0.00125   -> 32 steps

Do not compare simulations at different physical end times.

[x] Fit temporal order only before the spatial-error floor dominates.

[x] Plot/log:
        dt
        n_steps
        final time
        velocity L2 error
        continuity
        kinetic energy error
        pressure error if useful

[x] Repeat on a second analytic problem if needed, e.g. Taylor-Green.

Only if fixed-T evidence remains approximately first-order should production
BDF2 be considered defective.

===============================================================================
10. BDF2 ISOLATION PLAN
===============================================================================

If the fixed-T study confirms a defect, isolate it systematically.

Do not guess.

10.1 Assembly algebra

Current constant-step BDF2 should be:

    (3 U^{n+1} - 4 U^n + U^{n-1}) / (2 dt)

Equivalent matrix form:

    diagonal: 3/(2 dt)
    RHS history:
        2/dt * U^n
      - 1/(2 dt) * U^{n-1}

[x] Unit-test the assembled diagonal/RHS using a tiny synthetic matrix.

[x] Verify the actual assembled coefficients numerically.

[x] `assemble/momentum.py` currently contains both:
        `_add_transient_term(...)`
and duplicated transient coefficient logic in `assemble_momentum_equation()`.

Determine whether `_add_transient_term` is dead/redundant.

After correctness is established, prefer ONE source of truth for BDF coefficients.

Do not preserve duplicate coefficient logic unnecessarily.

10.2 Startup

[x] Verify step 1 uses BDF1 only.
[x] Verify steps >=2 use BDF2.
[x] Verify `_n_committed` semantics.
[x] Verify `velocity_old` / `velocity_older`.
[x] Verify `face_flux_old` / `face_flux_older`.

Experimental isolation:
seed an exact U^{n-1} state if practical and compare against the self-start case.

If exact-start BDF2 becomes second-order while self-start does not, investigate
startup consistency.

10.3 Rhie-Chow transient correction

Inspect:
    compute_ddt_flux_correction()

[x] Verify Euler and BDF2 branches.
[x] Verify history coefficients.
[x] Verify units.
[x] Verify which phi and U levels are used.
[x] Verify boundary treatment.
[x] Verify the correction is frozen appropriately through PIMPLE correctors.
[x] Compare formulation against authoritative OpenFOAM `fvc::ddtCorr` /
    `ddtCorr` behavior or literature.

The code comments and implementation must agree.

Do not “fix” the formula based on comments alone.

10.4 Projection / pressure splitting

Even if momentum BDF2 is algebraically correct, the FULL pressure-velocity
algorithm can still be first-order.

Audit:
[x] predictor pressure gradient time level;
[x] incremental vs non-incremental pressure correction;
[x] pressure update;
[x] velocity correction;
[x] face-flux correction;
[x] PIMPLE corrector sequencing;
[x] reuse of corrected flux;
[x] pressure boundary updates;
[x] Rhie-Chow correction;
[x] final committed velocity;
[x] final committed face flux.

Create a minimal manufactured transient incompressible problem if needed.

10.5 Outer correctors / relaxation

[x] Repeat temporal study with:
        n_outer_correctors = 1, 2, 3
[x] relaxation = 1
[x] tight linear tolerances

This distinguishes temporal discretization from incomplete nonlinear iteration.

Do not solve order problems by merely making tolerances absurdly tight unless the
convergence study proves iterative error is the cause.

===============================================================================
11. ABC SPATIAL CERTIFICATION
===============================================================================

Once temporal behavior is understood:

[x] rerun the existing ABC test.

Historical result:
    order ≈ 1.780
    required >= 1.8
    errors monotone
    continuity clean

Treat these as historical, NOT current ground truth.

If still below threshold:

[x] separate temporal and spatial error;
[x] test smaller dt at every mesh while holding final T fixed;
[x] test additional refinement levels;
[x] determine whether 6/8/12 is in the asymptotic range;
[x] inspect central face interpolation;
[x] inspect cyclic face pairing;
[x] inspect gradient reconstruction;
[x] inspect non-orthogonal terms;
[x] inspect Rhie-Chow consistency;
[x] inspect pressure null-space treatment;
[x] inspect convective skew symmetry / energy behavior.

Do not simply change 1.8 to 1.75.

If the test itself uses grids outside the asymptotic regime, prove this with
additional grids before considering a test change.

===============================================================================
12. TAYLOR-GREEN FVM CERTIFICATION
===============================================================================

Run:
    python -m pytest -q \
      tests/fvm/test_validation_taylor_green.py \
      --tb=short

Historical symptoms:
    central rel-L2 ≈ 2.18e-3
    upwind rel-L2 ≈ 5.64e-3
    expected central < upwind/4
    refinement orders approximately 1.69 and 1.84

Re-measure.

TODO:

[x] central KE decay against analytic solution
[x] upwind dissipation sign/magnitude
[x] central vs upwind velocity error
[x] spatial refinement order
[x] temporal refinement at fixed T
[x] kinetic-energy budget
[x] continuity
[x] pressure behavior
[x] central convective energy conservation

Potential core areas:
    convection interpolation
    convective operator
    gradients
    BDF2
    pressure projection
    Rhie-Chow
    cyclic boundaries

Fix the demonstrated common cause before proceeding to LES.

===============================================================================
13. WALE / TGV LES
===============================================================================

Only debug WALE AFTER baseline central/PIMPLE behavior is trusted.

Run:
    python -m pytest -q \
      tests/fvm/test_validation_les_decay.py \
      --tb=short

Historical symptom:
    dissipation peak around t≈1.28
    published DNS peak time around t≈8.86
    test expects physically plausible peak roughly 4<t<10

Re-measure current tree.

Audit the WALE formulation directly.

[x] velocity gradient convention
[x] transpose convention
[x] S_ij
[x] squared gradient tensor
[x] traceless symmetric Sd tensor
[x] numerator power
[x] denominator power
[x] epsilon treatment
[x] filter width Delta
[x] dimensions of nu_t
[x] nu_eff = nu + nu_t
[x] interpolation to faces
[x] explicit deviatoric stress term
[x] SGS dissipation sign
[x] SGS energy budget

Compare with original WALE literature or a trusted reference.

Build local/unit tests for:
[x] solid-body rotation behavior
[x] pure shear
[x] near-wall scaling if practical
[x] non-negative eddy viscosity
[x] tensor invariance under rotation

Do not tune c_w to force the DNS peak.

===============================================================================
14. IBM SCIENTIFIC CERTIFICATION
===============================================================================

Run:
    python -m pytest -q tests/fvm/test_ibm.py -m slow --tb=short
    python -m pytest -q tests/fvm/test_wall_force_certification.py --tb=short

Historical issues included:
    cylinder slip too large
    square force/wake mismatch
    suspicious/negative body-fitted drag

Re-measure.

Use analytic wall-force tests to separate:
    integration problem
from
    immersed-boundary forcing problem

Audit:

[x] Roma delta implementation
[x] interpolation partition of unity
[x] linear reproduction
[x] spreading/interpolation adjointness
[x] Pinelli quadrature weights
[x] marker spacing
[x] multidirect forcing
[x] force accumulation
[x] reaction-force sign
[x] pressure ghost at wall
[x] viscous stress
[x] PIMPLE/IBM ordering
[x] pressure correction after forcing
[x] second momentum solve behavior
[x] wake deficit
[x] force convergence under mesh refinement

Do not calibrate IBM coefficients against the expected drag.

Use body-conformed/body-fitted reference only after independently validating that
reference.

===============================================================================
15. FVM BROAD SCIENTIFIC AUDIT
===============================================================================

After known failures are fixed, run ALL FVM tests:

    python -m pytest tests/fvm -m "not mpi" -q --tb=short

Do not exclude slow tests for final scientific acceptance.

Then specifically review:

[x] diffusion MMS
[x] advection-diffusion MMS
[x] momentum MMS
[x] temporal order
[x] gradient convergence
[x] BDF2 integration
[x] cyclic boundaries
[x] pressure reference/nullspace
[x] Rhie-Chow
[x] PIMPLE final iteration
[x] non-orthogonal correction
[x] surface forces
[x] density convention
[x] LES model tests
[x] IBM
[x] 3D cavity / Poiseuille
[x] body-conformal mesh

No hidden known failures should remain unclassified.

===============================================================================
16. COUPLER SCIENTIFIC AUDIT
===============================================================================

Once standalone FVM is trustworthy, audit the FVM↔VPM coupling scientifically.

DO NOT use `referenceFlow/` as unquestioned truth.
It may be a useful comparison, but not an oracle unless independently verified.

Main physics questions:

[x] Does the coupling preserve the intended authority split?
        FVM authoritative near body
        VPM authoritative in wake
        computational support may overlap

[x] Is transfer weighting eta(x) mathematically consistent?

[x] Is FVM -> VPM vorticity/vortex-strength reconstruction correct?

[x] Is the relation between vorticity and particle vortex strength dimensionally
    correct everywhere?

[x] Are transfer conservation corrections correct for:
        total vortex strength
        linear impulse
        angular impulse where intended

[x] Is pruning conservative to the intended order?

[x] Do transfer diagnostics use `vortex_strength` terminology, not accidental
    scalar-circulation semantics?

[x] Do all supported regeneration threshold modes remain valid:
        budget
        relative_max
        absolute
        relative_local

IMPORTANT:
The Coupler MUST NOT impose a preference for `relative_local`.

Do not describe global methods as intrinsically invalid or “non-local”.
That was a previous hallucinated assumption and must not return.

[x] Verify time interpolation when FVM subcycles inside one VPM/coupling step.

[x] Verify no one-step lag in:
        velocity BC
        normal velocity
        tangential gradient
        pressure gradient
        transfer state

[x] Verify boundary flux correction.

[x] Verify pressure anchoring/datum handling.

[x] Verify pressure forces are unaffected by a pure pressure datum shift on closed
    bodies.

[x] Verify VPM-to-FVM boundary conditions in 3D.

Do not blindly extrapolate the familiar 2D relation:
    u.n = u_VPM.n
    d(u_t)/dn = omega x n

to a general 3D surface without deriving all tangential derivative terms.

Derive/check the actual 3D identity before claiming vorticity continuity.

[x] Check sign conventions with analytic manufactured fields.

[x] Check `(u, omega) -> (p, velocity)` reconstruction carefully.

[x] Check pressure-gradient calculation from VPM physics, if used.

[x] Check coupling stability under dt refinement.

[x] Check transfer under grid/particle-spacing refinement.

[x] Check spectral diagnostics only as diagnostics, not as a tuning target.

===============================================================================
17. COUPLED BENCHMARKS
===============================================================================

After the Coupler audit:

[x] cube
[x] cylinder shedding
[x] NACA4412

For each:
- startup;
- several coupling steps;
- finite fields;
- continuity;
- particle population;
- transfer diagnostics;
- boundary flux diagnostics;
- forces where meaningful;
- outputs;
- checkpoint;
- restart.

Do not require a 20-second high-resolution production run merely as a smoke test.

For expensive cases:
- leave canonical tutorial physics unchanged;
- create a temporary/test harness that imports the setup and replaces only
  duration/output cadence for a short execution.

Do NOT commit artificially weakened tutorial physics just to shorten CI.

At least one lightweight canonical tutorial should run completely as-is.

===============================================================================
18. VPM SCIENTIFIC AUDIT
===============================================================================

The fast VPM suite is large and currently strong, but perform a targeted
physics review after FVM/Coupler fixes.

[x] Biot-Savart direct kernel
[x] Gaussian regularization small-r behavior
[x] treecode/direct consistency
[x] stretching equation/sign
[x] transposed stretching mode
[x] advection RK2/RK3
[x] core spreading
[x] RWM
[x] DVH
[x] GBD
[x] vortex-strength conservation where expected
[x] impulse invariants
[x] filament refinement
[x] divergence relaxation
[x] LES model
[x] particle collision/removal
[x] VLM wake injection

Pay particular attention to the historical issue of uncontrolled particle
vortex-strength growth.

Do NOT introduce remeshing merely as a generic stabilizer for VLM+VPM wake
physics where release location/time is physically meaningful.

Do NOT attenuate vortex strength based on div(omega) or another heuristic without
a derivation and validation.

If investigating rVPM or strength-growth attenuation:
- obtain the actual equations from the paper/source;
- distinguish physical stretching from numerical growth;
- derive the attenuation mechanism;
- test it in canonical vortical flows;
- do not implement a speculative clamp.

===============================================================================
19. VLM / PANEL AUDIT
===============================================================================

True VLM circulation remains `circulation`.

Verify:

[x] VLM lifting-line behavior
[x] flat-plate loading
[x] spanwise tip taper
[x] frame equivalence
[x] rotated surface span coordinate
[x] symmetry/mirrored wing
[x] Kutta-Joukowski force convention
[x] panel Bernoulli force path
[x] reference velocity is a VECTOR where expected
[x] density propagation
[x] VPM-owned molecular viscosity agreement
[x] wake strength dimensions/sign
[x] transverse shedding
[x] particle absorption
[x] point-in-quad winding independence
[x] near-wake treatment
[x] collisions
[x] GPU/CPU path equivalence where supported

The relation used to create a VPM vortex-strength vector from VLM circulation
must be dimensionally and geometrically checked.

===============================================================================
20. OUTPUT / SAMPLER AUDIT
===============================================================================

User-visible output organization must be simple.

Required invariant:

    <case_dir>/
        samples/
        solution/

[x] FVM ForceSampler -> samples/
[x] FVM LineSampler -> samples/
[x] FVM SurfaceSampler -> samples/
[x] VPM LineSampler -> samples/
[x] VPM SurfaceSampler -> samples/
[x] VLM diagnostic CSV -> samples/ where appropriate
[x] flow integrals -> samples/
[x] restart/visualization state -> solution/

No:
    solution/samples/
unless there is a very explicit private/internal reason.

Test with cwd != case_dir.

Sampler names should be descriptive:
    fvm_centerline
    vpm_centerline
    fvm_slice_z0
etc.

No duplicate or ambiguous files.

===============================================================================
21. TUTORIAL EXECUTION ACCEPTANCE MATRIX
===============================================================================

Discover the exact setup filenames first.

Run a representative set.

At minimum:

FVM:
[x] Taylor-Green
[x] one wall/external-flow case:
        cube_flow OR airfoil_flow
[x] cylinder_ibm after IBM is fixed

VPM:
[x] Lamb-Oseen vortex
[x] vortex ring
[x] one VLM/VPM lifting case:
        flatPlate OR deltaWing

The delta-wing tutorial completed one Metal step with 576 VLM panels and
case-rooted outputs. Its VPM/VLM molecular viscosity is consistent and its
`0.0025 s` time step satisfies the wake-convection resolution criterion.

COUPLED:
[x] cube_flow serial short run
[x] cube_flow `mpirun -np 2` short run
[x] cylinderSheddingFlow short run
[x] naca4412Flow initialization + short run if computationally practical

For heavy cases:
    smoke execution may use a test/temp setup derived using dataclasses.replace
    or equivalent.

Do not permanently alter canonical tutorial duration/resolution for smoke tests.

For every executed tutorial inspect:

[x] exit code
[x] no exceptions
[x] no NaN/Inf
[x] expected samples exist
[x] expected solution files exist
[x] checkpoint names are correct
[x] no stray output outside CASE_DIR
[x] restart works when applicable
[x] logs are intelligible
[x] no user-facing MPI/rank noise

Keep a compact table in PROJECT_COMPLETION_TODO.md:
    case | mode | steps/time | result | output checked

===============================================================================
22. NIGHTLY TUTORIAL VALIDATION BUG
===============================================================================

Current `.github/workflows/nightly.yml` calls:

    scripts/validate_native_tutorials.sh

but that script is not present in the current repository tree.

Fix this.

Preferred approach:
- create a small cross-platform Python validation driver:
      scripts/validate_native_tutorials.py
- or restore a correct shell wrapper if there is a strong reason.

Then update nightly accordingly.

The validator should:
- run lightweight representative cases;
- use public installed APIs;
- execute outside the source checkout where practical;
- fail clearly;
- not modify canonical tutorial physics;
- not leave large outputs in the repo.

Also remove/modify any nightly comment suggesting genuine scientific failures
should simply be xfailed to get a green workflow.

Known scientific failures should remain visible until fixed.

Once the scientific suite is actually green, consider making it a stronger gate.

The validator and nightly jobs are implemented on `development`. The scheduled
trigger becomes active only after this workflow reaches GitHub's default `main`
branch; that branch-policy step remains listed in the completion review above.

===============================================================================
23. CI PLATFORM DECISION
===============================================================================

Current head removed:
    macos-15-intel / Python 3.11

from the wheel-import matrix.

Do not automatically restore or delete it again.

Determine official platform policy from:
- README;
- pyproject;
- install scripts;
- intended project support.

If Intel macOS is officially unsupported:
[x] Not applicable: the project continues to claim generic macOS support, so
    dropping Intel coverage was rejected.

If generic macOS support is claimed:
[x] restore Intel CI;
[x] diagnose packaging rather than deleting the failing target.

The Intel `macos-15-intel` / Python 3.11 wheel job is restored with the last
compatible Taichi dependency set and passes in hosted CI.

Never call removal of coverage a solver fix.

===============================================================================
24. QUALITY / PACKAGING GATES
===============================================================================

At phase boundaries run:

    python -m compileall -q source tests tutorials openonda scripts

    ruff check source tests tutorials scripts openonda

    ruff format --check source tests tutorials scripts openonda

    python -m openonda.verify_install

    python -m pytest -q tests/vpm -m "not gpu and not slow"

    python -m pytest -q tests/coupler -m "not mpi and not slow"

    python -m pytest -q \
        tests/fvm \
        -m "(unit or verification) and not slow and not mpi"

Run full scientific FVM before final acceptance:
    python -m pytest -q tests/fvm -m "not mpi"

Run MPI:
    existing FVM MPI tests
    coupled 2-rank smoke

When dependencies permit:
    pyrefly check

Do not increase the existing pyrefly error baseline.

Package:
    python -m build
    twine check dist/*

Create isolated environment and install wheel.
Run:
    python -m openonda.verify_install --require-site-packages
    python -m pip check

===============================================================================
25. GPU / OPTIONAL DEPENDENCIES
===============================================================================

On Apple Silicon, run applicable Metal VPM tests.

[x] GPU treecode
[x] GPU/direct comparisons
[x] GBD/DVH workspace if supported
[x] VLM Taichi paths

CUDA-specific tests may legitimately skip on Mac.

OpenVSP-dependent tests may skip when OpenVSP API is genuinely unavailable.

Report environmental skips accurately.
Do not convert code failures into “optional dependency” skips.

===============================================================================
26. PERFORMANCE AFTER CORRECTNESS
===============================================================================

Do not optimize broken numerics.

Once scientific gates are green:

[ ] rerun representative FVM benchmark after establishing a durable baseline
[ ] compare against the stored baseline
[ ] profile major phases in a representative post-correction production case
[x] check peak memory
[ ] quantify whether scientific fixes caused a material timing regression
[ ] profile a short coupled cube case against the retained baseline
[x] inspect VPM treecode/direct selection
[x] eliminate obvious duplicate allocations or computations only when safe

The profiler and benchmark tooling are present and regression-tested, including
phase, linear-solver, and memory telemetry. The unchecked items require a
retained timing baseline; inventing one after the fixes would not be a valid
before/after comparison and is therefore left as an explicit follow-up.

Do not tune physical resolution to claim performance improvement.

===============================================================================
27. CODE QUALITY / AI-ARTIFACT SWEEP
===============================================================================

Near the end perform a human-style cleanup.

Search for:
    TODO
    FIXME
    HACK
    legacy
    deprecated
    workaround
    backup
    AI
    agent
    previous version
    regression fix
    temporary
    debug

Classify each hit.

Tests may reasonably mention regressions.

User-facing solver/tutorial code should generally not narrate development history.

Remove:
[x] dead compatibility aliases
[x] dead imports
[x] `if TYPE_CHECKING: pass`
[x] self-assignments
[x] stale comments after renames
[x] debugging prints
[x] temporary instrumentation
[x] huge AI-generated comments
[x] duplicate helper logic where a single safe source of truth is possible

Do NOT reduce helpful scientific docstrings to cryptic code.

===============================================================================
28. PUBLIC NOMENCLATURE REGRESSION TEST
===============================================================================

Create/maintain a focused test so the cleanup cannot regress.

It should check public/setup/serialization surfaces for forbidden old names.

Examples to reject where semantically obsolete:
    processing_unit
    particles_kernel
    max_targets
    characteristic_distance
    backup_frequency
    backup_directory
    backup_file_name
    coupler_backup_period
    initial_p
    alpha_u
    alpha_p
    momentum_tol
    pressure_tol
    V_inf
    U_ref
    V_external
    V_wake_field

Do not forbid:
    genuine mathematical local variables
    VLM circulation
    OpenFOAM contract strings
    archived offline migration readers

AST/signature checks are better than brittle raw-text checks where feasible.

===============================================================================
29. GIT WORKFLOW
===============================================================================

At start:

    cd /Users/flaviomartins/OpenONDA
    git branch --show-current
    git status --short
    git fetch origin
    git log --oneline --decorate -15
    git rev-list --left-right --count origin/development...HEAD

Do not discard dirty changes.

Make local commits freely when one logical batch is complete.

Suggested commit families:

    Correct BDF2 temporal consistency
    Restore second-order ABC/TGV convergence
    Correct WALE LES dissipation behavior
    Correct IBM forcing consistency
    Harden coupled transfer verification
    Normalize checkpoint naming and paths
    Polish tutorial public setups
    Add native tutorial execution gate
    Final scientific verification and release cleanup

Commit messages should describe engineering changes.

Do not use:
    AI fix
    agent work
    make tests pass
    various fixes

Normal fast-forward push to `development` is authorized at coherent, fully-tested
phase boundaries.

Before any push:

    git fetch origin
    git rev-list --left-right --count origin/development...HEAD

If remote contains commits not in local:
STOP the push, inspect and integrate safely.

Never force-push development.

===============================================================================
30. TODO / EVIDENCE FORMAT
===============================================================================

Maintain something similar to:

# OpenONDA completion

## A. Repository/API
- [x] Canonical public solver modules
- [x] Rank-agnostic Coupler construction
- [x] Final stale-name scan
- [x] Public nomenclature regression test

## B. Checkpoints
- [x] Remove active “backup” terminology
- [x] Case-root path test
- [x] Standalone VPM restart parity
- [x] Standalone FVM restart parity
- [x] Coupled restart parity
- [x] Filename contract
- [x] Precision contract

## C. FVM core
- [x] Correct fixed-T temporal convergence study
- [x] BDF2 temporal order
- [x] ABC spatial order
- [x] TGV central/upwind
- [x] TGV refinement
- [x] full non-MPI FVM

## D. LES
- [x] WALE tensor unit tests
- [x] TGV decay timing
- [x] SGS budget

## E. IBM
- [x] cylinder slip
- [x] square force
- [x] wake deficit
- [x] refinement
- [x] wall-force certification

## F. Coupler
- [x] BC identity audit
- [x] pressure treatment
- [x] time interpolation
- [x] conservation
- [x] flux
- [x] threshold modes all accepted
- [x] serial smoke
- [x] MPI smoke
- [x] restart

## G. VPM/VLM
- [x] VPM strength-growth audit
- [x] VLM circulation/particle-strength dimensional audit
- [x] VLM loading
- [x] panel forces
- [x] absorption
- [x] GPU where available

## H. Tutorials
- [x] FVM setup files polished
- [x] VPM setup files polished
- [x] Coupled setup files polished
- [x] checkpoint names polished
- [x] sampler paths verified
- [x] lightweight tutorials run
- [x] coupled tutorials smoke-run
- [x] native tutorial validation script

## I. Packaging/CI
- [x] build
- [x] twine
- [x] isolated wheel
- [x] pip check
- [x] CI
- [x] nightly implementation (scheduled activation awaits the `main` merge)
- [x] platform support documented

Each completed scientific item should include a one-line evidence note:
    BEFORE:
    AFTER:
    COMMAND:

Do not put multi-page narratives into this file.

===============================================================================
31. FINAL RELEASE-FACING CLEANUP
===============================================================================

Before declaring completion:

[x] tracked project state clean (the unrelated local ALAC launcher remains
    deliberately untracked)
[x] `git diff --check`
[x] no temp scripts
[x] no generated solver state tracked; qualified `samples/` data is deliberately
    versioned for cross-device post-processing
[x] no `__pycache__`
[x] no `.DS_Store`
[x] no internal agent status dump
[x] no stale `CONSOLIDATION_STATUS.md` unless deliberately converted to useful docs
[x] no obsolete user-facing names
[x] no active `backup_*` terminology
[x] tutorial setup files concise
[x] README installation commands actually work
[x] README points to real tutorial filenames
[x] public API examples actually execute
[x] package imports outside checkout
[x] representative tutorials execute
[x] checkpoints have good names
[x] restart proven
[x] full FVM scientific failures resolved or explicitly reported with evidence
[x] VPM/Coupler gates clean
[x] MPI smoke clean

===============================================================================
32. DEFINITION OF DONE
===============================================================================

Do NOT declare the project complete merely because the fast test suite is green.

Completion requires:

1. Correct public API/nomenclature.
2. Correct checkpoint naming/path/restart.
3. Full FVM known scientific defects resolved or rigorously proven to be flawed
   tests and corrected with evidence.
4. WALE certification scientifically sensible.
5. IBM certification scientifically sensible.
6. Coupler scientific contracts audited.
7. VPM/VLM/Panel critical paths audited.
8. Representative tutorials executed.
9. Serial + MPI public API validated.
10. Build/install/CI evidence.
11. Clean user-facing tutorials.
12. No test weakening.

===============================================================================
33. FINAL REPORT TO USER
===============================================================================

After doing as much of the project as possible, return ONE consolidated report.

Include:

A. Git
    starting SHA
    final SHA
    commits made
    remote status

B. TODO status
    number completed
    remaining items
    blockers

C. Scientific FVM
    temporal order
    ABC order
    TGV errors/order
    WALE peak value/time
    IBM slip/forces/wake
    root causes fixed

D. VPM/VLM
    main audits
    strength-growth findings
    VLM circulation/strength correctness
    GPU results

E. Coupler
    BC audit
    conservation
    time interpolation
    pressure
    serial/MPI
    checkpoint/restart

F. Checkpoints
    exact naming
    directory structure
    restart parity
    path isolation

G. Tutorials
    files cleaned
    cases actually run
    duration/steps
    result
    output checked

H. Test matrix
    compileall
    Ruff
    format
    pyrefly
    FVM fast
    FVM full
    VPM
    Coupler
    MPI
    package
    tutorial validation

I. CI
    current jobs
    any failure
    whether failure is source, numerical, packaging, or environment

J. Remaining limitations
    only genuine unresolved limitations

Do not finish the report with:
    “Want me to continue?”

If anything remains that can be worked on locally, continue working on it first.
