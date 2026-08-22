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
