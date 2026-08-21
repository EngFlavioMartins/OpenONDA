# OpenONDA completion

Evidence ledger for the final completion / scientific debugging / tutorial QA /
release audit pass. Engineering checklist, not a diary. See git log for detail.

Repo: /Users/flaviomartins/OpenONDA
Start SHA (this pass): 18e7c275725aa64bbcdede76dd867661b2115bcf (origin/development == HEAD)

Working method this pass: heavy iteration (pytest/ruff/pyrefly/build) done in a
throwaway clone in a cloud sandbox for speed; logical commits are synced back
onto this checkout via `git fetch <bundle>` + fast-forward merge, never
`reset --hard`. See "Environment notes" for exact tool/version pins.

## A. Repository/API
- [x] Canonical public solver modules (verified: `openonda/` package present, prior
      commits migrate to `openonda.fvm`/`openonda.vpm`/`openonda.coupler`)
- [ ] Final stale-name scan
- [ ] Public nomenclature regression test

## B. Checkpoints
- [ ] Remove active "backup" terminology
- [ ] Case-root path test
- [ ] Standalone VPM restart parity
- [ ] Standalone FVM restart parity
- [ ] Coupled restart parity
- [ ] Filename contract
- [ ] Precision contract

## C. FVM core
- [x] Correct fixed-T temporal convergence study — `tests/fvm/test_temporal_order.py`
      already does this correctly (self-convergence at fixed physical T=0.4,
      mesh fixed, only n_steps varies 8/16/32). Re-measured:
      BEFORE (invalid protocol, historical): BDF2 order ~0.97 (dt varied with
      steps fixed ⇒ T changed per point — confounded with spatial error).
      AFTER (valid protocol, this test): Euler order 0.941 (≈1, correct),
      **BDF2 order 2.049** (≈2, correct). `pytest tests/fvm/test_temporal_order.py -v`
      → 2 passed. CONCLUSION: BDF2 is NOT defective; the historical "effectively
      first order" verdict was an artifact of an invalid dt-vs-steps-fixed study,
      exactly as PLAN.md §9 warned. No BDF2 production-code change needed/made.
- [ ] ABC spatial order — reproduced historical order=1.780 exactly
      (`tests/fvm/test_validation_abc_flow.py`, levels 6/8/12, errors
      [2.515e-3, 1.387e-3, 7.257e-4]). Root-cause narrowing done this pass
      (see docs/notes below); still OPEN — not yet fixed.
- [ ] TGV central/upwind — reproduced historical central relL2=2.18e-3,
      upwind=5.64e-3 (central < upwind/4 fails), refinement orders
      [1.69, 1.84] at levels (12,16,24); extended to level 32 → order 1.81
      (plateaus ~1.8, does not approach 2). Same investigation as ABC (shared
      root-cause family: cyclic-boundary central PIMPLE cases specifically).
- [ ] TGV refinement — see above, same finding.
- [x] full non-MPI fast FVM gate green after fixing the case-sensitivity bug below:
      `pytest tests/fvm -m "(unit or verification) and not slow and not mpi"`
      → 390 passed (was: 2 collection errors before the fix).
  - Fixed: `tests/fvm/test_adaptive_cartesian_mesh.py`,
    `tests/fvm/test_triangulated_surface.py`,
    `tests/fvm/test_rectilinear_compact_layout.py` hardcoded
    `tutorials/coupled_fvm_vpm/cube_flow/...` (lowercase); actual directory is
    `tutorials/coupled_FVM_VPM/cube_flow/...`. Invisible on macOS
    (case-insensitive FS) — real on Linux/CI. Commit: "Fix case-sensitive path
    bug in FVM tests referencing cube_flow assets".
  - [ ] Full (non-fast) `pytest tests/fvm -m "not mpi"` not yet run this pass.

### ABC/TGV spatial-order investigation — evidence so far (OPEN)

Both `test_validation_abc_flow.py` (3D, cyclic box, PIMPLE+BDF2+central) and
`test_validation_taylor_green.py` (2D, same family) show the same signature:
observed order plateaus around **1.7-1.85**, not 2.0, and does NOT improve
monotonically toward 2 with further mesh refinement (3D: 6/8/12→1.780,
8/12/16→1.671, pairwise 12→16 = 0.741 — refinement makes it *worse*, ruling
out "just not in the asymptotic range yet"). Ruled out this pass, with
evidence, as candidate causes:

1. **Temporal contamination.** Halving dt at fixed mesh barely moves the
   error (level 12, dt 0.005→0.0025→0.00125: err 4.340e-4→4.262e-4→4.202e-4,
   not a 4x/16x drop) — the temporal component is a small fraction of the
   total error at these dt.
2. **Insufficient outer-corrector convergence** (PLAN §10.5). Sweeping
   `n_outer_correctors` 2/4/8/16 changes the ABC order-study errors by <1e-5
   relative — solution is already outer-loop-converged; PIMPLE nonlinear
   iteration is not the bottleneck.
3. **Non-orthogonal correction.** `structured_box` meshes measure exactly
   `max_non_orthogonality_deg = 0.0` — the term is structurally inactive on
   these grids, so it cannot be the cause.
4. **Cyclic face interpolation weight.** Initially suspected (the generic
   boundary-face path in `mesh/geometry.py` hardcodes `face_weights=1.0`,
   which would be wrong for a cyclic pair). Checked
   `mesh/coupled.py::configure_cyclic_boundaries` — it *does* correctly
   overwrite `face_weights` for paired cyclic faces using the true periodic
   image distance; measured weights on the actual solver mesh are exactly
   0.5 at every cyclic face. Not the cause.

Positive lead: sweeping kinematic viscosity at fixed mesh (level 6/8/12,
`test_validation_abc_flow._run_abc`) shows order tracks **how
convection-dominated** the case is: nu=0.001 → order 1.755, nu=0.1
(baseline) → order 1.780, nu=2.0 (diffusion-dominated) → order 1.956 (i.e.
diffusion alone is essentially second order; convection/pressure-coupling in
the nonlinear momentum equation is what caps the order). The isolated linear
scalar-transport harness (`test_temporal_order.py`, frozen advecting
velocity, Picard-converged to 1e-12) gets clean orders with the *same*
`central` convection assembler, so the defect is specific to the full
nonlinear incompressible path (PIMPLE pressure-velocity coupling / Rhie-Chow
/ pressure gradient), not the convection interpolation formula in isolation.

Not yet checked (next steps, cheapest first): (a) isolate Rhie-Chow transient
+ pressure-gradient consistency with a tiny manufactured incompressible
problem per PLAN §10.4; (b) check whether the pressure field itself converges
at 2nd order on these cases (would separate velocity-only vs full p-U
coupling); (c) check convective skew-symmetry / kinetic-energy-conserving
form. Do NOT weaken the 1.8 threshold — root cause not yet found.

## D. LES
- [ ] WALE tensor unit tests
- [ ] TGV decay timing
- [ ] SGS budget
- Note: `docs/vpm-les-followup-2026-08-17.md` documents a *prior*, separate
  LES investigation (Mansfield dynamic-coefficient Germano-identity bug in
  `scripts/experiments/`, already fixed there) — that is VPM-LES SGS-model
  research, not the FVM WALE model this section covers. Not re-litigated.

## E. IBM
- [ ] cylinder slip
- [ ] square force
- [ ] wake deficit
- [ ] refinement
- [ ] wall-force certification

## F. Coupler
- [ ] BC identity audit
- [ ] pressure treatment
- [ ] time interpolation
- [ ] conservation
- [ ] flux
- [ ] threshold modes all accepted
- [ ] serial smoke
- [ ] MPI smoke
- [ ] restart

## G. VPM/VLM
- [x] Flat-plate spanwise loading (original project brief) — already fixed in
      a prior pass per `docs/AGENTS.md`: was a plotting/normalisation
      artifact, not a solver bug. Regression tests exist
      (`tests/vpm/test_vlm_standalone_lifting_line.py`,
      `test_vlm_loading_distribution.py`, `test_vlm_frame_equivalence.py`).
      Re-verify these still pass as part of the full `tests/vpm` gate (in
      progress this pass, see Environment notes).
- [ ] VPM strength-growth audit
- [ ] VLM circulation/particle-strength dimensional audit
- [ ] panel forces
- [ ] absorption
- [ ] GPU where available

## H. Tutorials
- [ ] FVM setup files polished
- [ ] VPM setup files polished
- [ ] Coupled setup files polished
- [ ] checkpoint names polished — `tutorials/VPM/flatPlate/setup_plate.py` still
      has `BACKUP_PERIOD` (should be `*_INTERVAL_TIME`); tracked under §B.
- [ ] sampler paths verified
- [ ] lightweight tutorials run
- [ ] coupled tutorials smoke-run
- [ ] native tutorial validation script

## I. Packaging/CI
- [ ] build
- [ ] twine
- [ ] isolated wheel
- [ ] pip check
- [ ] CI
- [ ] nightly
- [ ] platform support documented

## Environment notes (this session)
- Local checkout (`/Users/flaviomartins/OpenONDA`, via device bridge) is missing the
  `git-lfs` binary, which breaks `git status`/`git fetch` with a hard error unless
  LFS filters are disabled explicitly:
  `git -c filter.lfs.process= -c filter.lfs.smudge= -c filter.lfs.clean= status`.
  Without this, `git status` silently produces *no* output (the LFS filter error
  aborts before listing changes) — do not mistake that for a clean tree.
- Heavy iteration this pass used a throwaway clone + venv in a cloud sandbox
  (`pip install -e .`, Python 3.11.15, numpy 2.4.6, taichi 1.7.4, pyrefly 1.2.0,
  ruff 0.16.4, pytest 9.1.1) rather than the conda env on the local machine —
  faster for repeated `pytest`/`ruff`/`pyrefly` cycles. Needed system packages
  not preinstalled there: `libglu1-mesa`, `libopengl0`, `libxft2` (gmsh/vtk/pyvista
  runtime deps; their absence produced `OSError: libGLU.so.1`/`libXft.so.2`
  collection errors unrelated to solver logic).
- `tests/vpm -m "not gpu and not slow"` OOMs (SIGKILL) when run as one pytest
  process in an 8GB/2-core sandbox — Taichi CPU-backend JIT state accumulates
  across ~150+ solver instantiations. Worked around by running each test file
  in its own subprocess; not a repository bug, a constrained-sandbox artifact.
  Real dev/CI machines have not shown this.
- `openonda_bootstrap.py` has no git history in this repo (never tracked) and is not
  referenced by current tutorials — not a regression, ignore prior-memory references
  to it as stale.
- The local checkout has an unstaged deletion of `CONSOLIDATION_STATUS.md`
  (8-line chronological "commit X → commit Y" narrative, no lasting reference
  value per PLAN.md §1) and untracked `PLAN.md` (this pass's operating
  instructions, not meant to be versioned) plus this file. Committing the
  `CONSOLIDATION_STATUS.md` removal and this ledger as one "release cleanup"
  commit.
- `git status` on the local checkout also shows 3 binary files under
  `docs/dns/` and `docs/references/` as modified with no corresponding source
  edit — consistent with local git-lfs smudge/clean filter corruption (the
  binary missing above), not a real content change. Left untouched; do not
  `git add` these without git-lfs installed and the working tree re-verified
  clean against origin's LFS objects.
