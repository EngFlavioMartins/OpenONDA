# Flat-Plate VLM+VPM Spanwise-Loading Fix — Handoff Report

**Date:** 2026-08-13
**Author of this handoff:** OpenONDA coding agent session
**Audience:** a fresh AI agent taking over the VLM+VPM work

---

## 1. Original plan

The working instructions (from the Claude Cowork project context in `AGENTS.md`)
specified perfecting the VLM+VPM solver for OpenONDA. The concrete complaint to
resolve was:

> The flat-plate case (`tutorials/VPM/flatPlate`) does not show a reasonable
> match to the parabolic-like lift distribution; instead it shows an almost
> constant lift that does not even drop to zero at the tips.

The expected behaviour (from finite-wing lifting-line theory) is a spanwise
loading that is ~elliptic/parabolic shaped and tapers to zero at the wingtips.

Plan that emerged:

1. Reproduce the symptom and separate *solver-physics* from *presentation*.
2. Verify the solver's AIC / RHS / linear-solve path against an independent
   reference (numpy), at native solver precision.
3. Verify the spanwise loading against the project's own lifting-line reference
   (`tutorials/VPM/flatPlate/assets/theoretical_model.py`).
4. Identify the mechanism that erases the tip taper, fix it surgically without
   disturbing the validated coupled VLM+particle path.
5. Add a regression test and re-run the VLM gate.

---

## 2. What was done

### Earlier phases — VPM subsystem restructure (completed in a prior session)

- [x] Phase A — VPM core namespace split (`core/`, `physics/`, `numerics/`, `io/`,
      `initial_conditions/`, `config/`, `diagnostics/`)
- [x] Phase B — physics: diffusion, stretching (CS/ILS), divergence & Pedrizzetti
      relaxation
- [x] Phase C — numerics: treecode/LBVH, Fourier integrals, evaluation integrals
- [x] Phase D — stabilization manager (`core/`)
- [x] Phase E — FVM↔VPM coupler
- [x] Phase F — config/constants refactor (`_DVH_BETA`), types
- [x] Phase G — diagnostics/offline, sampler, runtime profiler
- [x] Phase H — dependency enforcement + `tests/vpm/test_architecture.py`
- [x] Final consolidated report

### This session — spanwise-loading investigation & fix

- [x] Reproduced the symptom. The stored legacy data and the current solver
      *both* taper correctly when handled with the right conventions; the
      "almost constant" symptom reproduces **only** in standalone
      (`coupled=False`) VLM mode. Coupled mode matched lifting-line throughout.
- [x] Ruled out AIC-assembly defects: `compute_AIC_matrix` matches an analytic
      numpy implementation **exactly** (probed entries `(447,440)`, `(895,840)`,
      `(0,0)` gave `-15.475`, `-0.106`, `-5.132` on both sides).
      The earlier discrepancy was a red herring caused by an experiment edit to
      `influence.py` (see Issues #4).
- [x] Ruled out solver/numerics: `linear_solvers.py` does not mutate the AIC
      (SCIPY solves on copies); f32-vs-f64 arithmetic on the far legs is not the
      problem (numpy f32 == f64 == taichi).
- [x] Verified the tutorial-coupled pipeline end-to-end (isolated `advance_coupled`
      trace, static wind-frame, CPU, 150 steps): `cl` root 0.771 → 0.647 (0.768
      semi-span) → 0.291 (0.984) — matches lifting-line at every station.
- [x] Fixed a **presentation/closure artifact** that made legacy plots look too
      full near the tips:
      - `loading_distribution.py`: spanwise stations now map strictly inside
        ±1 using physical span edges (`span_edge_min/max`), instead of the old
        outer-cell-centre-==-tip convention.
      - plot script prep closes the curve with Γ=0 at ±1 before plotting.
- [x] **Root cause of the standalone bug isolated.** `solve()` calls
      `update_trailing_directions`, which re-aims the far-wake legs along the
      freestream direction. In the wind-frame flat plate that direction carries
      the angle-of-attack vertical component (up 8°), so the far legs climb
      ≈1400 chords out of the wing plane, suppressing the tip downwash and
      erasing the tip taper. Proof: same solver, only the far-leg angle changed
      (NS=56, α=8°, tutorial geometry) — horizontal legs give tip Γ/Γ_root
      0.1818 (= lifting line); slanted legs give 0.3797.
- [x] **Fix implemented** in `mesh.py`: the far trailing points
      (`vortex_points[:,0]` and `[:,3]`) are now placed along the freestream
      direction **projected onto the wing tangent plane**
      (`d_plane = trail_dir − (trail_dir·n̂)n̂`). `trailing_dirs` (used for VPM
      particle shedding) and the coupled AIC (which reads corners, not far
      points) are untouched, so the validated coupled path is unchanged.
- [x] Verified the fix:
      - standalone tip ratio: NS=14 → 0.3556 (was 0.4582); NS=28 → 0.2555
        (was 0.4096, coupled reference 0.2532); NS=56 → 0.1818 (was 0.3797,
        ~lifting line).
      - coupled unchanged: NS=14 0.370 vs 0.367, NS=28 0.253 vs 0.252,
        NS=56 0.171 vs 0.171 (pre-fix values in parens).
- [x] Added regression test
      `tests/vpm/test_vlm_loading_distribution.py::test_standalone_far_wake_lies_in_wing_plane_tip_taper`
      (marked `@pytest.mark.verification`): NS=28 tutorial-config flat plate,
      asserts `standalone_tip <= coupled_tip * 1.25` and `standalone_tip < 0.35`
      (pre-fix 0.41 fails; post-fix 0.2555 passes). Anchors against the coupled
      solve at identical resolution/stations to sidestep point-vs-cell-averaged
      lifting-line conventions near the tip.
- [x] Re-ran the gate: `pytest tests/vpm -m "(unit or verification) and not slow
      and not mpi"` → **80 passed** (79 prior + 1 new).

---

## 3. Files changed in this session

| File | Change |
|---|---|
| `source/solvers/VPM/boundary_elements/vlm/solver/mesh.py` | **The fix.** `_update_trailing_uniform_kernel` and `_update_trailing_local_kernel` now take `normals` and place far points along the wing-plane-projected trailing direction. Callers pass `lattice.normals`. |
| `tests/vpm/test_vlm_loading_distribution.py` | Added the lifting-line tip-taper regression test (marked `verification`); added `math`/`VLMMeshSetup` imports. |
| `source/solvers/VPM/boundary_elements/vlm/solver/influence.py` | **No change.** Restored to committed state (see Issues #4). |
| `source/solvers/VPM/boundary_elements/vlm/solver/loading_distribution.py` | (Prior session) span-edge mapping fix for y_over_b. |

---

## 4. Issues and open questions

1. **Standalone far-wake slant (FIXED).** `update_trailing_directions` re-aims
   the far legs along the full relative-velocity vector; the AoA vertical
   component lifts them out of the wing plane and suppresses tip downwash.
   Fixed by wing-plane projection (Section 2). Note the same far points feed
   `compute_induced_velocities` (Cp via `vortex_ring_velocity`) — previously
   inconsistent with the AIC geometry, now consistent.

2. **Static-α=5° result missing from the tutorial outputs.**
   `tutorials/VPM/flatPlate/solution/` contains `exp_static_aoan02/aoan05/aoan10`
   and `exp_moving_aoa00/02/05/08/...` but **no `exp_static_aoa05`**. The plot
   script (`assets/plot_plate_spanwise.py`) compares the moving aoa05 case
   against a static reference and must degrade gracefully/be pointed at an
   existing case when `exp_static_aoa05` is absent. Verify the plot still
   renders for the current solution set.

3. **Coupled near-tip sensitivity to mesh resolution + wake_offset.**
   In an *unrefined* low-resolution mesh (NS=12 uniform, no geometric tip
   refinement) with `dt=None` (wake_offset=0), even coupled mode showed a
   too-full outer station (tip Γ/root 0.87). The validated LL-matching regime
   is the tutorial configuration: refined tip mesh
   (`VLMMeshSetup.geometric(ratio=4.0, region="end")`) + `dt`-based
   one-step wake offset (U_ref·dt). If a user runs coupled at coarse/unrefined
   resolution or with `dt=None`, expect the outer-station ratio to be higher
   than lifting-line — worth documenting or hardening later, but NOT a defect
   of the solved bug.

4. **AIC "mismatch" red herring — keep `influence.py` pristine.** Earlier probes
   seemed to show `compute_AIC_matrix` disagreeing with numpy. This was caused
   by an experiment edit to `influence.py` (a sign-broken
   `horseshoe_semi_infinite_velocity` standalone branch that made the two far
   legs cancel). That edit was reverted (`git checkout -- influence.py`); the
   loaded kernel matches numpy exactly. Do **not** re-introduce a
   `horseshoe_semi_infinite_velocity` call for the standalone AIC — the wrapper
   in `kernels/biot_savart.py` is sign-broken for this horseshoe orientation.
   A copy of the broken experiment is saved at
   `/var/folders/kw/.../opencode/influence.patch` — safe to delete.

5. **Lifting-line comparison conventions near the tip.** The VLM outer-station
   Γ/Γ_root is a **cell-averaged** statistic; the project's
   `liftingline_circulation` (Anderson, Eq. 5.59, `μ = a0·c/(4·b)`) gives point
   values that drop much faster near the tip (e.g. NS=28 outer center 4.9374:
   point 0.322, cell-average 0.301 vs solver 0.253). Any future assertion
   against the analytic lifting line must say *which* station statistic it
   compares. The regression test deliberately uses coupled-vs-standalone at
   identical resolution instead. (Also: a naive Fourier lifting-line solver I
   wrote during investigation disagreed with the project model — always use
   `theoretical_model.liftingline_circulation` as the reference.)

6. **Operational constraints discovered this session.**
   - Taichi kernels **cannot be compiled from stdin/heredoc** scripts
     (`OSError: could not get source code` / bpy / "Not in Blender environment")
     — all kernel probes must be written to real files.
   - `timeout` and `rg` binaries are unavailable in this shell; use
     `python -m pytest` and grep/glob tools.
   - Pre-existing pyrefly baseline (~240 errors) untouched; `source/solvers/VPM`
     is excluded from Pyrefly by design — no pyrefly required for these edits.
   - pre-commit is report-only (flags, never rewrites).

7. **Other uncommitted changes in the tree (prior phases).** `git status` shows
   modifications not yet committed: `openonda/vpm.py`,
   `source/solvers/VPM/boundary_elements/panels/solver/loading_distribution.py`,
   `source/solvers/VPM/boundary_elements/vlm/solver/diagnostics.py`,
   `source/solvers/VPM/boundary_elements/vlm/solver/loading_distribution.py`,
   `source/solvers/VPM/boundary_elements/vlm/solver/mesh.py` (this fix),
   `source/solvers/VPM/config/constants.py`, `config/types.py`,
   `core/solver.py`, `diagnostics/__init__.py`, and a deletion of
   `source/solvers/VPM/diagnostics/fourier_integrals.py`.
   These should be reviewed and committed by the user/next agent;
   nothing in this session was committed.

8. **Diagnostic scratch files.** A set of probe scripts lives under
   `/var/folders/kw/.../opencode/` (`probe_slant.py`, `probe_solve_path.py`,
   `probe_aic_file.py`, `probe_legs_lattice.py`, `verify_fix2.py`,
   `probe_cellavg.py`, etc.) — investigation artifacts, safe to delete or keep
   as reference. Not part of the repo.

---

## 5. Suggested next steps for the receiving agent

- [ ] Review/un-commit the tree state (Issue #7) and the `mesh.py` diff.
- [ ] Re-run the flatPlate tutorial (coupled) end-to-end and regenerate the
      spanwise plot to visually confirm the taper; verify the missing
      `exp_static_aoa05` handling (Issue #2).
- [ ] Re-run gate: `python -m pytest tests/vpm -p no:cacheprovider
      -m "(unit or verification) and not slow and not mpi"` (expect 80 passed).
- [ ] Optionally document Issue #3 (coarse-mesh / dt=None coupled behaviour)
      and Issue #5 (station-statistic conventions) in the tutorial docs.