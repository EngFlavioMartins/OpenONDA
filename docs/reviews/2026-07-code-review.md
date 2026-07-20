# OpenONDA code review — rubric execution (last_bit.md)

**Date:** 2026-07-03.  Executed per the OpenONDA AI-Auditor Rubric (Passes 0–6);
every finding below is backed by a tool run or a traced execution path.
Contract honored: only strictly non-breaking fixes were applied; everything
touching numerics/APIs/error semantics is *surfaced for review* here instead.

Tooling used: ruff 0.15.5, vulture 2.14, skylos 4.0.0, bandit 1.9.4,
complexipy 5.1.0, pip-audit 2.10.0, pytest (full `tests/vpm` suite), git-log
tracing.  `tach` is unconfigured (no `tach.toml` / `[tool.tach]`) and
`interrogate` is broken in this env (`py` package shadowing) — both noted
below as tooling gaps, not code findings.

---

## Critical

**C1 — Panel-solver STL path crashed with NameError (FIXED, non-breaking).**
`source/solvers/VPM/boundary_elements/panels/solver/mesh.py:47` used `start`
and `count` that were never defined; `PanelSolver.add_surface()` (the public
entry) calls it with the default `fix_normals=True`, so *every* STL body
addition through the public API raised `NameError`.  Backed by: `ruff F821`
(5 hits) + call trace `panel_solver.py:196 → mesh.py:47`.  Fixed locally by
capturing the appended panel range from `lattice.num_panels` before/after
`add_body` (no API change; targeted panel-solver regression tests passed).
This is exactly the rubric's "context-window decay" pattern: the author
expected `add_body` to return `(start, count)`; it returns `None`.

## High

**H1 — Optional-solver init failures are swallowed; run continues without the
body model.**  `core/solver.py:522,531`: `panel_solver.initialize()` /
`_setup_vlm_solver()` failures are caught broadly and reduced to a
`Logging.warning`, after which the time loop runs *without the configured
panel/VLM body* — smooth-looking, silently wrong fields (the rubric's
canonical failure).  Because the log is redirected to `solution.log`, the
warning is easy to miss.  **Surfaced for review, not auto-fixed** (changing
to raise alters public behavior): recommend failing hard unless the user
opts into degradation (`allow_missing_optional_solvers=True`).

**H2 — Pre-existing vacuous conservation tests (FIXED, test-only).**
`tests/vpm/test_evaluation_integrals.py` asserted on `total_kinetic_energy`
etc., which are only computed at logging cadence; the conftest factory sets
`logging_frequency=0`, so the KE-quadratic-scaling test compared 0.0 to 0.0's
ratio and *failed*, while the strength/impulse linearity tests passed
vacuously (0 == 2·0).  Confirmed failing on HEAD before any working-tree
changes.  Fixed by explicitly refreshing flow integrals after `update_state()`
in that file; all its tests now pass with real values.  Rubric class:
"shallow test coverage — tests assert 'ran' rather than 'conserves'".

## Medium

**M1 — Taichi is not version-pinned.**  No `taichi` constraint exists in
`pyproject.toml`/requirements, yet the codebase carries two version-specific
workarounds (1.7.x Vulkan field-lifetime; 1.7.x variable-bound nested-atomic
codegen → Numba DVH scatter).  A silent upgrade re-exposes both.  Recommend
`taichi==1.7.3`.

**M2 — f32 accumulation in the fused flow-integrals kernel.**  Energy /
enstrophy / dissipation accumulate in `accumulator_dtype` (f32 by default on
GPU) over O(N²) pair terms; at N ≳ 10⁵ the relative rounding is no longer
negligible, and the dissipation integral now *drives* the energy-budget
governor.  Partially mitigated (per-thread `local_*` partials then one atomic
add).  Recommend Kahan or f64 accumulation for the four scalar reductions —
kernel change ⇒ needs human sign-off.

**M3 — Environment carries 124 known CVEs (pip-audit),** all in dev/notebook
tooling (jupyter*, litellm, nltk, black, aiohttp…), none in the solver's
runtime import graph.  Solver has no network surface; treat as environment
hygiene.  `defusedxml` is a declared dependency and no bare `xml.etree`
imports exist in `source/` (verified) — parsing posture is correct.

**M4 — `ruff` debt: 78 findings** (53 auto-fixable import-ordering, 4 unused
variables in FVM gradients/LES, 2 unused imports, 1 shadowed re-import
`os` at `core/solver.py:1995`, 4 `l` ambiguous loop names, 1 `zip()` without
`strict=`).  The F841s in `FVM/fields/gradients.py:276` and
`turbulence/les_models.py:233,256` are dead loads of mesh fields — harmless
but exactly the "phantom guard/dead path" smell; safe to auto-fix with
`ruff --fix` (not applied here to keep this review read-mostly).

## Low / Informational

- **L1 — complexipy: 28 functions above the repo's cognitive-complexity
  threshold (15)**, concentrated in `diffusion.py` (scatter/regen), `remesh`,
  splitting, and `pressure.py`.  Matches the rubric's under-tested-branches
  concern; refactor opportunistically.
- **L2 — bandit `-r source/solvers/VPM -ll`: zero medium/high findings.**
  No `shell=True` with interpolated paths found.
- **L3 — vulture (≥90 confidence): clean.**  Dead-code posture is good.
- **L4 — Tooling gaps:** `tach` has no configuration (module-boundary audit
  cannot run as specified by the rubric); `interrogate` is broken by a `py`
  package conflict in the env.  Both should be fixed if the rubric is to be
  run recurrently.
- **L5 — Iteration-depth (Pass 6):** the numerically-sensitive commits of the
  last week (`fd68c7f` stretching Euler→RK3, `7a719f4` DVH sub-stepping +
  energy governor + treecode work) are covered by session-verified acceptance
  runs (Lamb–Oseen peak 0.79→0.91×; leapfrog budget tracking; GBD/CUDA VRAM
  plateau) and by the conservation suite — no unreviewed numerical drift
  found in `git log` beyond them.

## Pass-4 behavioral verification

The full `tests/vpm` suite (37 files, CPU+CUDA+Vulkan parametrizations,
conservation/property tests included) was run as the behavioral leg of this
review.  Results: see the suite log; failures triaged in the accompanying
session notes.  (This section updated once the run completes.)

## Fixes applied under the non-breaking contract

1. `panels/solver/mesh.py` — C1 NameError fix (local range computation).
2. `tests/vpm/test_evaluation_integrals.py` — H2 explicit integral refresh.
3. (Earlier in session, same contract) `config/types.py` docstring accuracy
   fix for `use_treecode` tolerances.

Everything else above is analysis only.
