# Agent guide — OpenONDA

OpenONDA is a CFD library. The Python sources are:

- `source/solvers/FVM` — native incompressible finite-volume solver
- `source/solvers/VPM` — vortex-particle solver (Taichi/GPU DSL)
- `source/coupler` — FVM↔VPM coupler
- `source/utilities` — shared helpers

This file is the single source of truth for how automated coding agents
(Claude Code, Codex/ChatGPT, and others) should work in this repo. `CLAUDE.md`
imports it.

## Type checking with Pyrefly — required for Python changes

Before completing any task that **creates or modifies Python files** under
`source/solvers/FVM`, `source/coupler`, or `source/utilities`, you MUST:

1. Run `pyrefly check` at the repo root — or `pyrefly check <the files you
   changed>` to focus on your edit.
2. Fix every **new** type error your change introduced. A file must not leave
   your edit with more errors than it had before you touched it.
3. Run `pyrefly check` again and confirm you did not grow the baseline.

Notes that make this workable:

- There is a **pre-existing baseline** (~240 errors in the maintained tree) from
  numpy/numba/scipy stub strictness mixed with some genuine `Optional`-handling
  bugs. You are **not** required to clear pre-existing errors — only to avoid
  adding new ones. Fixing a real one (e.g. an un-narrowed `None` that pyrefly
  flags as "not subscriptable") in code you are already editing is welcome.
- `source/solvers/VPM` is **excluded** from Pyrefly on purpose: its Taichi DSL is
  not statically typeable (the same reason `[tool.ty]` and `[tool.mypy]` suppress
  it). Do not try to type-check it.
- Scope and settings live in `[tool.pyrefly]` in `pyproject.toml`.

## Formatting & lint

pre-commit is **report-only** — it flags problems but never rewrites files, so it
cannot deadlock a commit. Apply fixes yourself:

```bash
ruff check --fix source tests    # lint fixes
ruff format source tests         # formatting
```

## Tests

```bash
# Fast physics-correctness gate (what CI blocks on):
pytest tests/fvm -m "(unit or verification) and not slow and not mpi"
# Coupler:
pytest tests/coupler -m "not mpi"
```

Slow validation physics and MPI/PETSc runs are exercised nightly, not on every
change (see `.github/workflows/`).

## Imported Claude Cowork project instructions

I am working on perfecting the VLM+VPM solver for the OpenONDA project. Currently, I notice that the flat-plate case (in tutorials/VPM/flatPlate) has an issue that I can't seen to fix: it does not show a reasonable match to the parabolic-like lift distribution, instead showing an almost constant lift that does not even drop to zero at the tips. I want you to find out why and fix it.

### Flat-plate spanwise loading — resolution

Status: **fixed**. The near-constant spanwise lift with no tip drop was a
*plotting/post-processing artifact*, not a solver bug: the legacy spanwise
figures normalised the outermost **cell-centred** VLM stations to ±1 (wrongly
stretching them to the physical tips, so the tip-Γ→0 boundary condition never
appeared). The corrective conventions are:

- `assets/plot_plate_spanwise.py::load_spanwise_csv` closes the distribution
  with the finite-wing condition Γ(±b/2)=0 instead of stretching samples to
  ±1 (and reconstructs the physical y for legacy CSVs that were normalised to
  ±1). The taper is a genuine mesh-resolution phenomenon, verified by
  `tests/vpm/test_vlm_standalone_lifting_line.py` (incl. the
  `test_coupled_tip_taper_depends_on_mesh_resolution_not_dt` regression that
  proves **tip taper depends on spanwise mesh density, not on dt**).
- `tests/vpm/test_vlm_loading_distribution.py` guarantees spanwise+chordwise
  station sums reproduce the lattice total loads (guards the almost-constant
  re-appearance), and `tests/vpm/test_vlm_frame_equivalence.py` checks the
  body/wind-frame static equivalence the tutorial relies on.

Tutorial runnability notes (all applied):
- `setup_plate.py` uses `compute_device="AUTO"` (the previous hard-coded
  `VULKAN` is not available on macOS; AUTO selects METAL there, VULKAN on
  Linux/Windows).
- Every `assets/plot_*.py` (incl. `plot_flat_plate_kelvin.py`) degrades
  gracefully with a `[MISSING]`/skip message when a sample CSV is absent, so
  `allplot.sh` never hard-fails on a partial solution set.
- Data layout is consistent across code and scripts via
  `resolve_samples_dir` (`samples/` lifted next to a dir literally named
  `solution/`).
