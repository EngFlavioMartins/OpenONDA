# Agent guide — OpenONDA

OpenONDA is a CFD library. The Python sources are:

- `source/solvers/fvm` — native incompressible finite-volume solver
- `source/solvers/vpm` — vortex-particle solver (Taichi/GPU DSL)
- `source/coupler` — FVM↔VPM coupler

This file is the single source of truth for how automated coding agents
(Claude Code, Codex/ChatGPT, and others) should work in this repo. `CLAUDE.md`
imports it.

## Type checking with Pyrefly — required for Python changes

Before completing any task that **creates or modifies Python files** under
`source/solvers/fvm` or `source/coupler`, you MUST:

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
- `source/solvers/vpm` is **excluded** from Pyrefly on purpose: its Taichi DSL is
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
pytest tests
```

The maintained suite is deliberately limited to the nomenclature/API gate,
one FVM restart contract, strict VPM state validation, and one coupled
FVM--VPM checkpoint round trip. Case studies, numerical sweeps, benchmarks,
and MPI-specific regressions are not part of the repository test suite.

## Imported Claude Cowork project instructions

I am working on perfecting the VLM+VPM solver for the OpenONDA project. Currently, I notice that the flat-plate case (in tutorials/vpm/flat_plate) has an issue that I can't seen to fix: it does not show a reasonable match to the parabolic-like lift distribution, instead showing an almost constant lift that does not even drop to zero at the tips. I want you to find out why and fix it.

### Flat-plate spanwise loading — resolution

Status: **fixed**. The near-constant spanwise lift with no tip drop was a
*plotting/post-processing artifact*, not a solver bug: the legacy spanwise
figures normalised the outermost **cell-centred** VLM stations to ±1 (wrongly
stretching them to the physical tips, so the tip-Γ→0 boundary condition never
appeared). The corrective conventions are:

- `assets/plot_plate_spanwise.py::load_spanwise_csv` closes the distribution
  with the finite-wing condition Γ(±b/2)=0 instead of stretching samples to
  ±1 (and reconstructs the physical y for legacy CSVs that were normalised to
  ±1). The taper remains a mesh-resolution property of the finite-wing model.

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
