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
pytest tests/fvm -m "(unit or verification) and not slow and not mpi and not openfoam"
# Coupler:
pytest tests/coupler -m "not mpi and not openfoam"
```

Slow validation physics and MPI/PETSc runs are exercised nightly, not on every
change (see `.github/workflows/`).

## Imported Claude Cowork project instructions

I am working on perfecting the VLM+VPM solver for the OpenONDA project. Currently, I notice that the flat-plate case (in tutorials/VPM/flatPlate) has an issue that I can't seen to fix: it does not show a reasonable match to the parabolic-like lift distribution, instead showing an almost constant lift that does not even drop to zero at the tips. I want you to find out why and fix it.
