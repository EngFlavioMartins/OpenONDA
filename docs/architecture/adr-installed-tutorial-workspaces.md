# ADR: installed tutorial workspaces

## Status

Accepted.

## Context

OpenONDA's solver modules were installable, but tutorial launchers and plotting
assets existed only in a repository checkout. Some launchers imported the
top-level `tutorials` package and plotters resolved a theme below `docs/`, so
copying a single case or invoking it outside the repository failed. Running a
case directly from an installed wheel would also write large solver output into
`site-packages`, which may be read-only and must remain immutable.

Shipping all repository data is not viable: generated samples, animations,
ParaView sessions, papers, and obsolete mesh dumps add hundreds of megabytes
and are not inputs to the maintained solver configurations.

## Decision

The wheel includes the public `openonda` and `source` packages, the maintained
tutorial source packages, compact geometry/reference inputs, and the shared
Matplotlib theme and font. The `openonda` command exposes tutorial discovery,
materialization, execution, plotting, and cleaning.

A tutorial is copied into a workspace that preserves the repository-relative
layout:

```text
workspace/
  docs/themes/
  tutorials/<family>/<case>/
```

Launchers always run in that case directory. The CLI selects the interpreter
that owns the installed command through `OPENONDA_PYTHON` and `PATH`, and gives
Matplotlib a writable workspace cache. Existing materialized cases are never
overwritten. Large results and presentation-only artifacts are excluded from
the distribution.

LaTeX is an optional plotting enhancement. The shared theme enables it only
when both `latex` and `dvipng` are available; otherwise normal Matplotlib math
text is used.

## Consequences

- Imports, docstrings, editor type information, tutorials, and plotting work
  without `PYTHONPATH` or a repository clone on supported macOS/Linux systems.
- Tutorial code is visible and editable in the user workspace.
- Installed package files remain immutable and reproducible.
- A generated workspace owns potentially large result data and can be archived
  independently.
- Compact source/input resources are release artifacts and are verified from a
  clean wheel before release.
