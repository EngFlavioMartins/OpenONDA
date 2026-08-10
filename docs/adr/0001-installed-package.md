# ADR 0001: Installed Python package

Status: accepted, 2026-07-30

## Decision

Normal OpenONDA users install a wheel through pip or the Conda installer.
Contributors may use a PEP 660 editable installation. Tutorials import the
public `openonda` package and run with the active environment's `python`.

Repository-path injection, interpreter discovery, and automatic Conda
re-execution are not part of case files.

## Consequences

- Installed imports work from any directory and do not depend on the checkout.
- A normal source change requires reinstalling; contributors use `pip install
  -e ".[dev]"` when immediate source reflection is desired.
- Conda users activate `OpenONDA` before running a case.
