"""Canonical on-disk layout for solver visualization artifacts.

The solution root is intentionally small: a user opens one ParaView collection
per physical representation there, while immutable frame files live in a
component directory. Restart persistence has its own lifecycle and is not part
of this visualization layout.
"""

from __future__ import annotations

from pathlib import Path

_COMPONENTS = frozenset(("fvm", "vpm", "vlm"))


def component_directory(solution_directory: str | Path, component: str) -> Path:
    """Return the canonical frame directory for one solution component.

    Parameters
    ----------
    solution_directory : str or pathlib.Path
        Case-local solution root.
    component : {"fvm", "vpm", "vlm"}
        Physical representation that owns the frames.

    Returns
    -------
    pathlib.Path
        ``<solution_directory>/<component>``.  The directory is not created.

    Raises
    ------
    ValueError
        If ``component`` is not a supported solution representation.
    """
    if component not in _COMPONENTS:
        raise ValueError(f"Unsupported solution component {component!r}")
    return Path(solution_directory) / component


def collection_path(solution_directory: str | Path, component: str) -> Path:
    """Return the root-level ParaView collection path for one component."""
    if component not in _COMPONENTS:
        raise ValueError(f"Unsupported solution component {component!r}")
    return Path(solution_directory) / f"{component}.pvd"
