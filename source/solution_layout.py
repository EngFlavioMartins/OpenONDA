"""Canonical on-disk layout and discovery for solver output artifacts.

The solution root is intentionally small: a user opens one ParaView collection
per physical representation there, while immutable frame files live in a
component directory. Native VPM checkpoints accompany those frames; coupled
restart bundles retain their own lifecycle.
"""

from __future__ import annotations

from pathlib import Path
import re

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


def vpm_backup_files(directory: str | Path) -> list[Path]:
    """Find native checkpoints in a solution root or an explicit frame directory.

    Prefer the canonical ``vpm/`` component when present. Otherwise accept a
    frame directory directly (including older flat archives). Never recursively
    combine separate cases or mix stale root-level files into a current series.
    Step widths are a minimum of six digits, as in the solver's ``:06d`` writer.
    """
    directory = Path(directory)
    component = component_directory(directory, "vpm")
    frames = component if component.is_dir() else directory
    pattern = re.compile(r"^vpm_(\d{6,})\.h5$")
    records = []
    for path in frames.glob("vpm_*.h5"):
        match = pattern.fullmatch(path.name)
        if match and path.is_file():
            records.append((int(match[1]), path))
    return [path for _, path in sorted(records)]
