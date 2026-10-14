"""Standard on-disk layout and discovery for solver output files.

The solution root is intentionally small: a user opens one ParaView collection
per physical representation there, while immutable frame files live in a
component directory. Numbered FVM and VPM backups accompany those frames;
coupled restart files are saved separately.
"""

from __future__ import annotations

from pathlib import Path
import re

_COMPONENTS = frozenset(("fvm", "vpm", "vlm"))


def component_directory(solution_directory: str | Path, component: str) -> Path:
    """Return the standard frame directory for one solution component.

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
    """Find native checkpoints in the solution root's ``vpm/`` component.

    Never recursively combine separate cases or mix root-level files into
    the component series.
    Step widths are a minimum of six digits, as in the solver's ``:06d`` writer.
    """
    directory = Path(directory)
    component = component_directory(directory, "vpm")
    frames = component
    pattern = re.compile(r"^vpm_(\d{6,})\.h5$")
    records = []
    for path in frames.glob("vpm_*.h5"):
        match = pattern.fullmatch(path.name)
        if match and path.is_file():
            records.append((int(match[1]), path))
    return [path for _, path in sorted(records)]


def fvm_backup_files(directory: str | Path) -> list[Path]:
    """Find complete numbered backups in the solution root's ``fvm/`` component.

    A serial backup is ``fvm_{step:06d}.npz``; a partitioned backup is a
    directory of that name holding ``checkpoint_info.json``. The list is
    sorted by step.
    """
    pattern = re.compile(r"^fvm_(\d{6,})(\.npz)?$")
    records = []
    for path in component_directory(directory, "fvm").glob("fvm_*"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        complete = (
            path.is_file()
            if match[2]
            else path.is_dir() and (path / "checkpoint_info.json").is_file()
        )
        if complete:
            records.append((int(match[1]), path))
    return [path for _, path in sorted(records)]


def coupled_backup_directories(directory: str | Path) -> list[Path]:
    """Find complete numbered coupled backups in the solution root's ``coupled/`` directory."""
    pattern = re.compile(r"^coupled_(\d{6,})$")
    records = []
    for path in (Path(directory) / "coupled").glob("coupled_*"):
        match = pattern.fullmatch(path.name)
        if match and (path / "checkpoint_info.json").is_file():
            records.append((int(match[1]), path))
    return [path for _, path in sorted(records)]
