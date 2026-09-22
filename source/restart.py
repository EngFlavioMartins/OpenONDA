"""Common start selection for native and coupled simulation lifecycles."""

import csv
from pathlib import Path
import re
import shutil
import tempfile

from source.solution_layout import vpm_backup_files


def select_backup(start_from, *, directory, kind, backup_path=None):
    """Select a committed backup, never a temporary file or visualization.

    ``None`` keeps the caller's in-memory state. ``initial`` requires a clean
    solution; ``latest`` starts at zero only when no committed backup exists.
    Explicit paths are passed to the strict native reader without fallback.
    """
    if start_from is None:
        return None
    if not isinstance(start_from, (str, Path)):
        raise TypeError("start_from must be 'latest', 'initial', a backup path, or None")
    if start_from not in ("latest", "initial"):
        return Path(start_from)
    directory = Path(directory)
    if kind == "vpm":
        candidates = vpm_backup_files(directory)
        latest = candidates[-1] if candidates else None
        if latest is not None:
            import h5py

            try:
                with h5py.File(latest, "r") as archive:
                    recorded = int(archive["solver"].attrs["step"])
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid latest VPM backup {latest}: {exc}") from exc
            named = int(latest.stem.removeprefix("vpm_"))
            if recorded != named:
                raise ValueError(
                    f"VPM backup {latest} records step {recorded}, but its filename says {named}"
                )
    elif kind == "fvm":
        target = Path(backup_path)
        if not target.is_absolute():
            target = directory / target
        latest = target if target.is_file() or (target / "manifest.json").is_file() else None
    elif kind == "coupled":
        target = directory / "backups"
        latest = target if (target / "manifest.json").is_file() else None
    else:
        raise ValueError(f"Unknown restart kind {kind!r}")
    previous_run = any((directory / "restart-branches").glob("run-before-*"))
    if start_from == "initial":
        if latest is not None or previous_run or any(directory.glob("*.pvd")):
            raise ValueError(
                "Existing backup found; run ./allclean.sh before starting from initial"
            )
        return None
    if latest is None:
        # An old visualization-only run cannot be reconstructed exactly. Do
        # not silently mix a new zero-time solution with that run's history.
        from defusedxml.ElementTree import parse

        if previous_run:
            raise FileNotFoundError(
                f"Previous output exists in {directory}, but no committed {kind} backup was "
                "found. Restore a backup or run ./allclean.sh for a fresh simulation."
            )
        for collection in directory.glob("*.pvd"):
            if any(
                float(item.attrib["timestep"]) > 0
                for item in parse(collection).findall(".//DataSet")
            ):
                raise FileNotFoundError(
                    f"Solution output exists in {directory}, but no committed {kind} backup "
                    "was found. Restore a backup or run ./allclean.sh for a fresh simulation."
                )
    return latest


def rewind_vpm_frames(directory, step, time):
    """Archive discarded native frames so discovery cannot revive a branch."""
    from source.solvers.fvm.io.solver_io import SolverIO

    directory = Path(directory)
    for component in ("vpm", "vlm"):
        SolverIO._rewind_pvd(directory / f"{component}.pvd", time)
    future = []
    for component in ("vpm", "vlm"):
        for folder in (directory / component, directory):
            for path in folder.glob(f"{component}_*"):
                match = re.fullmatch(rf"{component}_(\d+)\.(h5|vtu|vtp)(\.tmp)?", path.name)
                if match and int(match[1]) > step and path.is_file():
                    future.append(path)
    if future:
        root = directory / "restart-branches"
        root.mkdir(exist_ok=True)
        branch = Path(tempfile.mkdtemp(prefix="frames-before-", dir=root))
        for path in future:
            destination = branch / path.relative_to(directory)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, destination)


def output_has_time(directory, name, time):
    """Check a reconciled CSV/PVD stream for an already committed event."""
    directory = Path(directory)
    table = directory / f"{name}.csv"
    if table.is_file():
        with table.open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            if "time" not in (rows.fieldnames or ()):
                return False
            if any(abs(float(row["time"]) - time) <= 1e-12 for row in rows):
                return True
    collection = directory / f"{name}.pvd"
    if collection.is_file():
        from defusedxml.ElementTree import parse

        return any(
            abs(float(item.attrib["timestep"]) - time) <= 1e-12
            for item in parse(collection).findall(".//DataSet")
        )
    return False


def archive_run_metadata(path):
    """Keep the previous invocation's status before construction replaces it."""
    path = Path(path)
    if path.is_file():
        root = path.parent / "restart-branches"
        root.mkdir(exist_ok=True)
        branch = Path(tempfile.mkdtemp(prefix="run-before-", dir=root))
        shutil.copy2(path, branch / path.name)
