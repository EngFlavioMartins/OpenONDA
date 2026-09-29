"""Common start selection for native and coupled simulation lifecycles."""

import csv
from pathlib import Path
import re
import shutil
import tempfile

from source.solution_layout import vpm_backup_files


def select_backup(start_from, *, directory, kind, backup_path=None):
    """Select a committed backup, never a temporary file or visualization.

    ``None`` keeps the caller's in-memory state. ``initial`` ignores all saved
    states; ``latest`` starts at zero only when no committed backup exists.
    Explicit paths are passed to the strict native reader without fallback.
    """
    if start_from is None:
        return None
    if not isinstance(start_from, (str, Path)):
        raise TypeError("start_from must be 'latest', 'initial', a backup path, or None")
    if start_from == "initial":
        return None
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
    return latest


def reset_run_outputs(
    directory, *, kind, backup_path=None, samples_dir=None, owned_sample_names=()
):
    """Retire an old output series before a fresh initial run.

    Move only native artifacts and named sampler streams into the existing
    restart archive, without opening checkpoints or requiring their validity.
    This prevents a longer old VPM run from winning subsequent latest-backup
    discovery. Current constructor metadata, open logs and unrelated files
    remain in place. Call on the writing rank before publishing initial output.
    No additional restart marker or backup format is introduced.
    """
    if kind not in {"fvm", "vpm", "coupled"}:
        raise ValueError(f"Unknown restart kind {kind!r}")
    directory = Path(directory).resolve()
    artifacts = {}
    components = {"fvm": ("fvm",), "vpm": ("vpm", "vlm"), "coupled": ()}[kind]
    for component in components:
        index = directory / f"{component}.pvd"
        if index.exists():
            artifacts[index] = Path("solution") / index.name
        pattern = re.compile(
            rf"{component}_\d+(?:-rank-\d+|_particles)?\.(?:h5|vtu|vtp|pvtu)(?:\.tmp)?$"
        )
        for folder in (directory, directory / component):
            for path in folder.glob(f"{component}_*"):
                if path.is_file() and pattern.fullmatch(path.name):
                    artifacts[path] = Path("solution") / path.relative_to(directory)
    journals = {
        "coupled": ("coupler_diagnostics.jsonl",),
        "fvm": ("diagnostics.jsonl", "performance.jsonl"),
        "vpm": (),
    }[kind]
    for name in journals:
        path = directory / name
        if path.exists():
            artifacts[path] = Path("solution") / name
    target = None
    if kind == "coupled":
        target = directory / "backups"
    elif kind == "fvm" and backup_path is not None:
        target = Path(backup_path)
        target = (target if target.is_absolute() else directory / target).resolve()
    if target is not None and target.exists():
        if directory.is_relative_to(target):
            raise ValueError("The backup path must not contain the solution directory")
        artifacts[target] = Path("backup") / target.name
    if samples_dir is not None:
        samples = Path(samples_dir).resolve()
        for name in owned_sample_names:
            for suffix in (".csv", ".pvd"):
                path = samples / f"{name}{suffix}"
                if path.is_file():
                    artifacts[path] = Path("samples") / path.relative_to(samples)
            frame_pattern = re.compile(rf"{re.escape(str(name))}_\d+\.(?:vts|vtu|vtp|csv)$")
            for path in samples.glob(f"{name}_*"):
                if path.is_file() and frame_pattern.fullmatch(path.name):
                    artifacts[path] = Path("samples") / path.relative_to(samples)
    # Retain prior constructor metadata with the retired output series.
    for path in (directory / "restart-branches").glob("run-before-*"):
        artifacts[path] = Path("metadata") / path.name
    if not artifacts:
        return
    archive = directory / "restart-branches"
    archive.mkdir(parents=True, exist_ok=True)
    branch = Path(tempfile.mkdtemp(prefix="initial-before-", dir=archive))
    for path, relative in artifacts.items():
        destination = branch / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, destination)


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
