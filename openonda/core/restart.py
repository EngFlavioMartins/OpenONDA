"""Common start selection for native and coupled simulation runs."""

import csv
from pathlib import Path
import re
import shutil

from openonda.core.file_output import TIME_TOLERANCE
from openonda.core.solution_layout import (
    coupled_backup_directories,
    fvm_backup_files,
    vpm_backup_files,
)
from openonda.core.vtk_output import read_collection, rewind_collection


def select_backup(start_from, *, directory, kind):
    """Select a complete backup, never a temporary file or visualization.

    ``None`` keeps the caller's in-memory state. ``initial`` ignores all saved
    states; ``latest`` returns the numbered backup with the highest step, or
    ``None`` when there is none. Explicit paths are passed to the strict
    native reader without fallback.
    """
    if start_from is None:
        return None
    if not isinstance(start_from, (str, Path)):
        raise TypeError("start_from must be 'latest', 'initial', a backup path, or None")
    if start_from == "initial":
        return None
    if start_from != "latest":
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
        candidates = fvm_backup_files(directory)
        latest = candidates[-1] if candidates else None
    elif kind == "coupled":
        candidates = coupled_backup_directories(directory)
        latest = candidates[-1] if candidates else None
    else:
        raise ValueError(f"Unknown restart kind {kind!r}")
    return latest


def reset_run_outputs(directory, *, kind, samples_dir=None, sample_names=()):
    """Delete an old output series before a fresh initial run.

    Only native output files, numbered backups, journals, the coupled backup
    directory and the named sampler streams are removed, without opening
    backups.
    A longer old run therefore cannot win a later latest-backup discovery.
    Current constructor metadata, open logs and unrelated files remain. Call
    on the writing rank before writing initial output.
    """
    if kind not in {"fvm", "vpm", "coupled"}:
        raise ValueError(f"Unknown restart kind {kind!r}")
    directory = Path(directory).resolve()
    output_files = []
    components = {"fvm": ("fvm",), "vpm": ("vpm", "vlm"), "coupled": ()}[kind]
    for component in components:
        output_files.append(directory / f"{component}.pvd")
        pattern = re.compile(
            rf"{component}_\d+(?:-rank-\d+|_particles)?\.(?:h5|npz|vtu|vtp|pvtu)(?:\.tmp)?$"
        )
        for folder in (directory, directory / component):
            output_files.extend(
                path for path in folder.glob(f"{component}_*") if pattern.fullmatch(path.name)
            )
    journals = {
        "coupled": ("coupler_diagnostics.jsonl",),
        "fvm": ("diagnostics.jsonl", "performance.jsonl"),
        "vpm": (),
    }[kind]
    output_files.extend(directory / name for name in journals)
    if samples_dir is not None:
        samples = Path(samples_dir).resolve()
        for name in sample_names:
            output_files.extend(samples / f"{name}{suffix}" for suffix in (".csv", ".pvd"))
            frame_pattern = re.compile(rf"{re.escape(str(name))}_\d+\.(?:vts|vtu|vtp|csv)$")
            output_files.extend(
                path for path in samples.glob(f"{name}_*") if frame_pattern.fullmatch(path.name)
            )
    for path in output_files:
        if path.is_file():
            path.unlink()
    backup_directories = []
    if kind == "coupled":
        backup_directories.append(directory / "coupled")
    elif kind == "fvm":
        pattern = re.compile(r"fvm_\d{6,}")
        backup_directories.extend(
            path
            for path in (directory / "fvm").glob("fvm_*")
            if pattern.fullmatch(path.name) and path.is_dir()
        )
    for path in backup_directories:
        if path.is_dir():
            shutil.rmtree(path)


def rewind_vpm_frames(directory, step, time):
    """Delete output frames beyond the restored step and flow time."""
    directory = Path(directory)
    for component in ("vpm", "vlm"):
        rewind_collection(directory / f"{component}.pvd", time)
    for component in ("vpm", "vlm"):
        for folder in (directory / component, directory):
            for path in folder.glob(f"{component}_*"):
                match = re.fullmatch(rf"{component}_(\d+)\.(h5|vtu|vtp)(\.tmp)?", path.name)
                if match and int(match[1]) > step and path.is_file():
                    path.unlink()


def output_has_time(directory, name, time):
    """Check whether a saved CSV/PVD time series contains the sampling time."""
    directory = Path(directory)
    table = directory / f"{name}.csv"
    if table.is_file():
        with table.open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            if "time" not in (rows.fieldnames or ()):
                return False
            if any(abs(float(row["time"]) - time) <= TIME_TOLERANCE for row in rows):
                return True
    return any(
        abs(frame_time - time) <= TIME_TOLERANCE
        for frame_time, _ in read_collection(directory / f"{name}.pvd")
    )
