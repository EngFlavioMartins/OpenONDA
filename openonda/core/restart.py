"""Common start selection for native and coupled simulation runs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import csv
from pathlib import Path
import re
import shutil

from openonda.core.clock import time_tolerance
from openonda.core.vtk_output import read_collection


def select_backup(start_from: str | Path | None, candidates: Sequence[Path]) -> Path | None:
    """Select a complete backup, never a temporary file or visualization.

    ``candidates`` are the complete numbered backups of the calling solver,
    sorted by step. ``None`` keeps the caller's in-memory state. ``initial``
    ignores all saved states; ``latest`` returns the last candidate, or
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
        path = Path(start_from)
        if not path.exists():
            raise FileNotFoundError(
                f"start_from={str(start_from)!r} is not 'latest', 'initial' or an existing "
                "backup path"
            )
        return path
    return candidates[-1] if candidates else None


def reset_run_outputs(
    directory: str | Path,
    *,
    kind: str,
    samples_dir: str | Path | None = None,
    sample_names: Iterable[str] = (),
) -> None:
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
        output_files.extend(
            path
            for path in (directory / component).glob(f"{component}_*")
            if pattern.fullmatch(path.name)
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


def output_has_time(directory: str | Path, name: str, time: float) -> bool:
    """Check whether a saved CSV/PVD time series contains the sampling time."""
    directory = Path(directory)
    table = directory / f"{name}.csv"
    if table.is_file():
        with table.open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            if "time" not in (rows.fieldnames or ()):
                return False
            if any(abs(float(row["time"]) - time) <= time_tolerance(time) for row in rows):
                return True
    return any(
        abs(frame_time - time) <= time_tolerance(time)
        for frame_time, _ in read_collection(directory / f"{name}.pvd")
    )
