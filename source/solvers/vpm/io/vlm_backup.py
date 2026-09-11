"""ParaView surface companions to native numerical backups."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil

import h5py
import numpy as np

from ..boundary_elements.vlm.solver.vtk_export import CELL_FIELDS, write_lattice_vtk
from .sampler import OutputManager


@dataclass(frozen=True)
class VLMSeriesMigration:
    """Summary returned after reconciling a legacy samples surface series."""

    moved_frames: int
    deduplicated_frames: int
    solution_frames: int


def _surface_series_entries(directory: Path) -> list[tuple[float, str]]:
    """Validate one VLM PVD and its referenced files before migration."""
    entries = OutputManager._read_pvd(directory, "vlm")
    if len({time for time, _ in entries}) != len(entries):
        raise ValueError(f"VLM surface series {directory / 'vlm.pvd'} has duplicate times")
    for _, filename in entries:
        path = Path(filename)
        if path.name != filename or path.is_absolute() or not filename.startswith("vlm_"):
            raise ValueError(f"VLM surface index contains an unsafe file reference: {filename!r}")
        if not (directory / filename).is_file():
            raise ValueError(f"VLM surface index references a missing file: {directory / filename}")
    indexed = {filename for _, filename in entries}
    actual = {path.name for path in directory.glob("vlm_*.vtp") if path.is_file()}
    if actual != indexed:
        missing = sorted(indexed - actual)
        unindexed = sorted(actual - indexed)
        raise ValueError(
            f"VLM surface series is not self-contained (missing={missing}, unindexed={unindexed})"
        )
    return entries


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_owner_backups(directory: Path, entries: list[tuple[float, str]]) -> None:
    """Require every surface frame to belong to a saved coupled VPM state."""
    for time, filename in entries:
        step_text = Path(filename).stem.removeprefix("vlm_")
        if not step_text.isdigit():
            raise ValueError(f"VLM surface filename has no native step: {filename!r}")
        step = int(step_text)
        checkpoint = directory / f"vpm_{step:06d}.h5"
        if not checkpoint.is_file():
            raise ValueError(f"VLM surface {filename!r} has no matching VPM backup: {checkpoint}")
        with h5py.File(checkpoint, "r") as file:
            solver = file.get("solver")
            if solver is None or "vlm" not in solver:
                raise ValueError(f"VLM surface requires a coupled VPM backup: {checkpoint}")
            native_step = solver.attrs["step"]
            native_time = float(solver.attrs["time"])
        if native_step != step or not np.isfinite(native_time) or native_time != time:
            raise ValueError(
                f"VLM surface {filename!r} step/time conflicts with VPM backup {checkpoint}"
            )


def migrate_vlm_surface_series(samples_directory, solution_directory) -> VLMSeriesMigration:
    """Move a legacy samples VLM series into the native solution directory.

    The source run must be finalized before calling this function. It merges
    only indexed, existing frames that match the native VPM checkpoint clock.
    Unpaired surfaces, clock conflicts and byte-different duplicates fail before
    any files are changed. The source PVD and VTPs are deleted only after the
    merged solution PVD has been written.
    """
    source = Path(samples_directory)
    target = Path(solution_directory)
    if source.resolve() == target.resolve():
        raise ValueError("VLM migration requires distinct samples and solution directories")
    source_pvd = source / "vlm.pvd"
    target_pvd = target / "vlm.pvd"
    if not source_pvd.is_file() and any(source.glob("vlm_*.vtp")):
        raise ValueError(f"VLM surface files exist without an index: {source}")
    if not target_pvd.is_file() and any(target.glob("vlm_*.vtp")):
        raise ValueError(f"VLM surface files exist without an index: {target}")
    source_entries = _surface_series_entries(source) if source_pvd.is_file() else []
    target_entries = _surface_series_entries(target) if target_pvd.is_file() else []
    _validate_owner_backups(target, [*target_entries, *source_entries])
    if not source_entries:
        return VLMSeriesMigration(0, 0, len(target_entries))

    target_by_name = {filename: time for time, filename in target_entries}
    target_by_time = dict(target_entries)
    unique: list[tuple[float, str]] = []
    duplicates = 0
    for time, filename in source_entries:
        source_path = source / filename
        if filename in target_by_name:
            if target_by_name[filename] != time:
                raise ValueError(f"VLM surface filename/time conflict for {filename!r}")
            target_path = target / filename
            if _file_digest(source_path) != _file_digest(target_path):
                raise ValueError(f"VLM surface duplicate is not byte-identical: {filename!r}")
            duplicates += 1
            continue
        if time in target_by_time:
            raise ValueError(f"VLM surface time conflicts with {target_by_time[time]!r}")
        destination = target / filename
        if destination.exists():
            raise ValueError(f"VLM migration destination already exists: {destination}")
        unique.append((time, filename))

    target.mkdir(parents=True, exist_ok=True)
    for _, filename in unique:
        source_path = source / filename
        destination = target / filename
        temporary = target / f".{filename}.migration.tmp"
        shutil.copy2(source_path, temporary)
        temporary.replace(destination)

    merged = sorted((*target_entries, *unique))
    OutputManager._write_pvd(target, "vlm", merged)
    for _, filename in source_entries:
        (source / filename).unlink()
    source_pvd.unlink()
    return VLMSeriesMigration(len(unique), duplicates, len(merged))


def _surface_entry(directory, step, time):
    """Merge by checkpoint name, allowing an older saved state to be filled in."""
    filename = f"vlm_{step:06d}.vtp"
    entries = OutputManager._read_pvd(directory, "vlm")
    for existing_time, existing_file in entries:
        if (existing_file == filename) != (existing_time == time):
            raise ValueError("VLM backup filename/time conflicts with the existing series")
    entries = sorted({*entries, (float(time), filename)})
    return filename, entries


def write_vlm_backup(vlm, directory, *, step, time):
    """Save the accepted surface on the VPM backup clock, independent of samplers."""
    directory = Path(directory)
    filename, entries = _surface_entry(directory, step, time)
    vlm.save_results(str((directory / filename).with_suffix("")), time=time)
    OutputManager._write_pvd(directory, "vlm", entries)
    return directory / filename


def export_vlm_backup(checkpoint):
    """Recreate a surface companion from stored fields, without loading a solver.

    The HDF5 file is read-only. Older checkpoints omit properties they did not
    store; no motion, circulation or force is inferred from the current setup.
    Return ``None`` for a particle-only checkpoint.
    """
    checkpoint = Path(checkpoint)
    with h5py.File(checkpoint, "r") as file:
        state = file.get("solver/vlm")
        if state is None:
            return None
        required = {"panel_corner_position", "circulation", "normal", "panel_force"}
        if not required <= set(state):
            raise ValueError(f"{checkpoint} is missing saved VLM surface fields")
        names = ("panel_corner_position", "vortex_point_position", *CELL_FIELDS)
        fields = {name: np.asarray(state[name]) for name in names if name in state}
        if "pressure_coefficient" in state:
            # Legacy checkpoints store this speed-based diagnostic under its
            # historical name; vtk_export writes it only as an explicit
            # ``speed_pressure_coefficient`` field.
            fields["pressure_coefficient"] = np.asarray(state["pressure_coefficient"])
        reference_speed = float(state.attrs["reference_speed"])
        step = int(file["solver"].attrs["step"])
        time = float(file["solver"].attrs["time"])
        force_density = state.attrs.get("force_density")
    filename, entries = _surface_entry(checkpoint.parent, step, time)
    result = write_lattice_vtk(
        fields,
        checkpoint.parent / filename,
        reference_speed=reference_speed,
        time=time,
        force_density=force_density,
    )
    OutputManager._write_pvd(checkpoint.parent, "vlm", entries)
    return result
