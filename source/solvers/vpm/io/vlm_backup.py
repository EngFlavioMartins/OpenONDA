"""ParaView surface companions to native numerical backups."""

from pathlib import Path

from defusedxml.ElementTree import ParseError, parse
import h5py
import numpy as np

from source.solution_layout import collection_path, component_directory

from ..boundary_elements.vlm.solver.vtk_export import CELL_FIELDS, write_lattice_vtk


def _read_surface_entries(solution_directory: Path) -> list[tuple[float, str]]:
    """Read the canonical VLM collection and validate its relative entries."""
    collection = collection_path(solution_directory, "vlm")
    if not collection.is_file():
        return []
    try:
        root = parse(collection).getroot()
        entries = [
            (float(dataset.attrib["timestep"]), dataset.attrib["file"])
            for dataset in root.findall(".//DataSet")
        ]
    except (ParseError, OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid VLM collection {collection}") from exc
    if entries != sorted(entries) or len({filename for _, filename in entries}) != len(entries):
        raise ValueError(f"VLM collection {collection} is not monotonic and unique")
    return entries


def _write_surface_entries(solution_directory: Path, entries: list[tuple[float, str]]) -> None:
    """Atomically publish the root-level VLM collection."""
    collection = collection_path(solution_directory, "vlm")
    collection.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
        *(f'    <DataSet timestep="{time:.17g}" file="{filename}"/>' for time, filename in entries),
        "  </Collection>",
        "</VTKFile>",
        "",
    ]
    temporary = collection.with_name(f".{collection.name}.tmp")
    try:
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(collection)
    finally:
        temporary.unlink(missing_ok=True)


def _surface_entry(solution_directory, step, time):
    """Merge by checkpoint name, allowing an older saved state to be filled in."""
    frame_directory = component_directory(solution_directory, "vlm")
    filename = (frame_directory / f"vlm_{step:06d}.vtp").relative_to(solution_directory).as_posix()
    entries = _read_surface_entries(solution_directory)
    for existing_time, existing_file in entries:
        if (existing_file == filename) != (existing_time == time):
            raise ValueError("VLM backup filename/time conflicts with the existing series")
    entries = sorted({*entries, (float(time), filename)})
    return filename, entries


def write_vlm_backup(vlm, directory, *, step, time):
    """Save the accepted surface on the VPM backup clock, independent of samplers."""
    solution_directory = Path(directory)
    filename, entries = _surface_entry(solution_directory, step, time)
    destination = solution_directory / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    vlm.save_results(str(destination.with_suffix("")), time=time)
    _write_surface_entries(solution_directory, entries)
    return destination


def export_vlm_backup(checkpoint):
    """Recreate a surface companion from stored fields, without loading a solver.

    The HDF5 file is read-only. Only recorded optional properties are exported;
    no motion, circulation or force is inferred from the current setup.
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
            # Export the stored Bernoulli diagnostic with its physical meaning.
            fields["pressure_coefficient"] = np.asarray(state["pressure_coefficient"])
        reference_speed = float(state.attrs["reference_speed"])
        step = int(file["solver"].attrs["step"])
        time = float(file["solver"].attrs["time"])
        force_density = state.attrs.get("force_density")
    source_directory = checkpoint.parent
    solution_directory = (
        source_directory.parent if source_directory.name in {"backups", "vpm"} else source_directory
    )
    filename, entries = _surface_entry(solution_directory, step, time)
    destination = solution_directory / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = write_lattice_vtk(
        fields,
        destination,
        reference_speed=reference_speed,
        time=time,
        force_density=force_density,
    )
    _write_surface_entries(solution_directory, entries)
    return result
