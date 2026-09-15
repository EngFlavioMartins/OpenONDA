"""ParaView surface companions to native numerical backups."""

from pathlib import Path

import h5py
import numpy as np

from ..boundary_elements.vlm.solver.vtk_export import CELL_FIELDS, write_lattice_vtk
from .sampler import OutputManager


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
