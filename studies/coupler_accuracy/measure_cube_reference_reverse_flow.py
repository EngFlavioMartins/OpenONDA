"""Measure the 3D reverse-flow region in saved native FVM reference fields.

Usage:
    python studies/coupler_accuracy/measure_cube_reference_reverse_flow.py \
        --pvd tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/solution/fine/fine.pvd \
        --times 8 15 20 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pyvista as pv


def _native_frames(pvd: Path) -> dict[float, Path]:
    """Return the native PVTU path at each recorded physical time [s]."""
    root = ElementTree.parse(pvd).getroot()
    frames = {}
    for entry in root.findall(".//DataSet"):
        time = float(entry.attrib["timestep"])
        frames[time] = pvd.parent / entry.attrib["file"]
    if not frames:
        raise ValueError(f"PVD contains no native FVM frames: {pvd}")
    return frames


def _reverse_flow(pvtu: Path, time: float) -> dict[str, float | int | list[float]]:
    """Measure downstream cells with negative streamwise velocity in the 3D box.

    The cube occupies [-0.5,0.5]^3 m. Select fluid cell centres with x>0.5 m
    and |y|,|z|<1.5 m; volume is summed from native cell volumes [m³]. This
    excludes potential reverse flow beyond the selected transverse box.
    """
    mesh = pv.read(pvtu)
    centre = np.asarray(mesh.cell_centers().points, dtype=np.float64)
    velocity = np.asarray(mesh.cell_data["velocity"], dtype=np.float64)
    vorticity = np.asarray(mesh.cell_data["vorticity"], dtype=np.float64)
    volume = np.asarray(mesh.cell_data["cell_volume"], dtype=np.float64)
    if (
        centre.shape != (mesh.n_cells, 3)
        or velocity.shape != centre.shape
        or vorticity.shape != centre.shape
        or volume.shape != (mesh.n_cells,)
        or not np.isfinite(centre).all()
        or not np.isfinite(velocity).all()
        or not np.isfinite(vorticity).all()
        or not np.isfinite(volume).all()
        or np.any(volume <= 0)
    ):
        raise ValueError(f"Invalid native FVM fields in {pvtu}")
    downstream_box = (
        (centre[:, 0] > 0.5) & (np.abs(centre[:, 1]) < 1.5) & (np.abs(centre[:, 2]) < 1.5)
    )
    reverse = downstream_box & (velocity[:, 0] < 0)
    if not reverse.any():
        raise ValueError(f"No downstream reverse-flow cells in {pvtu}")
    compact_reverse = reverse & (centre[:, 0] < 1.5) & (np.abs(centre[:, 1]) < 1.25)
    compact_reverse &= np.abs(centre[:, 2]) < 1.25
    if not compact_reverse.any():
        raise ValueError(f"No compact-box reverse-flow cells in {pvtu}")
    omega = vorticity[compact_reverse]
    omega_rms = np.sqrt(np.mean(omega * omega, axis=0))
    return {
        "time": time,
        "all_fluid_cells": mesh.n_cells,
        "reverse_flow_cells": int(reverse.sum()),
        "reverse_flow_volume_m3": float(volume[reverse].sum()),
        "furthest_reverse_flow_centre_x_m": float(centre[reverse, 0].max()),
        "maximum_reverse_flow_abs_y_m": float(np.abs(centre[reverse, 1]).max()),
        "maximum_reverse_flow_abs_z_m": float(np.abs(centre[reverse, 2]).max()),
        "compact_reverse_flow_cells": int(compact_reverse.sum()),
        "compact_reverse_flow_vorticity_component_rms_1_s": omega_rms.tolist(),
        "compact_reverse_flow_streamwise_vorticity_rms_fraction": float(
            omega_rms[0] / np.linalg.norm(omega_rms)
        ),
    }


def main() -> None:
    """Read saved fields without modifying or advancing either solver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvd", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    args = parser.parse_args()
    frames = _native_frames(args.pvd)
    missing = [time for time in args.times if time not in frames]
    if missing:
        raise ValueError(f"Missing exact native FVM frame times: {missing}")
    results = [_reverse_flow(frames[time], time) for time in args.times]
    print(json.dumps({"source_pvd": str(args.pvd), "results": results}, indent=2))


if __name__ == "__main__":
    main()
