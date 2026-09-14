"""Make z=0 slices from the fine reference's saved fields for the cube plots.

The fine reference currently records forces and lines, but no slice sampler.
Only field times also present in the coupled samples are extracted, and each
slice is reused on subsequent plotting runs. The reference simulation is not
started or modified.
"""

from __future__ import annotations

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ._reference_util import sample_vtu

CASE_DIR = Path(__file__).resolve().parents[1]
SPACING = min(0.125, 2 * 0.06)


def _frames(pvd: Path) -> list[tuple[float, Path]]:
    if not pvd.is_file():
        return []
    return sorted(
        (float(item.attrib["timestep"]), pvd.parent / item.attrib["file"])
        for item in ET.parse(pvd).iter("DataSet")
    )


def _write_index(pvd: Path, frames: list[tuple[float, Path]]) -> None:
    vtk = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(vtk, "Collection")
    for time, path in sorted(frames):
        ET.SubElement(
            collection,
            "DataSet",
            timestep=format(time, ".17g"),
            file=path.name,
        )
    temporary = pvd.with_name(pvd.name + ".tmp")
    ET.ElementTree(vtk).write(temporary, encoding="utf-8", xml_declaration=True)
    os.replace(temporary, pvd)


def _write_slice(source: Path, destination: Path) -> None:
    import pyvista as pv

    x = np.arange(-1.5, 1.5 + SPACING / 2, SPACING, dtype=np.float64)
    y = np.arange(-1.5, 1.5 + SPACING / 2, SPACING, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    grid = pv.StructuredGrid(xx, yy, np.zeros_like(xx))
    values = sample_vtu(source, np.asarray(grid.points))
    if "velocity" not in values:
        raise ValueError(f"Reference field has no velocity: {source}")
    velocity = np.asarray(values["velocity"], dtype=np.float32)
    interior = (np.abs(grid.points[:, 0]) < 0.5) & (np.abs(grid.points[:, 1]) < 0.5)
    valid = np.asarray(values["valid"], dtype=bool) & ~interior
    velocity[~valid] = np.nan
    grid.point_data["velocity"] = velocity
    grid.point_data["vtkValidPointMask"] = valid.astype(np.uint8)
    if "vorticity" in values:
        vorticity = np.asarray(values["vorticity"], dtype=np.float32)
        vorticity[~valid] = np.nan
        grid.point_data["vorticity"] = vorticity
    grid.field_data["surface_ordering"] = np.asarray([1], dtype=np.uint8)
    temporary = destination.with_name(destination.stem + ".tmp.vts")
    grid.save(temporary, binary=True)
    os.replace(temporary, destination)


def prepare_fine_reference(
    reference_solution: Path,
    reference_samples: Path,
    coupled_samples: Path,
) -> int:
    """Extract available, exactly coincident reference fields; return new count."""
    if not (reference_samples / "forces_history.csv").is_file():
        return 0
    source_pvd = reference_solution / "fine.pvd"
    coupled_fvm = _frames(coupled_samples / "fvm_slice_z0.pvd")
    coupled_vpm = _frames(coupled_samples / "vpm_slice_z0.pvd")
    if not coupled_fvm or not coupled_vpm:
        return 0
    shared = [
        time
        for time, _ in coupled_fvm
        if any(np.isclose(time, other, rtol=0, atol=1e-9) for other, _ in coupled_vpm)
    ]
    if not shared:
        return 0
    target_pvd = reference_samples / "slice_z0.pvd"
    existing = _frames(target_pvd)
    count = 0
    for time, source in _frames(source_pvd):
        if time <= 0 or not source.is_file():
            continue
        if not any(np.isclose(time, other, rtol=0, atol=1e-9) for other in shared):
            continue
        if any(np.isclose(time, old, rtol=0, atol=1e-9) and path.is_file() for old, path in existing):
            continue
        destination = reference_samples / f"slice_z0_t{time:.9f}.vts"
        _write_slice(source, destination)
        existing = [(old, path) for old, path in existing if not np.isclose(old, time, rtol=0, atol=1e-9)]
        existing.append((time, destination))
        _write_index(target_pvd, existing)
        count += 1
    return count


if __name__ == "__main__":
    count = prepare_fine_reference(
        CASE_DIR / "reference_flow" / "solution" / "fine",
        CASE_DIR / "reference_flow" / "samples" / "fine",
        CASE_DIR / "samples",
    )
    if count:
        print(f"Prepared {count} fine-reference field slice(s).")
