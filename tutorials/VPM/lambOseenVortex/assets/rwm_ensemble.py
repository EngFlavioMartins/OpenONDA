"""Average samples from independent random-walk realizations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_COLUMNS = (
    "x_core",
    "y_core",
    "core_radius",
    "separation",
    "core_area",
    "surface_circulation",
    "core_0_x",
    "core_0_y",
    "core_1_x",
    "core_1_y",
)


def _nanmean(values: np.ndarray) -> np.ndarray:
    """Average columns that may intentionally contain only NaNs."""

    result = np.full(values.shape[1:], np.nan)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    populated = counts > 0
    result[populated] = np.nansum(values, axis=0)[populated] / counts[populated]
    return result


def _average_csv(target: Path, sources: list[Path]) -> None:
    frames = [pd.read_csv(path, comment="#") for path in sources]
    coordinates = ("x", "y", "z")

    for coordinate in coordinates:
        reference = frames[0][coordinate].to_numpy()
        if any(not np.allclose(frame[coordinate], reference) for frame in frames[1:]):
            raise ValueError(f"RWM ensemble grids differ in {coordinate!r}")

    averaged = frames[0].copy()
    field_columns = [column for column in averaged.columns if column not in coordinates]
    averaged[field_columns] = sum(frame[field_columns] for frame in frames) / len(frames)

    with target.open("r", encoding="utf-8") as handle:
        flow_time = handle.readline()
    with target.open("w", encoding="utf-8") as handle:
        handle.write(flow_time)
        averaged.to_csv(handle, index=False)


def _average_vts(target: Path, sources: list[Path]) -> None:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    grids = []
    for path in sources:
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(str(path))
        reader.Update()
        grids.append(reader.GetOutput())

    output = vtk.vtkStructuredGrid()
    output.DeepCopy(grids[0])
    point_data = output.GetPointData()

    for index in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(index)
        if name in {"VelocityMagnitude", "VorticityMagnitude"}:
            continue
        arrays = [vtk_to_numpy(grid.GetPointData().GetArray(name)) for grid in grids]
        vtk_to_numpy(point_data.GetArray(name))[:] = np.mean(arrays, axis=0)

    velocity = vtk_to_numpy(point_data.GetArray("Velocity"))
    vorticity = vtk_to_numpy(point_data.GetArray("Vorticity"))
    vtk_to_numpy(point_data.GetArray("VelocityMagnitude"))[:] = np.linalg.norm(velocity, axis=1)
    vtk_to_numpy(point_data.GetArray("VorticityMagnitude"))[:] = np.linalg.norm(vorticity, axis=1)

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(target))
    writer.SetInputData(output)
    if writer.Write() != 1:
        raise OSError(f"Could not write averaged RWM surface to {target}")


def average_final_samples(
    target_dir: Path,
    member_dirs: list[Path],
    *,
    realizations: int,
    particle_replicas: int,
) -> None:
    """Replace the primary final samples with the ensemble mean."""

    case_name = target_dir.name
    for axis in ("x", "y"):
        file_name = f"{case_name}_{axis}.csv"
        _average_csv(target_dir / file_name, [directory / file_name for directory in member_dirs])

    surface_name = max(target_dir.glob(f"{case_name}_z0_*.vts")).name
    _average_vts(
        target_dir / surface_name,
        [directory / surface_name for directory in member_dirs],
    )

    metadata_path = target_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "rwm_realizations": realizations,
            "rwm_particle_replicas": particle_replicas,
            "ensemble_averaged_samples": [
                f"{case_name}_x.csv",
                f"{case_name}_y.csv",
                surface_name,
            ],
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def average_pair_diagnostics(
    target_dir: Path,
    member_dirs: list[Path],
    *,
    realizations: int,
) -> None:
    """Replace a stochastic pair history with its independent ensemble mean."""

    frames = [pd.read_csv(directory / "pair_diagnostics.csv") for directory in member_dirs]
    reference_steps = frames[0]["time_step"].to_numpy()
    reference_times = frames[0]["flow_time"].to_numpy()
    for frame in frames[1:]:
        if not np.array_equal(frame["time_step"].to_numpy(), reference_steps):
            raise ValueError("RWM ensemble time steps differ")
        if not np.allclose(frame["flow_time"].to_numpy(), reference_times):
            raise ValueError("RWM ensemble flow times differ")

    averaged = frames[0].copy()
    for column in PAIR_COLUMNS:
        values = np.stack([frame[column].to_numpy(float) for frame in frames])
        averaged[column] = _nanmean(values)

    angles = np.stack(
        [np.unwrap(frame["angle_rad"].to_numpy(float)) for frame in frames]
    )
    averaged["angle_rad"] = _nanmean(angles)
    averaged["merged"] = (
        np.mean(
            [frame["merged"].astype(str).str.lower().eq("true") for frame in frames],
            axis=0,
        )
        >= 0.5
    )
    averaged.to_csv(target_dir / "pair_diagnostics.csv", index=False)

    integral_frames = [pd.read_csv(directory / "flow_integrals.csv") for directory in member_dirs]
    integral_times = integral_frames[0]["time"].to_numpy()
    if any(
        not np.allclose(frame["time"].to_numpy(), integral_times)
        for frame in integral_frames[1:]
    ):
        raise ValueError("RWM ensemble flow-integral times differ")
    averaged_integrals = integral_frames[0].copy()
    numeric_columns = averaged_integrals.select_dtypes(include=[np.number]).columns
    averaged_integrals[numeric_columns] = sum(
        frame[numeric_columns] for frame in integral_frames
    ) / len(integral_frames)
    averaged_integrals.to_csv(target_dir / "flow_integrals.csv", index=False)

    metadata_path = target_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "rwm_realizations": realizations,
            "ensemble_averaged_samples": [
                "pair_diagnostics.csv",
                "flow_integrals.csv",
            ],
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
