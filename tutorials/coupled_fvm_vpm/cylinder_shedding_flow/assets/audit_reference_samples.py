#!/usr/bin/env python3
"""Audit completeness and physical consistency of reference sample output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


LINE_FILES = (
    "centreline.csv",
    "transverse_x1.csv",
    "transverse_x2.csv",
    "transverse_x4.csv",
    "spanwise_line.csv",
)
POINT_FILES = ("forces_history.csv", "midspan_probe.csv")
FULL_FIELD_STEP = re.compile(r"_(\d{6})\.pvtu$")


def _csv_table(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
        if name != "patch"
    }


def _audit_csv(samples: Path, failures: list[str]) -> dict:
    report = {}
    for name in (*POINT_FILES, *LINE_FILES):
        path = samples / name
        if not path.is_file():
            failures.append(f"missing {name}")
            continue
        table = _csv_table(path)
        if any(not np.all(np.isfinite(values)) for values in table.values()):
            failures.append(f"non-finite values in {name}")
        times = table["time"]
        steps = table["step"]
        if np.any(np.diff(times) < 0.0) or np.any(np.diff(steps) < 0.0):
            failures.append(f"non-monotone row history in {name}")
        unique_times, counts = np.unique(times, return_counts=True)
        if np.unique(counts).size != 1:
            failures.append(f"incomplete frames in {name}: point counts {np.unique(counts)}")
        for sample_time in unique_times:
            if np.unique(steps[times == sample_time]).size != 1:
                failures.append(f"multiple solver steps in the t={sample_time:g} frame of {name}")
        if name in POINT_FILES and counts[0] != 1:
            failures.append(f"{name} must contain one row per frame")
        report[name] = {
            "rows": int(times.size),
            "frames": int(unique_times.size),
            "points_per_frame": int(counts[0]),
            "first_time": float(unique_times[0]),
            "last_time": float(unique_times[-1]),
        }
    return report


def _expected_spanwise_centres(case: Path) -> np.ndarray:
    metadata_path = case / "solution" / "benchmark_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mesh = metadata["mesh"]
    domain = np.asarray(mesh["effective_domain"], dtype=float)
    spacing = float(mesh["spanwise_cell_size"])
    count = int(mesh["spanwise_cells"])
    z_min, z_max = domain[4:6]
    if count < 1 or spacing <= 0.0 or not np.isclose(
        count * spacing, z_max - z_min, atol=1.0e-12
    ):
        raise ValueError("inconsistent spanwise mesh metadata")
    return z_min + (np.arange(count, dtype=float) + 0.5) * spacing


def _audit_spanwise(case: Path, samples: Path, failures: list[str]) -> dict:
    path = samples / "spanwise_line.csv"
    if not path.is_file():
        return {}
    table = _csv_table(path)
    try:
        expected_z = _expected_spanwise_centres(case)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"cannot determine spanwise slab centres: {error}")
        return {}
    ranges = {name: [] for name in ("velocity_x", "velocity_y", "velocity_z")}
    maximum_abs_z_velocity = 0.0
    for sample_time in np.unique(table["time"]):
        frame = table["time"] == sample_time
        order = np.argsort(table["position_z"][frame])
        z = table["position_z"][frame][order]
        if z.size != expected_z.size or not np.allclose(z, expected_z, atol=1.0e-12):
            failures.append(f"spanwise line at t={sample_time:g} is not aligned with slab centres")
            continue
        for name in ranges:
            values = table[name][frame][order]
            ranges[name].append(float(np.ptp(values)))
        maximum_abs_z_velocity = max(
            maximum_abs_z_velocity, float(np.max(np.abs(table["velocity_z"][frame])))
        )
    maxima = {name: max(values, default=float("nan")) for name, values in ranges.items()}
    if max(maxima.get("velocity_x", np.inf), maxima.get("velocity_y", np.inf)) > 2.0e-3:
        failures.append("spanwise in-plane velocity range exceeds 0.002 U_inf")
    if maximum_abs_z_velocity > 1.0e-3:
        failures.append("spanwise-normal velocity exceeds 0.001 U_inf")
    return {
        "maximum_range": maxima,
        "maximum_absolute_velocity_z": maximum_abs_z_velocity,
        "in_plane_range_limit": 2.0e-3,
        "normal_velocity_limit": 1.0e-3,
    }


def _spanwise_field_metrics(
    centres: np.ndarray, velocity: np.ndarray, pressure: np.ndarray
) -> dict[str, object]:
    """Measure deviations from the x-y cellwise span average."""
    centres = np.asarray(centres, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64).reshape(-1)
    if centres.shape != velocity.shape or velocity.shape[1] != 3 or pressure.shape != (
        len(centres),
    ):
        raise ValueError("full-field coherence arrays have incompatible shapes")
    _, groups = np.unique(np.round(centres[:, :2], decimals=12), axis=0, return_inverse=True)
    count = np.bincount(groups).astype(np.float64)
    mean_velocity = np.column_stack(
        [np.bincount(groups, weights=velocity[:, axis]) / count for axis in range(3)]
    )
    mean_pressure = np.bincount(groups, weights=pressure) / count
    velocity_deviation = velocity - mean_velocity[groups]
    pressure_deviation = pressure - mean_pressure[groups]
    return {
        "velocity_deviation_rms": {
            f"velocity_{axis}": float(np.sqrt(np.mean(velocity_deviation[:, index] ** 2)))
            for index, axis in enumerate(("x", "y", "z"))
        },
        "pressure_deviation_rms": float(np.sqrt(np.mean(pressure_deviation**2))),
        "maximum_absolute_velocity_z": float(np.max(np.abs(velocity[:, 2]))),
        "spanwise_groups": int(count.size),
        "slabs_per_group": sorted({int(value) for value in count}),
    }


def _audit_latest_full_field(case: Path, failures: list[str]) -> dict:
    """Audit the latest complete PVTU field, which line probes can miss."""
    import pyvista as pv

    candidates = []
    for path in (case / "solution").glob("*.pvtu"):
        match = FULL_FIELD_STEP.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        failures.append("no full-field PVTU backup available for coherence audit")
        return {}
    step, path = max(candidates)
    mesh = pv.read(path)
    ghost = np.asarray(mesh.cell_data.get("vtkGhostType", np.zeros(mesh.n_cells)), dtype=np.uint8)
    keep = ghost == 0
    metrics = _spanwise_field_metrics(
        np.asarray(mesh.cell_centers().points)[keep],
        np.asarray(mesh.cell_data["velocity"])[keep],
        np.asarray(mesh.cell_data["kinematic_pressure"])[keep],
    )
    metrics.update(
        {
            "file": path.name,
            "step": step,
            "normal_velocity_limit": 1.0e-3,
            "in_plane_deviation_rms_limit": 2.0e-3,
        }
    )
    rms = metrics["velocity_deviation_rms"]
    if max(rms["velocity_x"], rms["velocity_y"]) > 2.0e-3:
        failures.append("latest full field has excessive spanwise in-plane velocity deviation")
    if metrics["maximum_absolute_velocity_z"] > 1.0e-3:
        failures.append("latest full field has spanwise-normal velocity above 0.001 U_inf")
    return metrics


def _audit_slices(samples: Path, failures: list[str]) -> dict:
    import pyvista as pv

    paths = sorted(samples.glob("slice_z0_*.vts"))
    pvd = samples / "slice_z0.pvd"
    if not paths:
        failures.append("no corrected slice_z0 VTS frames")
        return {"frames": 0}
    if not pvd.is_file():
        failures.append("missing slice_z0.pvd")
        listed = []
    else:
        tree = ET.parse(pvd)
        listed = [entry.attrib["file"] for entry in tree.findall(".//DataSet")]
        existing = {path.name for path in paths}
        if set(listed) != existing or len(listed) != len(existing):
            failures.append("slice_z0.pvd entries do not match VTS files")

    invalid_total = 0
    for path in paths:
        grid = pv.read(path)
        points = np.asarray(grid.points)
        valid = np.asarray(grid.point_data["vtkValidPointMask"], dtype=bool)
        radius = np.linalg.norm(points[:, :2], axis=1)
        if np.any((~valid) & (radius >= 0.5 - 1.0e-6)):
            failures.append(f"{path.name} masks fluid points outside the circular cylinder")
        if np.any(valid & (radius < 0.5 - 1.0e-6)):
            failures.append(f"{path.name} exposes sampled flow inside the circular cylinder")
        for field in ("velocity", "vorticity", "kinematic_pressure"):
            values = np.asarray(grid.point_data[field])
            if not np.all(np.isfinite(values[valid])):
                failures.append(f"{path.name} has non-finite valid {field} values")
        invalid_total += int(np.count_nonzero(~valid))
    return {
        "frames": len(paths),
        "pvd_entries": len(listed),
        "invalid_solid_points": invalid_total,
    }


def _audit_provenance(case: Path, failures: list[str]) -> dict:
    """Prove that the reference uses a conformal wall, not IBM forcing."""
    solution = case / "solution"
    metadata_path = solution / "benchmark_metadata.json"
    manifest_path = solution / "run_manifest.json"
    if not metadata_path.is_file() or not manifest_path.is_file():
        failures.append("missing benchmark metadata or run manifest")
        return {}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mesh = metadata.get("mesh", {})
    forces = metadata.get("forces", {})
    active = manifest.get("active_components", {})
    configuration = manifest.get("configuration", {})
    samplers = configuration.get("samplers", [])
    cylinder_boundaries = [
        boundary
        for boundary in configuration.get("boundaries", [])
        if boundary.get("name") == "cylinder"
    ]

    checks = {
        "body_fitted_conformal_mesh": mesh.get("geometry_treatment")
        == "body-fitted_conformal_surface_mesh",
        "metadata_immersed_boundary_disabled": mesh.get("immersed_boundary") is False,
        "manifest_immersed_boundary_inactive": active.get("immersed_boundary") is False,
        "wall_traction_forces": forces.get("method")
        == "pressure_and_viscous_traction_on_cylinder_wall_patch",
        "immersed_boundary_forcing_disabled": forces.get("immersed_boundary_forcing") is False,
        "cylinder_is_wall_patch": len(cylinder_boundaries) == 1
        and cylinder_boundaries[0].get("mesh_type") == "wall",
        "no_ibm_sampler": all("IBM" not in str(sampler.get("type", "")) for sampler in samplers),
        "no_ibm_sample_files": not any(case.joinpath("samples").glob("*ibm*")),
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(f"provenance check failed: {name}")
    return {
        "surface_file": mesh.get("surface_file"),
        "surface_sha256": mesh.get("surface_sha256"),
        "force_sampler_types": [sampler.get("type") for sampler in samplers],
        "checks": checks,
    }


def audit(case: Path) -> dict:
    samples = case / "samples"
    failures: list[str] = []
    report = {
        "schema": 1,
        "case": "fully_meshed_body_fitted_reference",
        "csv": _audit_csv(samples, failures),
        "spanwise_coherence": _audit_spanwise(case, samples, failures),
        "latest_full_field_coherence": _audit_latest_full_field(case, failures),
        "surface_slices": _audit_slices(samples, failures),
        "provenance": _audit_provenance(case, failures),
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }
    output = case / "solution" / "sample_quality.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reference_flow",
    )
    args = parser.parse_args()
    report = audit(args.case.resolve())
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
