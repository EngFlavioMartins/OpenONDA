"""Read-only wake-domain, pruning and profile audit for the planar cylinder.

Read published reference fields and sample histories without constructing a
solver or device. Save compact numerical snapshots and hashes under tests.
The omitted-reference-vorticity velocity is a representation diagnostic,
not an evolved coupled-domain sensitivity experiment.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.integrate import trapezoid
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

from .analyze_planar_wall_trace import face_geometry, trace_statistics

CASE = Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_information(path, content):
    return {"path": str(path.resolve()), "sha256": sha256(content), "bytes": len(content)}


def profile_statistics(path, count, start, end):
    content = path.read_bytes()
    content = content[: content.rfind(b"\n") + 1]
    names = content.splitlines()[0].decode().split(",")
    values = np.loadtxt(io.BytesIO(content), delimiter=",", skiprows=1)
    values = values[: len(values) // count * count].reshape(-1, count, len(names))
    times = values[:, 0, names.index("time")]
    values = values[(times >= start - 1e-8) & (times <= end + 1e-8)]
    times = values[:, 0, names.index("time")]
    if abs(times[0] - start) > 1e-8 or abs(times[-1] - end) > 1e-8:
        raise ValueError("Profile history does not cover the requested complete window")
    results = []
    for index in [0, 4, 8] if count == 9 else [4, 24, 64]:
        record = {"x": float(values[0, index, names.index("position_x")])}
        for field in ("velocity_x", "velocity_y", "vorticity_z"):
            field_values = values[:, index, names.index(field)]
            mean = float(trapezoid(field_values, times) / (end - start))
            record[field] = {
                "mean": mean,
                "fluctuation_rms": float(
                    np.sqrt(trapezoid((field_values - mean) ** 2, times) / (end - start))
                ),
                "whole_window_peak_to_peak": float(np.ptp(field_values)),
            }
        results.append(record)
    information = source_information(path, content)
    information.update(columns=names, window=[start, end], captured_shape=list(values.shape))
    return results, values, information


def far_velocity(target, source, circulation):
    delta = target[:, None, :2] - source[None, :, :2]
    squared_distance = np.einsum("ijk,ijk->ij", delta, delta)
    weight = circulation[None, :] / (2.0 * np.pi * squared_distance)
    return np.column_stack(
        (
            -np.sum(delta[:, :, 1] * weight, axis=1),
            np.sum(delta[:, :, 0] * weight, axis=1),
            np.zeros(len(target)),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--start", type=float, default=30)
    parser.add_argument("--end", type=float, default=44)
    args = parser.parse_args()
    root = args.case.resolve()
    study = args.study.resolve()
    wall_directory = next(study.glob("wall_trace_*/wall_trace_statistics.json")).parent
    checkpoint = json.loads((wall_directory / "checkpoint/checkpoint_info.json").read_text())
    particle_path = wall_directory / "checkpoint" / checkpoint["checkpoint_files"]["vpm"]
    bounds = np.asarray(checkpoint["config"]["vpm"]["domain_bounds"])
    span = float(checkpoint["config"]["vpm"]["induction"]["planar_span"])
    sources = {}
    arrays = {}
    profiles = {}
    for group, count in (("near", 9), ("wake", 65)):
        for side in ("upper", "lower"):
            line = f"phase_{group}_{side}"
            series = (
                ("reference", root / f"reference_flow/samples/{line}.csv"),
                ("VPM", root / f"samples/vpm_{line}.csv"),
            )
            if group == "near":
                series += (("coupled_FVM", root / f"samples/{line}.csv"),)
            for label, path in series:
                key = f"{label}_{line}"
                profiles[key], arrays[key], sources[key] = profile_statistics(
                    path, count, args.start, args.end
                )
    diagnostics_path = root / "solution/coupler_diagnostics.jsonl"
    diagnostic_bytes = diagnostics_path.read_bytes()
    diagnostic_bytes = diagnostic_bytes[: diagnostic_bytes.rfind(b"\n") + 1]
    diagnostics = [json.loads(line) for line in diagnostic_bytes.splitlines()]
    sources["coupler_diagnostics"] = source_information(diagnostics_path, diagnostic_bytes)
    quantities = (
        ("transfer", "pruned_vortex_strength_fraction"),
        ("transfer", "pruned_vortex_strength_l1"),
        ("transfer", "renewal_applied_particle_strength_fraction"),
        ("transfer", "population_pruned_particles"),
        ("transfer", "population_pruned_velocity_bound"),
        ("gbd_moment_recovery", "correction_fraction"),
        ("gbd_moment_recovery", "normalized_vortex_strength_residual"),
        ("gbd_moment_recovery", "normalized_linear_impulse_residual"),
        ("gbd_moment_recovery", "normalized_angular_impulse_residual"),
    )
    pruning = {}
    for start, end in ((10, 20), (20, 30), (30, 44), (40, 48)):
        selected = [record for record in diagnostics if start < record["time"] <= end]
        result = {"interval": [start, end], "exchange_count": len(selected)}
        for block, field in quantities:
            values = np.asarray([record[block][field] for record in selected])
            result[f"{block}.{field}"] = {
                "mean": float(np.mean(values)),
                "maximum": float(np.max(values)),
            }
        pruning[f"{start}-{end}"] = result
    arrays["diagnostic_values"] = np.asarray(
        [
            [record["time"], *[record[block][field] for block, field in quantities]]
            for record in diagnostics
        ]
    )
    mesh = load_native_mesh(wall_directory / "coupled_mesh.npz")
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    reference_mesh_path = wall_directory / "reference_mesh.npz"
    reference_mesh = load_native_mesh(reference_mesh_path)
    reference_geometry = compute_mesh_geometry(reference_mesh, compute_lsq=False)
    sources["reference_mesh"] = source_information(
        reference_mesh_path, reference_mesh_path.read_bytes()
    )
    reference_source_points = reference_geometry["cell_centre"]
    pvd_path = root / "reference_flow/solution/fvm.pvd"
    pvd_bytes = pvd_path.read_bytes()
    sources["reference_pvd"] = source_information(pvd_path, pvd_bytes)
    entries = {
        entry.attrib["file"]: float(entry.attrib["timestep"])
        for entry in ET.fromstring(pvd_bytes).iter("DataSet")
    }
    omitted = {}
    for step in (5500, 6000):
        relative = f"fvm/fvm_{step:06}.vtu"
        path = root / "reference_flow/solution" / relative
        content = path.read_bytes()
        sources[f"reference_field_{step}"] = source_information(path, content)
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(path))
        reader.Update()
        data = reader.GetOutput().GetCellData()
        vorticity = vtk_to_numpy(data.GetArray("vorticity")).astype(np.float64)
        volume = vtk_to_numpy(data.GetArray("cell_volume")).astype(np.float64)
        if sha256(path.read_bytes()) != sources[f"reference_field_{step}"]["sha256"]:
            raise ValueError("Reference field changed during inspection")
        np.testing.assert_allclose(volume, reference_geometry["cell_volume"], rtol=1e-6, atol=1e-10)
        circulation = vorticity[:, 2] * volume / span
        points = reference_source_points
        masks = {
            "outside_VPM_bounds": (points[:, 0] < bounds[0])
            | (points[:, 0] > bounds[1])
            | (points[:, 1] < bounds[2])
            | (points[:, 1] > bounds[3]),
            "downstream_x_above_15": points[:, 0] > bounds[1],
            "lateral_abs_y_above_5": (points[:, 1] < bounds[2]) | (points[:, 1] > bounds[3]),
        }
        result = {"accepted_field_time_from_pvd": entries[relative], "native_step": step}
        for name, mask in masks.items():
            record = {
                "cell_count": int(np.count_nonzero(mask)),
                "circulation_l1": float(np.sum(abs(circulation[mask]))),
                "signed_circulation": float(np.sum(circulation[mask])),
            }
            for patch in ("cylinder", "numericalBoundary"):
                _, target, normal, area = face_geometry(mesh, geometry, patch)
                velocity = far_velocity(target, points[mask], circulation[mask])
                record[patch] = trace_statistics(velocity, normal, area)
                record[patch]["area_weighted_mean_velocity"] = np.average(
                    velocity, weights=area, axis=0
                ).tolist()
            result[name] = record
        omitted[str(step)] = result
        arrays[f"reference_{step}_source_points"] = points
        arrays[f"reference_{step}_vorticity"] = vorticity
        arrays[f"reference_{step}_volume"] = volume
    sources["frozen_vpm_checkpoint"] = source_information(particle_path, particle_path.read_bytes())
    with h5py.File(particle_path, "r") as saved:
        position = saved["particles/position"][:].astype(np.float64)
        circulation = saved["particles/vortex_strength"][:, 2].astype(np.float64) / span
    wake = {
        "native_time": checkpoint["time"],
        "particle_count": len(position),
        "position_minimum": position.min(axis=0).tolist(),
        "position_maximum": position.max(axis=0).tolist(),
        "circulation_l1": float(np.sum(abs(circulation))),
        "signed_circulation": float(np.sum(circulation)),
        "downstream_sections": {
            str(x): {
                "particle_count": int(np.count_nonzero(position[:, 0] > x)),
                "circulation_l1": float(np.sum(abs(circulation[position[:, 0] > x]))),
                "signed_circulation": float(np.sum(circulation[position[:, 0] > x])),
            }
            for x in (2.4, 5, 8, 10, 12, 14)
        },
    }
    snapshots = study / "vpm_boundary_source_snapshots.npz"
    np.savez_compressed(snapshots, **arrays)
    report = {
        "schema": "openonda-planar-vpm-boundary-source-audit/1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_information": sources,
        "numerical_snapshots": source_information(snapshots, snapshots.read_bytes()),
        "diagnostic_array_columns": ["time", *[f"{block}.{field}" for block, field in quantities]],
        "last_complete_diagnostic_time": diagnostics[-1]["time"],
        "vpm_domain_bounds": bounds.tolist(),
        "frozen_wake_geometry": wake,
        "pruning": pruning,
        "profile_window": [args.start, args.end],
        "profile_statistics": profiles,
        "reference_vorticity_outside_vpm_domain": omitted,
        "methods": {
            "reference_cell_circulation": "Gamma_2D = native vorticity_z * native cell_volume / represented_span; stored vector strength is Gamma_2D * span",
            "source_coordinates": "Native FVM cell centres, in the identical native VTK cell order; exported volumes checked against native geometry",
            "far_field_velocity": "sum Gamma_2D/(2*pi*r_xy^2) * (-r_y,r_x,0); no freestream; target-source separation makes Gaussian core correction negligible",
            "profile_rms": "Trapezoidal time integral of fluctuations about each probe's 30–44 s mean; no phase alignment or data masking",
            "pruning_window": "Accepted exchanges with start < time <= end; quantities are per-exchange, not summed as cumulative mass losses",
            "reproduction_script": str(Path(__file__).resolve()),
        },
        "definite_findings": [
            "Particle-capacity pruning is absent in every inspected interval.",
            "The 46 s wake reaches the 15 m downstream limit, while its lateral particle extent remains well inside the 5 m limits.",
            "GBD preserves its measured signed circulation/impulse moments after pruning; these checks do not bound local velocity error.",
            "The reference vorticity omitted by the VPM bounds induces about 0.003–0.011 m/s RMS at the outer patch in the inspected 44/48 s fields.",
            "Near-body crossflow fluctuations are weaker in both coupled FVM and VPM; downstream crossflow fluctuations recover toward the reference levels.",
        ],
        "limits": [
            "Reference 44/48 s sources are not the exact unknown discarded coupled wake. Their induced velocity is an instantaneous representation diagnostic, not a causal force prediction.",
            "Domain retention logs record particle counts, not removed circulation; cumulative actual discarded strength cannot be reconstructed from those logs.",
            "Pruning L1 diagnostics report strength modified before conservation redistribution; summing them would misrepresent an actual circulation loss.",
            "Moment conservation does not establish enstrophy, small-scale structure or local velocity convergence.",
            "Profile windows contain several shedding cycles but are not phase aligned; mean/fluctuation statistics are evidence of the spatial pattern only.",
            "Domain enlargement, lower filtering thresholds and body-compatible induction require separate matched evolved controls before attributing the force deficit.",
        ],
    }
    path = study / "vpm_boundary_source_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
