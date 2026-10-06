"""Read-only Gaussian planar induction at native cylinder and coupling faces.

This diagnostic uses a hash-checked committed checkpoint and streamed NumPy
evaluation. It creates no solver, Taichi device, or particle evolution. Core-size
variations are instantaneous diagnostic evaluations of unchanged sources, not
alternative simulations or evidence of recovered force amplitudes.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

from source.solvers.fvm.io.backup import decode_state
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

CASE = Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def freeze_checkpoint(directory, destination):
    """Retry checkpoint publication; persist only a complete unchanged generation."""
    path = directory / "checkpoint_info.json"
    for attempt in range(6):
        try:
            metadata_bytes = path.read_bytes()
            metadata = json.loads(metadata_bytes)
            if metadata["kind"] != "openonda.coupled_backup" or metadata["format_version"] != 13:
                raise ValueError("Require a native format-13 coupled checkpoint")
            config_digest = digest(
                json.dumps(metadata["config"], sort_keys=True, separators=(",", ":")).encode()
            )
            if config_digest != metadata["config_sha256"]:
                raise ValueError("Checkpoint configuration digest mismatch")
            content = {
                key: (directory / name).read_bytes()
                for key, name in metadata["checkpoint_files"].items()
            }
            if any(digest(value) != metadata["file_sha256"][key] for key, value in content.items()):
                raise ValueError("Committed checkpoint artifact digest mismatch")
            if path.read_bytes() != metadata_bytes:
                continue
            destination.mkdir()
            (destination / "checkpoint_info.json").write_bytes(metadata_bytes)
            paths = {}
            for key, value in content.items():
                paths[key] = destination / metadata["checkpoint_files"][key]
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                paths[key].write_bytes(value)
            return (
                metadata,
                paths,
                {
                    "source_checkpoint_info": str(path),
                    "checkpoint_info_sha256": digest(metadata_bytes),
                    "artifact_sha256": {key: digest(value) for key, value in content.items()},
                    "publication_attempts": attempt + 1,
                },
            )
        except (FileNotFoundError, ValueError, KeyError):
            if (
                path.exists()
                and "metadata_bytes" in locals()
                and path.read_bytes() == metadata_bytes
            ):
                raise
        time.sleep(0.15)
    raise RuntimeError("Checkpoint publication did not settle after six attempts")


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return decode_state({key: archive[key].copy() for key in archive.files})


def gaussian_planar_velocity(
    target, source, strength, radius, span, background, *, radius_scale=1.0
):
    """Evaluate the native infinite-filament formula in streamed float64 sums."""
    result = np.zeros((len(target), 3), dtype=np.float64)
    for first in range(0, len(source), 1024):
        selected = slice(first, first + 1024)
        delta = target[:, None, :2] - source[None, selected, :2]
        squared_distance = np.einsum("ijk,ijk->ij", delta, delta)
        inverse_radius_squared = 1.0 / (radius[None, selected] * radius_scale) ** 2
        weight = np.divide(
            -np.expm1(-squared_distance * inverse_radius_squared),
            squared_distance,
            out=np.broadcast_to(inverse_radius_squared, squared_distance.shape).copy(),
            where=squared_distance > 0,
        )
        weight *= strength[None, selected, 2] / (2.0 * np.pi * span)
        result[:, 0] -= np.sum(delta[:, :, 1] * weight, axis=1)
        result[:, 1] += np.sum(delta[:, :, 0] * weight, axis=1)
    return result + background


def face_geometry(mesh, geometry, name):
    patch = next(item for item in mesh["boundary"] if item["name"] == name)
    faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
    area = geometry["face_area"][faces]
    return (
        faces,
        geometry["face_centre"][faces],
        geometry["face_area_vector"][faces] / area[:, None],
        area,
    )


def trace_statistics(velocity, normal, area):
    scalar_normal = np.einsum("ij,ij->i", velocity, normal)
    tangential = velocity - scalar_normal[:, None] * normal
    return {
        "velocity_rms": float(np.sqrt(np.average(np.sum(velocity**2, axis=1), weights=area))),
        "velocity_maximum": float(np.max(np.linalg.norm(velocity, axis=1))),
        "normal_velocity_rms": float(np.sqrt(np.average(scalar_normal**2, weights=area))),
        "normal_velocity_maximum": float(np.max(abs(scalar_normal))),
        "tangential_velocity_rms": float(
            np.sqrt(np.average(np.sum(tangential**2, axis=1), weights=area))
        ),
        "tangential_velocity_maximum": float(np.max(np.linalg.norm(tangential, axis=1))),
        "integrated_normal_flux": float(np.dot(scalar_normal, area)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.case = args.case.resolve()
    output = args.output.resolve() / f"wall_trace_{stamp}"
    output.mkdir()
    checkpoint, paths, source_information = freeze_checkpoint(
        args.case / "solution/backups", output / "checkpoint"
    )
    fvm = load_npz(paths["fvm"])
    boundary = load_npz(paths["vpm_boundary_condition"])
    with h5py.File(paths["vpm"], "r") as saved:
        particles = {
            key: saved["particles"][key][:].astype(np.float64)
            for key in ("position", "vortex_strength", "core_radius", "particle_volume")
        }
        attributes = dict(saved["solver"].attrs)
    configuration = json.loads(attributes["numerical_configuration"])
    if configuration != checkpoint["config"]["vpm"]:
        raise ValueError("Stored VPM numerical configuration differs from the coupled checkpoint")
    induction = configuration["induction"]
    if induction["method"] != "PLANAR" or induction["kernel"] != "GAUSSIAN":
        raise ValueError("This diagnostic requires Gaussian PlanarInduction")
    if (
        abs(float(fvm["time"]) - checkpoint["time"]) > 1e-9
        or abs(float(attributes["time"]) - checkpoint["time"]) > 1e-9
    ):
        raise ValueError("Native FVM/VPM clocks disagree")
    if any(not np.isfinite(value).all() for value in particles.values()):
        raise ValueError("Non-finite particle fields")
    if np.any(particles["vortex_strength"][:, :2] != 0) or np.any(
        abs(particles["position"][:, 2] - induction["plane_z"]) > 1e-6
    ):
        raise ValueError("Sources violate planar geometry/strength conditions")
    geometry_data = {}
    for label, directory in (("coupled", args.case), ("reference", args.case / "reference_flow")):
        source = directory / "solution/fvm/mesh.npz"
        data = source.read_bytes()
        destination = output / f"{label}_mesh.npz"
        destination.write_bytes(data)
        mesh = load_native_mesh(destination)
        geometry = compute_mesh_geometry(mesh, compute_lsq=False)
        geometry_data[label] = mesh, geometry
        source_information[f"{label}_mesh_sha256"] = digest(data)
    mesh, geometry = geometry_data["coupled"]
    background = np.asarray(attributes["freestream_velocity"], dtype=np.float64)
    arguments = (
        particles["position"],
        particles["vortex_strength"],
        particles["core_radius"],
        induction["planar_span"],
        background,
    )
    arrays = {}
    statistics = {}
    for name in ("cylinder", "numericalBoundary"):
        faces, centres, normal, area = face_geometry(mesh, geometry, name)
        velocity = gaussian_planar_velocity(centres, *arguments)
        fvm_trace = fvm["velocity"][mesh["n_cells"] + faces - mesh["n_interior_faces"]]
        arrays.update(
            {
                f"{name}_centre": centres,
                f"{name}_normal": normal,
                f"{name}_area": area,
                f"{name}_vpm_velocity": velocity,
                f"{name}_fvm_velocity": fvm_trace,
            }
        )
        statistics[name] = {
            "face_count": len(faces),
            "vpm_trace": trace_statistics(velocity, normal, area),
            "fvm_trace": trace_statistics(fvm_trace, normal, area),
            "vpm_minus_fvm_trace": trace_statistics(velocity - fvm_trace, normal, area),
        }
        if name == "numericalBoundary":
            statistics[name]["vpm_minus_committed_boundary_history"] = trace_statistics(
                velocity - boundary["velocity"], normal, area
            )
            arrays["committed_boundary_velocity"] = boundary["velocity"]
        else:
            statistics[name]["unchanged_source_core_radius_sensitivity"] = {
                str(scale): trace_statistics(
                    gaussian_planar_velocity(centres, *arguments, radius_scale=scale), normal, area
                )
                for scale in (0.25, 0.5, 2.0)
            }
            angle = np.arctan2(centres[:, 1], centres[:, 0])
            radial_velocity = np.einsum(
                "ij,ij->i", velocity, centres / np.linalg.norm(centres, axis=1)[:, None]
            )
            modes = np.column_stack(
                [np.ones(len(angle))]
                + [
                    value
                    for order in range(1, 13)
                    for value in (np.cos(order * angle), np.sin(order * angle))
                ]
            )
            coefficient = np.linalg.lstsq(
                modes * np.sqrt(area)[:, None], radial_velocity * np.sqrt(area), rcond=None
            )[0]
            statistics[name]["radial_velocity_fourier_modes"] = {
                "mean": float(coefficient[0]),
                "amplitudes": {
                    str(order): float(np.linalg.norm(coefficient[2 * order - 1 : 2 * order + 1]))
                    for order in range(1, 13)
                },
                "fit_residual_rms": float(
                    np.sqrt(np.average((radial_velocity - modes @ coefficient) ** 2, weights=area))
                ),
            }
    surface_geometry = {}
    for label, (local_mesh, local_geometry) in geometry_data.items():
        faces, centres, normal, area = face_geometry(local_mesh, local_geometry, "cylinder")
        radial_distance = np.linalg.norm(centres[:, :2], axis=1)
        surface_geometry[label] = {
            "face_count": len(faces),
            "surface_area": float(np.sum(area)),
            "face_radius_minimum": float(np.min(radial_distance)),
            "face_radius_mean": float(np.mean(radial_distance)),
            "face_radius_maximum": float(np.max(radial_distance)),
            "owner_wall_distance_mean": float(np.mean(local_geometry["wall_distance"][faces])),
        }
        if label == "reference":
            distance, nearest = cKDTree(centres).query(arrays["cylinder_centre"])
            surface_geometry["reference"]["coupled_wall_centre_nearest_distance_maximum"] = float(
                np.max(distance)
            )
            surface_geometry["reference"]["coupled_wall_normal_difference_maximum"] = float(
                np.max(np.linalg.norm(normal[nearest] - arrays["cylinder_normal"], axis=1))
            )
    radial_distance = np.linalg.norm(particles["position"][:, :2], axis=1)
    report = {
        "schema": "openonda-planar-cylinder-wall-trace/1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "native_committed_time": checkpoint["time"],
        "native_coupling_step": checkpoint["coupling_step"],
        "source_information": source_information,
        "particle_geometry": {
            "count": len(radial_distance),
            "minimum_radius": float(np.min(radial_distance)),
            "centres_inside_radius_0_5": int(np.sum(radial_distance < 0.5)),
            "centres_within_one_core_radius_of_wall": int(
                np.sum(radial_distance < 0.5 + particles["core_radius"])
            ),
            "core_radius_minimum": float(np.min(particles["core_radius"])),
            "core_radius_maximum": float(np.max(particles["core_radius"])),
            "represented_span": induction["planar_span"],
            "volume_minimum": float(np.min(particles["particle_volume"])),
            "volume_maximum": float(np.max(particles["particle_volume"])),
        },
        "freestream_velocity": background.tolist(),
        "statistics": statistics,
        "cylinder_geometry_comparison": surface_geometry,
        "method": {
            "formula": "u = freestream + sum Gamma_z/(2 pi span) * (1-exp(-r_xy^2/sigma^2))/r_xy^2 * (-r_y,r_x,0)",
            "precision": "float64 streamed NumPy summation of the exact stored float32 source values",
            "normal_convention": "FVM outward face normal; cylinder normals point into the solid",
            "weights": "Actual native FVM face areas",
            "source_script": str(Path(__file__).resolve()),
        },
        "limitations": [
            "The checkpoint is one accepted endpoint, not a complete-cycle boundary-condition control experiment.",
            "Nonzero VPM wall trace shows the represented free-space field is not exactly a no-slip cylinder solution. It does not prove how much of the force deficit it causes.",
            "The physical cylinder wall itself is enforced as no-slip by FVM; VPM trace evaluated there is a diagnostic, not the FVM wall boundary value.",
            "Changing only Gaussian core radii is an instantaneous representation sensitivity; no evolved trajectory or force improvement is inferred.",
            "At this time the startup taper is complete and the native background velocity is exactly [1,0,0].",
        ],
    }
    np.savez_compressed(output / "wall_trace_fields.npz", **arrays)
    (output / "wall_trace_statistics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(output), "time": checkpoint["time"], "statistics": statistics}, indent=2
        )
    )


if __name__ == "__main__":
    main()
