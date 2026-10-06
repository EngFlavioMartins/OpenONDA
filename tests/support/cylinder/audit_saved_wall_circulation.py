"""Read-only wall-circulation targets from one frozen native planar checkpoint.

Compare wall velocity taper and circulation-cell exclusion before Gaussian
representation correction, blending, moment correction and pruning. These
frozen target differences do not establish an evolved force-amplitude effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from source.coupler.geometry import SolidBoundary, TriangulatedWall
from source.coupler.interpolation import FVMVelocityInterpolator
from source.solvers.fvm.fields.gradients import compute_lsq_geometry, compute_lsq_gradient
from source.solvers.fvm.io.backup import decode_state
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.coupled import configure_cyclic_boundaries
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_donor_geometry(directory, state_path, mesh_filename="coupled_mesh.npz"):
    mesh = load_native_mesh(directory / mesh_filename)
    for patch in mesh["boundary"]:
        if patch["name"] in ("zmin", "zmax"):
            patch.update(
                velocity_type="cyclic",
                pressure_type="cyclic",
                neighbour_patch="zmax" if patch["name"] == "zmin" else "zmin",
            )
    geometry = compute_mesh_geometry(mesh, gradient_scheme="lsq", compute_lsq=False)
    configure_cyclic_boundaries(mesh, geometry)
    geometry.update(compute_lsq_geometry(mesh, geometry))
    with np.load(state_path, allow_pickle=False) as stored:
        state = decode_state({key: stored[key].copy() for key in stored.files})
    velocity = state["velocity"]
    gradient = compute_lsq_gradient(velocity, mesh, geometry)[: mesh["n_cells"]]
    vertices = mesh["vertex_position"]
    triangles = []
    for patch in mesh["boundary"]:
        if patch["type"] != "wall":
            continue
        for face in range(patch["start_face"], patch["start_face"] + patch["n_faces"]):
            nodes = np.asarray(mesh["faces"][face], dtype=np.int64)
            polygon = vertices[nodes[nodes >= 0]]
            centre = polygon.mean(axis=0)
            for edge in range(len(polygon)):
                triangle = np.array([centre, polygon[edge], polygon[(edge + 1) % len(polygon)]])
                if np.linalg.norm(np.cross(triangle[1] - centre, triangle[2] - centre)) > 0:
                    triangles.append(triangle)
    outer_faces = np.concatenate(
        [
            np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
            for patch in mesh["boundary"]
            if patch["type"] != "wall"
        ]
    )
    outer_centres = geometry["face_centre"][outer_faces]
    domain = np.column_stack((outer_centres.min(axis=0), outer_centres.max(axis=0))).ravel()
    wall = TriangulatedWall(np.asarray(triangles), domain)
    boundary = SolidBoundary((wall,))
    centres = geometry["cell_centre"]
    trace = FVMVelocityInterpolator(centres, cKDTree(centres), solid_boundary=boundary)
    return mesh, geometry, state, gradient, boundary, wall, trace


def gaussian_velocity_and_gradient(points, sources, strength, sigma):
    """Independent streamed planar Gaussian velocity/Jacobian in float64."""
    velocity = np.zeros_like(points)
    gradient = np.zeros((len(points), 3, 3))
    inverse_sigma2 = 1 / sigma**2
    for start in range(0, len(sources), 2048):
        delta = points[:, None, :2] - sources[None, start : start + 2048, :2]
        squared = np.einsum("fpi,fpi->fp", delta, delta)
        q = squared * inverse_sigma2
        decay = np.exp(-q)
        f = np.divide(
            -np.expm1(-q), squared, out=np.full_like(q, inverse_sigma2), where=squared > 0
        )
        derivative = inverse_sigma2**2 * (-0.5 + q / 3 - q**2 / 8)
        regular = q > 1e-4
        derivative[regular] = (q[regular] * decay[regular] + np.expm1(-q[regular])) / squared[
            regular
        ] ** 2
        coefficient = strength[None, start : start + 2048] / (2 * math.pi)
        dx, dy = delta[:, :, 0], delta[:, :, 1]
        velocity[:, 0] -= np.sum(coefficient * dy * f, axis=1)
        velocity[:, 1] += np.sum(coefficient * dx * f, axis=1)
        gradient[:, 0, 0] -= np.sum(coefficient * dy * 2 * dx * derivative, axis=1)
        gradient[:, 0, 1] -= np.sum(coefficient * (f + 2 * dy**2 * derivative), axis=1)
        gradient[:, 1, 0] += np.sum(coefficient * (f + 2 * dx**2 * derivative), axis=1)
        gradient[:, 1, 1] += np.sum(coefficient * dx * 2 * dy * derivative, axis=1)
    return velocity, gradient


def velocity_statistics(velocity, normals, area):
    normal = np.einsum("fi,fi->f", velocity, normals)
    tangent = velocity - normal[:, None] * normals
    return {
        "velocity_rms": float(np.sqrt(np.average(np.sum(velocity**2, axis=1), weights=area))),
        "velocity_maximum": float(np.linalg.norm(velocity, axis=1).max()),
        "normal_velocity_rms": float(np.sqrt(np.average(normal**2, weights=area))),
        "tangential_velocity_rms": float(
            np.sqrt(np.average(np.sum(tangent**2, axis=1), weights=area))
        ),
    }


def audit(directory):
    metadata_path = directory / "checkpoint/checkpoint_info.json"
    metadata = json.loads(metadata_path.read_text())
    state_path = metadata_path.parent / metadata["checkpoint_files"]["fvm"]
    paths = (
        metadata_path,
        state_path,
        directory / "coupled_mesh.npz",
        directory / "wall_trace_fields.npz",
    )
    hashes = {str(path): digest(path) for path in paths}
    mesh, geometry, state, gradient, boundary, wall, trace = load_donor_geometry(
        directory, state_path
    )
    if wall.revision not in metadata["config"]["solid_geometry"]["wall_revisions"]:
        raise ValueError("Reconstructed wall differs from the native checkpoint wall revision")
    spacing = metadata["config"]["vpm"]["viscous"]["particle_spacing"]
    anchor = np.asarray(metadata["config"]["transfer_lattice"]["anchor"])
    coordinates = [
        anchor[axis]
        + spacing
        * np.arange(
            math.ceil((-0.68 - anchor[axis]) / spacing),
            math.floor((0.68 - anchor[axis]) / spacing) + 1,
        )
        for axis in (0, 1)
    ]
    x, y = np.meshgrid(*coordinates, indexing="ij")
    positions = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    solid = boundary.contains(positions, include_boundary=False)
    targets = {"untapered": np.zeros(len(positions)), "tapered": np.zeros(len(positions))}
    velocity = state["velocity"][: mesh["n_cells"]]

    def sample(points, taper):
        distance = boundary.signed_distance(points)
        fluid = ~boundary.contains(points, include_boundary=False)
        result = np.zeros_like(points)
        if np.any(fluid):
            result[fluid] = trace.prepare(points[fluid]).sample(velocity, gradient)
        if taper:
            phase = np.clip(distance / spacing, 0, 1)
            result *= (phase**2 * (3 - 2 * phase))[:, None]
        return result

    for axis, component, sign in ((0, 1, 1.0), (1, 0, -1.0)):
        for side in (-1.0, 1.0):
            faces = positions.copy()
            faces[:, axis] += side * spacing / 2
            untapered = sample(faces, False)
            distance = boundary.signed_distance(faces)
            phase = np.clip(distance / spacing, 0, 1)
            targets["untapered"] += sign * side * spacing * untapered[:, component]
            targets["tapered"] += (
                sign * side * spacing * untapered[:, component] * phase**2 * (3 - 2 * phase)
            )
    native_target = targets["tapered"].copy()
    native_target[solid] = 0
    with np.load(directory / "wall_trace_fields.npz", allow_pickle=False) as stored:
        fields = {key: stored[key].copy() for key in stored.files}
    rows, arrays = {}, {"positions": positions, "solid_interior": solid, **targets}
    for taper in ("tapered", "untapered"):
        for support in ("exterior_nodes_only", "complete_cell_circulation"):
            target = targets[taper].copy()
            if support == "exterior_nodes_only":
                target[solid] = 0
            difference = target - native_target
            selected = np.abs(difference) > 1e-16
            name = f"{taper}_{support}"
            row = {
                "changed_nodes": int(selected.sum()),
                "absolute_strength_difference": float(np.sum(np.abs(difference))),
                "circulation_difference": float(np.sum(difference)),
                "linear_impulse_difference": [
                    float(0.5 * np.sum(positions[:, 1] * difference)),
                    float(-0.5 * np.sum(positions[:, 0] * difference)),
                    0.0,
                ],
            }
            for patch in ("cylinder", "numericalBoundary"):
                points = fields[f"{patch}_centre"]
                normals = fields[f"{patch}_normal"]
                area = fields[f"{patch}_area"]
                delta, jacobian = gaussian_velocity_and_gradient(
                    points, positions[selected], difference[selected], spacing
                )
                row[patch] = {
                    "raw_target_induced_velocity_change": velocity_statistics(delta, normals, area),
                    "existing_particle_velocity_plus_raw_target_change": velocity_statistics(
                        fields[f"{patch}_vpm_velocity"] + delta, normals, area
                    ),
                }
                derivative = np.einsum("fij,fj->fi", jacobian, normals)
                tangent_derivative = (
                    derivative - np.einsum("fi,fi->f", derivative, normals)[:, None] * normals
                )
                row[patch]["raw_target_tangential_normal_gradient_change_rms"] = float(
                    np.sqrt(np.average(np.sum(tangent_derivative**2, axis=1), weights=area))
                )
                arrays[f"{name}_{patch}_velocity_change"] = delta
                arrays[f"{name}_{patch}_jacobian_change"] = jacobian
            rows[name] = row

    contours = {}
    for name, y_values in (
        ("whole_wall_rectangle", coordinates[1]),
        ("upper_wall_rectangle", coordinates[1][coordinates[1] > 0]),
        ("lower_wall_rectangle", coordinates[1][coordinates[1] < 0]),
    ):
        selected = (positions[:, 1] >= y_values[0] - 1e-12) & (
            positions[:, 1] <= y_values[-1] + 1e-12
        )
        contours[name] = {}
        for taper in (False, True):
            contour = 0.0
            for axis, component, sign, values, bounds in (
                (
                    0,
                    1,
                    1.0,
                    y_values,
                    (coordinates[0][0] - spacing / 2, coordinates[0][-1] + spacing / 2),
                ),
                (
                    1,
                    0,
                    -1.0,
                    coordinates[0],
                    (y_values[0] - spacing / 2, y_values[-1] + spacing / 2),
                ),
            ):
                for side, bound in zip((-1.0, 1.0), bounds, strict=True):
                    query = np.zeros((len(values), 3))
                    query[:, axis] = bound
                    query[:, 1 - axis] = values
                    contour += sign * side * spacing * np.sum(sample(query, taper)[:, component])
            target = targets["tapered" if taper else "untapered"]
            complete = float(np.sum(target[selected]))
            exterior = float(np.sum(target[selected & ~solid]))
            contours[name]["tapered" if taper else "untapered"] = {
                "independent_contour_circulation": float(contour),
                "complete_cell_circulation": complete,
                "complete_cell_contour_error": complete - float(contour),
                "exterior_nodes_only_circulation": exterior,
                "exterior_nodes_only_contour_error": exterior - float(contour),
            }
    if any(digest(path) != value for path, value in hashes.items()):
        raise ValueError("Frozen evidence changed during analysis")
    return {
        "scope": "Frozen FVM wall targets before Gaussian representation correction, blending, moment correction and pruning; adding raw target differences to the recorded VPM field is a local diagnostic, not a native renewal or evolved cylinder force result",
        "native_committed_time": float(state["time"]),
        "particle_spacing": spacing,
        "inputs_sha256": hashes,
        "native_wall_revision": wall.revision,
        "lattice_anchor": anchor.tolist(),
        "circulation_rectangles": contours,
        "changes_from_native_tapered_exterior_target": rows,
    }, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    report, arrays = audit(options.directory.resolve())
    with options.output.with_suffix(".npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with options.output.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
