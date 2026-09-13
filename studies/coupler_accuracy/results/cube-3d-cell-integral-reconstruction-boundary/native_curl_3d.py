#!/usr/bin/env python3
"""Separate the native FVM curl stencil from the continuous particle curl.

Frozen 3D measurement only. At the cube wall the native stencil consumes the
prescribed zero face velocity. Particle slip there is reported independently.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

import openonda.fvm as fvm
from source.coupler.boundary import evaluate_vpm_velocity, tangential_normal_velocity_gradient
from source.coupler.interpolation import FVMVelocityInterpolator
from source.coupler.renewal_projection import geometric_renewal_mask
from source.solvers.fvm.fields.gradients import _resolve_gradient_fn
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file, setup_for
from studies.coupler_accuracy.joint_reconstruction_3d import CubePanelResponse
from studies.coupler_accuracy.native_face_trace_3d import NativeFaceTrace


def native_gauss_curl_stencil(mesh, geometry, cells):
    """Return a sparse curl map and its cell/face-value dependency indices.

    Uses sum_faces(Sf x Uface)/V, with the exact native interpolation weights.
    Indices >= n_cells are stored boundary-face values, as in the FVM solver.
    Partitioned, cyclic and empty meshes are outside this offline experiment.
    """
    if np.any(np.asarray(mesh.get("boundary_neighbour_cell", [-1])) >= 0):
        raise ValueError("This experiment requires uncoupled physical boundaries")
    if any(b.get("velocity_type", b.get("type")) == "empty" for b in mesh["boundary"]):
        raise ValueError("A fully 3D mesh is required")
    cells = np.asarray(cells, dtype=int)
    n_cells, n_interior = mesh["n_cells"], mesh["n_interior_faces"]
    mapping = np.full(n_cells, -1)
    mapping[cells] = np.arange(len(cells))
    owner, neighbour = mesh["owners"], mesh["neighbours"]
    weight = geometry["face_interpolation_weight"]
    sf, volume = geometry["face_area_vector"], geometry["cell_volume"]
    rows, columns, values = [], [], []

    def add(target_cells, source_cells, vectors):
        for i, j, axis, sign in ((0, 1, 2, -1), (0, 2, 1, 1), (1, 0, 2, 1),
                                 (1, 2, 0, -1), (2, 0, 1, -1), (2, 1, 0, 1)):
            rows.append(3 * mapping[target_cells] + i)
            columns.append(3 * source_cells + j)
            values.append(sign * vectors[:, axis] / volume[target_cells])

    for target, sign in ((owner[:n_interior], 1), (neighbour[:n_interior], -1)):
        faces = np.flatnonzero(mapping[target] >= 0)
        add(target[faces], owner[faces], sign * sf[faces] * (1 - weight[faces, None]))
        add(target[faces], neighbour[faces], sign * sf[faces] * weight[faces, None])
    boundary = np.arange(n_interior, mesh["n_faces"])
    boundary = boundary[mapping[owner[boundary]] >= 0]
    add(owner[boundary], n_cells + boundary - n_interior, sf[boundary])
    columns = np.concatenate(columns)
    source_ids = np.unique(columns // 3)
    compact_columns = 3 * np.searchsorted(source_ids, columns // 3) + columns % 3
    matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), compact_columns)),
                        shape=(3 * len(cells), 3 * len(source_ids))).tocsr()
    matrix.eliminate_zeros()
    return matrix, source_ids


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    metadata_path = args.study / "joint-reconstruction-3d.json"
    if not metadata_path.exists():
        metadata_path = args.study / "native-reconstruction-3d.json"
    source_paths = [Path(__file__), args.study / "held-fields.npz", metadata_path,
                    *[args.oracle / name for name in ("initial-cell-fields.npz", "full-native-mesh.npz",
                                                     "small-native-mesh.npz", "cell-and-face-map.npz")]]
    source_paths += [ROOT / p for p in (
        "studies/coupler_accuracy/cube_boundary_oracle.py",
        "studies/coupler_accuracy/joint_reconstruction_3d.py",
        "studies/coupler_accuracy/native_face_trace_3d.py",
        "source/coupler/boundary.py", "source/coupler/interpolation.py",
        "source/coupler/renewal_projection.py",
        "source/solvers/fvm/core/solver.py", "source/solvers/fvm/mesh/geometry.py",
        "source/solvers/fvm/fields/gradients.py", "source/solvers/fvm/fields/diagnostics.py",
        "source/solvers/fvm/coupling/coupler_interface.py",
        "source/solvers/fvm/assemble/diffusion.py", "source/solvers/fvm/solve/simple_solver.py",
        "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py",
        "source/solvers/vpm/boundary_elements/panels/solver/linear_solvers.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py",
        "tutorials/coupled_fvm_vpm/02_cube_flow/assets/cube.stl")]
    sources = [hash_file(p) for p in source_paths]
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2) + "\n")
    for path in source_paths:
        if path.parent == Path(__file__).parent:
            (args.output / path.name).write_bytes(path.read_bytes())
    with np.load(args.study / "held-fields.npz", allow_pickle=False) as data:
        fields = {k: data[k].copy() for k in data.files}
    with np.load(args.oracle / "initial-cell-fields.npz", allow_pickle=False) as data:
        centres, velocity, pressure = (data[k].copy() for k in ("centres", "velocity", "pressure"))
    metadata = json.loads(metadata_path.read_text())
    mesh = load_native_mesh(args.oracle / "full-native-mesh.npz")
    small_mesh = load_native_mesh(args.oracle / "small-native-mesh.npz")
    small_geometry = compute_mesh_geometry(small_mesh, gradient_scheme="gauss", compute_lsq=False)
    patch = next(b for b in small_mesh["boundary"] if b["name"] == "numericalBoundary")
    cut = slice(patch["start_face"], patch["start_face"] + patch["n_faces"])
    face = small_geometry["face_centre"][cut]
    face_area = small_geometry["face_area"][cut]
    face_normal = small_geometry["face_area_vector"][cut] / face_area[:, None]
    with np.load(args.oracle / "cell-and-face-map.npz", allow_pickle=False) as data:
        face_ids, face_signs = data["face_ids"][cut], data["signs"][cut]
    ids = cKDTree(centres).query(fields["position"])[1]
    with fvm.create_fvm_solver(setup_for(mesh, "native-curl", 0.01, 1),
                               mesh=copy.deepcopy(mesh), case_dir=args.output / "donor") as solver:
        solver.set_initial_state(velocity, pressure)
        omega, volume = solver.get_vorticity_field().copy(), solver.get_cell_volume().copy()
        stencil, source_ids = native_gauss_curl_stencil(solver.mesh_data, solver.geo_data, ids)
        replay = (stencil @ solver.velocity[source_ids].ravel()).reshape(-1, 3)
        np.testing.assert_allclose(replay, omega[ids], rtol=0, atol=5e-13)
        boundary_ids = source_ids[source_ids >= mesh["n_cells"]]
        wall = next(p for p in solver.mesh_data["boundary"] if p["name"] == "cube")
        wall_face_ids = np.arange(wall["start_face"], wall["start_face"] + wall["n_faces"])
        wall_ghosts = mesh["n_cells"] + wall_face_ids - mesh["n_interior_faces"]
        assert np.all(np.isin(boundary_ids, wall_ghosts))
        np.testing.assert_array_equal(solver.velocity[boundary_ids], 0)
        expected_normal = solver.volumetric_face_flux[face_ids] * face_signs / face_area
        gradient = solver.get_velocity_gradient_field()
        trace = FVMVelocityInterpolator(centres, cKDTree(centres))
        jacobian = np.stack([trace.sample_cell_field(face, gradient[:, i, :]) for i in range(3)], axis=1).transpose(0, 2, 1)
        expected_tangent_gradient = tangential_normal_velocity_gradient(jacobian, face_normal)
        native_trace = NativeFaceTrace(solver.mesh_data, solver.geo_data, face_ids, face_signs)
        # Preserve the old audit's Sf/sum(triangle areas) convention for its
        # existing metrics. Warped faces make its norm slightly less than one.
        # The new native-flux comparison uses the true unit vector Sf/|Sf|.
        old_normal_length = np.linalg.norm(face_normal, axis=1)
        np.testing.assert_allclose(native_trace.normal, face_normal / old_normal_length[:, None],
                                   rtol=0, atol=1e-13)
        pressure_gradient = _resolve_gradient_fn(solver.geo_data)(
            solver.kinematic_pressure, solver.mesh_data, solver.geo_data)[:mesh["n_cells"], :, 0]
        native_face = native_trace.evaluate(velocity, gradient, solver.kinematic_pressure, pressure_gradient)
        expected_native_tangent_gradient = native_face["native_flux_tangential_gradient"]

    h = metadata["particle_spacing"]
    retained = np.linalg.norm(omega, axis=1) >= 0.02
    initial_position = centres[retained].astype(np.float32).astype(float)
    initial_strength = (omega * volume[:, None])[retained].astype(np.float32).astype(float)
    preserved = ~geometric_renewal_mask(initial_position, metadata["renewal_bounds"], particle_spacing=h)
    panel = CubePanelResponse()
    fluid = source_ids < mesh["n_cells"]
    targets = centres[source_ids[fluid]]
    u, background = panel.velocity_operator(targets, fields["renewable_position"], fields["radius"])
    outer, _ = panel.velocity_operator(targets, initial_position[preserved], h)
    background += outer @ initial_strength[preserved].ravel()
    # Evaluate just outside each real panel, avoiding its surface jump convention.
    wall_points = panel.centres + 1e-6 * panel.normal
    wall_u, wall_background = panel.velocity_operator(wall_points, fields["renewable_position"], fields["radius"])
    outer_wall, _ = panel.velocity_operator(wall_points, initial_position[preserved], h)
    wall_background += outer_wall @ initial_strength[preserved].ravel()
    boundary_u, boundary_background = panel.velocity_operator(face, fields["renewable_position"], fields["radius"])
    outer_boundary, _ = panel.velocity_operator(face, initial_position[preserved], h)
    boundary_background += outer_boundary @ initial_strength[preserved].ravel()
    epsilon = 1e-4 * h

    def normal_derivative(points, normal, step):
        derivative, base = None, None
        for sign in (1, -1):
            a, b = panel.velocity_operator(points + sign * step * normal, fields["renewable_position"], fields["radius"])
            outer, _ = panel.velocity_operator(points + sign * step * normal, initial_position[preserved], h)
            b += outer @ initial_strength[preserved].ravel()
            if derivative is None:
                derivative, base = a / (2 * step), b / (2 * step)
            else:
                derivative -= a / (2 * step)
                base -= b / (2 * step)
        return derivative, base

    derivative, derivative_background = normal_derivative(face, face_normal, epsilon)
    check, check_background = normal_derivative(face[:16], face_normal[:16], 2 * epsilon)
    fd_relative_difference = np.linalg.norm(derivative[:48] - check) / np.linalg.norm(check)
    assert fd_relative_difference < 1e-6
    np.testing.assert_allclose(derivative_background[:48], check_background, rtol=0, atol=1e-7)
    bounds = np.column_stack((small_mesh["vertex_position"].min(axis=0),
                              small_mesh["vertex_position"].max(axis=0))).ravel()
    field_adapter = SimpleNamespace(particles=SimpleNamespace(n_particles_total=len(fields["renewable_position"]) + int(preserved.sum())))
    records, boundary_fields = [], {}
    for row in metadata["results"]:
        name = row["name"]
        strength = fields[name + "__strength"].ravel()
        values = np.zeros((len(source_ids), 3))
        values[fluid] = (u @ strength + background).reshape(-1, 3)
        native_curl = (stencil @ values.ravel()).reshape(-1, 3)
        physical_curl = fields[name + "__velocity_curl"]
        surface = (wall_u @ strength + wall_background).reshape(-1, 3)
        normal = np.sum(surface * panel.normal, axis=1)
        tangent = surface - normal[:, None] * panel.normal
        boundary_values = (boundary_u @ strength + boundary_background).reshape(-1, 3)
        corrected, flux = evaluate_vpm_velocity(field_adapter, face, face_normal, face_area,
                                               freestream_velocity=np.array([1., 0., 0.]),
                                               fvm_box=bounds, particle_spacing=h,
                                               evaluated_velocity=boundary_values)
        normal_values = np.sum(corrected * face_normal, axis=1)
        dudn = (derivative @ strength + derivative_background).reshape(-1, 3)
        tangent_gradient = dudn - np.sum(dudn * face_normal, axis=1)[:, None] * face_normal
        native_dudn = dudn / old_normal_length[:, None]
        native_tangent_gradient = native_dudn - np.sum(native_dudn * native_trace.normal, axis=1)[:, None] * native_trace.normal
        record = {"name": name,
                  "held_native_curl_relative_error": field_rms(native_curl - omega[ids], volume[ids]) / field_rms(omega[ids], volume[ids]),
                  "held_native_vs_continuous_curl_relative_difference": field_rms(native_curl - physical_curl, volume[ids]) / field_rms(omega[ids], volume[ids]),
                  "wall_tangential_velocity_rms_over_Uinf": field_rms(tangent, panel.area),
                  "wall_normal_velocity_rms_over_Uinf": field_rms(normal[:, None], panel.area),
                  "boundary_normal_velocity_rms_error_over_Uinf": field_rms((normal_values - expected_normal)[:, None], face_area),
                  "boundary_tangential_normal_gradient_rms_error": field_rms(tangent_gradient - expected_tangent_gradient, face_area),
                  "boundary_native_flux_tangential_normal_gradient_rms_error": field_rms(native_tangent_gradient - expected_native_tangent_gradient, native_trace.area),
                  "boundary_flux": flux}
        boundary_fields[name + "__normal_velocity"] = normal_values
        boundary_fields[name + "__tangential_normal_gradient"] = tangent_gradient
        boundary_fields[name + "__native_tangential_normal_gradient"] = native_tangent_gradient
        print(json.dumps(record), flush=True)
        records.append(record)
    report = {"schema": "openonda-frozen-native-curl-3d/1", "spatial_dimensions": 3,
              "native_stencil_replay_max_error": float(np.max(np.abs(replay - omega[ids]))),
              "boundary_faces": len(face), "boundary_derivative_step": epsilon,
              "boundary_derivative_step_halving_relative_difference": float(fd_relative_difference),
              "native_flux_vs_interpolated_tangential_gradient_rms": field_rms(
                  expected_native_tangent_gradient - expected_tangent_gradient, face_area),
              "maximum_legacy_normal_length_defect": float(np.max(np.abs(old_normal_length - 1))),
              "results": records,
              "sources": sources,
              "limitations": ["The native curl uses prescribed no-slip wall values even if the particle velocity slips.",
                              "The original boundary-gradient metric uses an interpolated cell-gradient trace; a separate metric uses the actual native momentum diffusion trace.",
                              "Particle derivatives are centred differences of f64 complete velocity, with a step check.",
                              "This is a frozen coarse-cube diagnostic, not a new transfer method."]}
    (args.output / "native-curl-3d.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "boundary-fields.npz", position=face, normal=face_normal, area=face_area,
                        native_unit_normal=native_trace.normal, native_vector_area=native_trace.area,
                        fvm_normal_velocity=expected_normal,
                        fvm_interpolated_tangential_normal_gradient=expected_tangent_gradient,
                        fvm_native_flux_tangential_normal_gradient=expected_native_tangent_gradient,
                        **boundary_fields)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, default=ROOT / "studies/coupler_accuracy/results/cube-3d-oracle")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
