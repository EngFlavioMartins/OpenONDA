"""Qualify moment lifting against direct surface integrals, including walls."""

import numpy as np

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.native_normal_moment_lift_3d import NormalMomentLift
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def setup():
    mesh = box_mesh_3d(np.array([-.7, -.2, .3, 1.1]), np.array([-.4, 0., .5, 1.2]), np.array([-.6, -.1, .2, .8]))
    mesh["vertex_position"] += np.random.default_rng(718).normal(0, .007, mesh["vertex_position"].shape)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    volume, _ = native.cell_geometry(geometry["cell_centre"])
    return mesh, geometry, native, volume


def independent_integrals(mesh, native, centre, gradient):
    barycentric = (np.ones((3, 3)) + 3 * np.eye(3)) / 6
    relative = np.einsum("qv,tvi->tqi", barycentric, native.triangles - centre[native.face_ids, None])
    sf = np.cross(native.triangles[:, 1] - native.triangles[:, 0], native.triangles[:, 2] - native.triangles[:, 0]) / 2
    area = np.linalg.norm(sf, axis=1)
    value = np.einsum("tqi,ti->tq", relative, gradient[native.face_ids])
    flux, first = np.zeros(mesh["n_faces"]), np.zeros((mesh["n_faces"], 3))
    np.add.at(flux, native.face_ids, area * value.mean(axis=1))
    np.add.at(first, native.face_ids, area[:, None] * (relative * value[:, :, None]).mean(axis=1))
    cell = np.zeros((mesh["n_cells"], 3))
    np.add.at(cell, mesh["owners"], first)
    np.add.at(cell, mesh["neighbours"], -first[:mesh["n_interior_faces"]])
    energy = float(np.sum(area * np.mean(value**2, axis=1)))
    return flux, first, cell, energy


def test_lift_matches_cell_moments_keeps_wall_and_flux_and_certifies_energy():
    mesh, geometry, native, volume = setup()
    locked = np.arange(mesh["n_interior_faces"], mesh["n_interior_faces"] + 9)
    free = np.setdiff1d(np.arange(mesh["n_faces"]), locked)
    lift = NormalMomentLift.from_mesh(mesh, native, geometry["face_centre"], volume, free)
    target = np.random.default_rng(739).normal(size=(mesh["n_cells"], 3)) * volume[:, None] * .03
    answer = lift.solve(target, tolerance=1e-13)
    flux, moments, cells, energy = independent_integrals(mesh, native, lift.centre, answer["face_gradient"])
    np.testing.assert_allclose(flux, 0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(cells, target, rtol=0, atol=1e-13)
    np.testing.assert_allclose(moments, answer["face_first_moment"], rtol=0, atol=2e-16)
    np.testing.assert_array_equal(answer["face_gradient"][locked], 0)
    np.testing.assert_allclose(energy, answer["diagnostics"]["normal_change_integrated_squared"], rtol=2e-13, atol=1e-16)
    assert abs(answer["diagnostics"]["relative_primal_dual_energy_gap"]) < 1e-9


def test_locked_boundary_cannot_change_total_cell_velocity_integral():
    mesh, geometry, native, volume = setup()
    lift = NormalMomentLift.from_mesh(mesh, native, geometry["face_centre"], volume, np.arange(mesh["n_interior_faces"]))
    target = volume[:, None] * np.array([.02, -.01, .03])
    answer = lift.solve(target, tolerance=1e-12)
    _, _, recovered, _ = independent_integrals(mesh, native, lift.centre, answer["face_gradient"])
    np.testing.assert_allclose(recovered.sum(axis=0), 0, rtol=0, atol=2e-16)
    bound = np.linalg.norm(target.sum(axis=0)) / volume.sum()
    assert answer["diagnostics"]["velocity_constraint_rms"] >= bound * (1 - 1e-12)
    assert answer["diagnostics"]["velocity_constraint_rms"] > .03
