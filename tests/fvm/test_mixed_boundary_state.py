"""Mixed boundary state must stay coherent through public field setters."""

import numpy as np
import pytest

from source.solvers.fvm import (
    BoundaryConfig,
    FVMSetup,
    LinearSolverConfig,
    TimeConfig,
    TransportConfig,
)
from source.solvers.fvm.factory import create_fvm_solver
from source.solvers.fvm.fields.gradients import _resolve_gradient_fn
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d


@pytest.fixture
def solver(tmp_path):
    mesh = box_mesh_3d(np.linspace(-1, 1, 5), np.linspace(-1, 1, 4), np.linspace(-1, 1, 3))
    mesh["vertex_position"] = (
        mesh["vertex_position"] @ np.array([[1, 0.3, -0.2], [0.1, 1.2, 0.4], [0.2, -0.1, 0.8]]).T
    )
    setup = FVMSetup(
        case_name="mixed_boundary_state",
        time=TimeConfig(time_step_size=0.01, end_time=0.02),
        linear=LinearSolverConfig(linear_solver="spsolve"),
        transport=TransportConfig(density=1, kinematic_viscosity=0.01),
        boundaries=[BoundaryConfig.inlet(p["name"], [0.2, 0.3, -0.1]) for p in mesh["boundary"]],
        initial_velocity=[0.2, 0.3, -0.1],
    )
    with create_fvm_solver(setup, case_dir=tmp_path, mesh=mesh) as instance:
        yield instance


def data(solver):
    face = solver.get_boundary_face_centre_coordinates("inlet")
    normal = solver.get_boundary_face_normal("inlet")
    un = 0.3 + face @ [0.1, 0.2, -0.05]
    gt = np.tile([0.2, -0.7, 0.4], (len(face), 1))
    gt -= np.sum(gt * normal, axis=1)[:, None] * normal
    return normal, un, gt


@pytest.mark.parametrize("setter", ["set_initial_state", "set_initial_velocity"])
def test_initial_field_setters_refresh_mixed_faces_and_histories(solver, setter):
    normal, un, gt = data(solver)
    solver.set_normal_velocity_tangential_gradient_boundary_condition(un, gt, "inlet")
    x = solver.get_cell_centre_coordinates()
    velocity = x @ np.array([[0.2, -0.7, 1.3], [0.6, -0.1, -0.4], [-0.8, 1.2, -0.1]])
    if setter == "set_initial_state":
        solver.set_initial_state(velocity, x @ [0.1, 0.3, -0.2])
    else:
        solver.set_initial_velocity(velocity)
    patch = next(b for b in solver.boundaries if b["name"] == "inlet")
    faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
    ghosts = solver.mesh_data["n_cells"] + faces - solver.mesh_data["n_interior_faces"]
    owners = solver.mesh_data["owners"][faces]
    distance = np.sum(solver.geo_data["cell_connection_vector"][faces] * normal, axis=1)
    expected = velocity[owners] + (un - np.sum(velocity[owners] * normal, axis=1))[:, None] * normal
    expected += distance[:, None] * gt
    np.testing.assert_allclose(solver.velocity[ghosts], expected, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(solver.velocity_old, solver.velocity)
    np.testing.assert_array_equal(solver.velocity_older, solver.velocity)
    np.testing.assert_allclose(
        solver.volumetric_face_flux[faces],
        un * solver.geo_data["face_area"][faces],
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_array_equal(solver.volumetric_face_flux_old, solver.volumetric_face_flux)
    np.testing.assert_array_equal(solver.volumetric_face_flux_older, solver.volumetric_face_flux)


def test_mixed_trace_setter_invalidates_cached_gradient_and_publishes_state(solver):
    _, un, gt = data(solver)
    previous = solver.get_velocity_gradient_field().copy()
    revision = solver._state_revision
    solver.set_normal_velocity_tangential_gradient_boundary_condition(un, gt, "inlet")
    actual = solver.get_velocity_gradient_field()
    expected = _resolve_gradient_fn(solver.geo_data)(
        solver.velocity, solver.mesh_data, solver.geo_data
    )[: solver.mesh_data["n_cells"]]
    assert np.linalg.norm(expected - previous) > 0.1
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    assert solver._state_revision == revision + 1
