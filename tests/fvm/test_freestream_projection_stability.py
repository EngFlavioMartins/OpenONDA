"""Mixed far-field branches remain consistent through transient projection."""

import numpy as np
import pytest

import openonda.fvm as fvm
from source.solvers.fvm.core.solver import FVMSolver
from source.solvers.fvm.io.backup import capture_restart_state, restore_restart_state
from source.solvers.fvm.solve import simple_solver
from tests._tutorial_helpers import load_tutorial_module

rectilinear_box_2d = load_tutorial_module(
    "fvm/cylinder_ibm", "assets.mesh_rectilinear"
).rectilinear_box_2d


def make_solver(directory, immersed):
    h = 0.125
    mesh = rectilinear_box_2d(np.arange(-2, 4.0625, h), np.arange(-2, 2.0625, h), h)
    setup = fvm.FVMSetup(
        case_name="freestream_projection",
        time=fvm.TimeConfig(time_step_size=0.01171875, end_time=1),
        linear=fvm.LinearSolverConfig(linear_solver="spsolve"),
        schemes=fvm.DiscretizationConfig(gradient_scheme="gauss"),
        transport=fvm.TransportConfig(density=1, kinematic_viscosity=1 / 30),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", [1, 0, 0]),
            fvm.BoundaryConfig.outlet("outlet"),
            fvm.BoundaryConfig.freestream("bottom", [1, 0, 0]),
            fvm.BoundaryConfig.freestream("top", [1, 0, 0]),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ],
        initial_velocity=[1, 0, 0],
    )
    solver = FVMSolver(setup, str(directory), mesh_data=mesh)
    solver.auto_write = False
    if immersed:
        solver.set_immersed_bodies(
            fvm.ImmersedBody.cylinder_z(
                centre=[0, 0, h / 2],
                diameter=1,
                grid_spacing=h,
                marker_spacing_ratio=1,
                name="cylinder",
            ),
            grid_spacing=h,
        )
    return solver


@pytest.mark.parametrize("immersed", [False, True])
def test_multiple_pressure_correctors_preserve_uniform_flow_and_bounded_ibm_loading(
    tmp_path, immersed
):
    with make_solver(tmp_path, immersed) as solver:
        n = solver.mesh_data["n_cells"]
        volume = solver.geo_data["cell_volume"][:n]
        initial_energy = 0.5 * np.sum(volume)
        for step in range(40):
            if immersed and step == 20:
                checkpoint = capture_restart_state(solver)
                solver.advance()
                expected_velocity = solver.velocity.copy()
                expected_flux = solver.volumetric_face_flux.copy()
                expected_time = solver.time
                restore_restart_state(solver, checkpoint)
                for boundary in solver.boundaries:
                    if boundary.get("velocity_type") == "freestream":
                        boundary["_freestream_outflow"] = ~boundary["_freestream_outflow"]
                solver.advance()
                np.testing.assert_allclose(solver.velocity, expected_velocity, rtol=0, atol=1e-11)
                np.testing.assert_allclose(
                    solver.volumetric_face_flux, expected_flux, rtol=0, atol=1e-13
                )
                assert solver.time == expected_time
                continue
            solver.advance()
        velocity = solver.velocity[:n]
        energy = 0.5 * np.sum(volume * np.sum(velocity**2, axis=1))
        assert solver.time > 0.45
        assert np.max(np.linalg.norm(velocity, axis=1)) < 2
        assert energy < 1.1 * initial_energy
        if immersed:
            assert solver.ibm.slip_error(solver.velocity) < 0.01
        else:
            np.testing.assert_allclose(velocity, np.tile([1, 0, 0], (n, 1)), atol=1e-9, rtol=0)
        phi = solver.volumetric_face_flux
        owners = solver.mesh_data["owners"]
        neighbours = solver.mesh_data["neighbours"]
        ni = solver.mesh_data["n_interior_faces"]
        divergence = np.bincount(owners, weights=phi, minlength=n)
        divergence -= np.bincount(neighbours[:ni], weights=phi[:ni], minlength=n)
        assert np.max(np.abs(divergence / volume)) < 1e-10


@pytest.mark.parametrize("mask_key", ["_freestream_outflow", "_fixed_freestream_outflow"])
def test_frozen_pressure_branches_override_opposite_ghost_flow(mask_key):
    boundary = {
        "name": "farfield",
        "start_face": 0,
        "n_faces": 2,
        "pressure_type": "freestream",
        "velocity_type": "freestream",
        "kinematic_pressure_value": 0,
    }
    boundary[mask_key] = np.array([True, False])
    mesh = {"n_cells": 1, "n_interior_faces": 0, "n_faces": 2}
    geometry = {"face_area_vector": np.array([[1, 0, 0], [1, 0, 0]])}
    velocity = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
    codes, _, _ = simple_solver._build_boundary_face_arrays([boundary], 0, 2)
    np.testing.assert_array_equal(codes, [1, 0])
    assert not simple_solver._pressure_requires_constraint([boundary], velocity, mesh, geometry)
    boundary[mask_key][:] = False
    assert simple_solver._pressure_requires_constraint([boundary], velocity, mesh, geometry)
