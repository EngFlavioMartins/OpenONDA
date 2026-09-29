"""Pressure boundary history survives disk and in-memory state restoration."""

import numpy as np

from source.solvers.fvm import (
    BoundaryConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    TimeConfig,
    TransportConfig,
)
from source.solvers.fvm.io.backup import capture_restart_payload, publish_restart_payload
from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh


def make_solver(directory):
    velocity = [1.0, 0.2, 0]
    return FVMSolver(
        FVMSetup(
            case_name="fixed_flux_restart",
            linear=LinearSolverConfig(linear_solver="spsolve"),
            time=TimeConfig(time_step_size=0.01, end_time=0.1),
            transport=TransportConfig(kinematic_viscosity=0.01),
            initial_velocity=velocity,
            boundaries=[
                BoundaryConfig.wall("body"),
                BoundaryConfig(
                    name="numericalBoundary",
                    velocity_type="fixedValue",
                    velocity_value=velocity,
                    pressure_type="fixedFluxPressure",
                ),
            ],
        ),
        case_dir=directory,
        mesh_data=coupling_box_mesh(
            (-1.0, 1.0) * 3, 0.5, hole_box=(-0.5, 0.5) * 3, wall_patch_name="body"
        ),
    )


def test_fixed_flux_increment_is_reconstructed_from_checkpoint_ghosts(tmp_path):
    with make_solver(tmp_path / "first") as first:
        first.advance()
        first.advance()
        snapshot = capture_restart_payload(first)
        path = first.save_state(tmp_path / "accepted.npz")
        patch = next(b for b in first.boundaries if b["name"] == "numericalBoundary")
        increment = patch["fixed_flux_pressure_delta"].copy()
        assert np.linalg.norm(increment) > 1e-5
        first.advance()
        expected = capture_restart_payload(first)
        publish_restart_payload(first, snapshot)
        np.testing.assert_allclose(patch["fixed_flux_pressure_delta"], increment, atol=1e-14)
        first.advance()
        for field, values in expected.fields.items():
            np.testing.assert_allclose(getattr(first, field), values, atol=1e-13)
    with make_solver(tmp_path / "resumed") as resumed:
        resumed.load_state(path)
        patch = next(b for b in resumed.boundaries if b["name"] == "numericalBoundary")
        np.testing.assert_allclose(patch["fixed_flux_pressure_delta"], increment, atol=1e-14)
        resumed.advance()
        for field, values in expected.fields.items():
            np.testing.assert_allclose(getattr(resumed, field), values, atol=1e-13)
