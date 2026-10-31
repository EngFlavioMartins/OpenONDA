"""The direction-aware coupling boundary inside the FVM solver.

Coupling faces take the particle velocity where flow enters and the mixed
condition where it leaves, for any flow direction.
"""

import contextlib
import io

import numpy as np
import pytest

from openonda import fvm


@pytest.mark.parametrize(
    "velocity",
    [(0.8, -0.35, 0.5), (-0.6, 0.0, 0.9), (0.04, 1.0, -0.03)],
    ids=["oblique", "reversed-x", "near-tangent"],
)
def test_uniform_stream_in_any_direction_is_preserved(tmp_path, velocity):
    from openonda.fvm.mesher import coupling_box_mesh

    velocity = np.asarray(velocity)
    case = fvm.FVMCase(
        name="direction_aware",
        directory=tmp_path,
        numerics=fvm.Numerics(time_step_size=0.01),
        initial_conditions=fvm.InitialFields(velocity=tuple(velocity)),
        run=fvm.RunPlan(end_time=0.05),
        backup=fvm.Backup(write_at_end=False),
        logging=fvm.LoggingConfig(console=False),
        mesh=lambda: coupling_box_mesh((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), 0.125),
        boundaries=(
            fvm.BoundaryConfig(
                name="numericalBoundary",
                velocity_value=tuple(velocity),
                pressure_type="fixed_flux_pressure",
            ),
        ),
    )
    with contextlib.redirect_stdout(io.StringIO()), fvm.FVMSolver(case) as solver:
        solver.auto_write = False
        normals = solver.gather_boundary_face_normal("numericalBoundary")
        faces = np.tile(velocity, (len(normals), 1))
        weights = []
        for step in range(5):
            solver.set_mixed_velocity_boundary_condition(
                normals @ velocity,
                np.zeros_like(normals),
                "numericalBoundary",
                inflow_velocity=faces,
                transition_speed=0.1,
            )
            solver.set_flux_consistent_pressure_boundary_condition("numericalBoundary")
            weights = solver._patch("numericalBoundary")["inflow_weight_field"]
            solver.advance(time=0.01 * (step + 1), time_step_size=0.01)
        n = solver.mesh_data["n_cells"]
        np.testing.assert_allclose(solver.velocity[:n], np.tile(velocity, (n, 1)), atol=1e-10)
    assert np.any(weights == 1.0) and np.any(weights < 1.0)


def test_the_plain_mixed_condition_keeps_its_meaning(tmp_path):
    from openonda.fvm.mesher import coupling_box_mesh

    case = fvm.FVMCase(
        name="plain_mixed",
        directory=tmp_path,
        numerics=fvm.Numerics(time_step_size=0.01),
        initial_conditions=fvm.InitialFields(velocity=(1.0, 0.0, 0.0)),
        run=fvm.RunPlan(end_time=0.01),
        backup=fvm.Backup(write_at_end=False),
        logging=fvm.LoggingConfig(console=False),
        mesh=lambda: coupling_box_mesh((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), 0.25),
        boundaries=(
            fvm.BoundaryConfig(
                name="numericalBoundary",
                velocity_value=(1.0, 0.0, 0.0),
                pressure_type="fixed_flux_pressure",
            ),
        ),
    )
    with contextlib.redirect_stdout(io.StringIO()), fvm.FVMSolver(case) as solver:
        normals = solver.gather_boundary_face_normal("numericalBoundary")
        solver.set_mixed_velocity_boundary_condition(
            normals[:, 0],
            np.zeros_like(normals),
            "numericalBoundary",
            inflow_velocity=np.tile([1.0, 0.0, 0.0], (len(normals), 1)),
            transition_speed=0.1,
        )
        solver.set_mixed_velocity_boundary_condition(
            normals[:, 0], np.zeros_like(normals), "numericalBoundary"
        )
        patch = solver._patch("numericalBoundary")
        assert "inflow_weight_field" not in patch and "inflow_velocity_field" not in patch
        with pytest.raises(ValueError, match="together"):
            solver.set_mixed_velocity_boundary_condition(
                normals[:, 0],
                np.zeros_like(normals),
                "numericalBoundary",
                inflow_velocity=np.zeros_like(normals),
            )
