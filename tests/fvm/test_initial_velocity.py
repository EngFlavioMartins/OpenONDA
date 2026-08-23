import numpy as np
import pytest

from source.solvers.fvm import BoundaryConfig, FVMSetup, FVMSolver, TransportConfig


def _config() -> FVMSetup:
    return FVMSetup(
        case_name="initial_velocity",
        transport=TransportConfig(kinematic_viscosity=0.01),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax"),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
    )


def test_set_initial_velocity_rebuilds_history_boundaries_and_flux(hand_built_3d_mesh, tmp_path):
    solver = FVMSolver(_config(), case_dir=tmp_path, mesh_data=hand_built_3d_mesh)
    values = np.tile([0.25, -0.5, 0.75], (hand_built_3d_mesh["n_cells"], 1))

    solver.set_initial_velocity(values)

    np.testing.assert_allclose(solver.velocity[0], values[0])
    np.testing.assert_allclose(solver.velocity_old, solver.velocity)
    np.testing.assert_allclose(solver.velocity_older, solver.velocity)
    assert np.any(np.abs(solver.volumetric_face_flux) > 0.0)


def test_set_initial_velocity_rejects_invalid_or_late_values(hand_built_3d_mesh, tmp_path):
    solver = FVMSolver(_config(), case_dir=tmp_path, mesh_data=hand_built_3d_mesh)

    with pytest.raises(ValueError, match="shape"):
        solver.set_initial_velocity(np.zeros((2, 3)))

    solver.step = 1
    with pytest.raises(RuntimeError, match="before the first time step"):
        solver.set_initial_velocity(np.zeros((hand_built_3d_mesh["n_cells"], 3)))
