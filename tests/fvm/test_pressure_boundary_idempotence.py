"""Repeated coupling updates preserve the native pressure-gradient history."""

from types import SimpleNamespace

import numpy as np
import pytest

from openonda import fvm
from source.coupler.boundary import tangential_normal_velocity_gradient
from source.solvers.fvm.core.solver import FVMSolver
from source.solvers.fvm.coupling.coupler_interface import CouplerInterfaceMixin
from tests.support.fvm_mesh import structured_box


def _rotation_velocity(points):
    return np.column_stack((0.4 - points[:, 1], points[:, 0] - 0.4, np.zeros(len(points))))


def _rotation_pressure(points):
    return 0.5 * ((points[:, 0] - 0.4) ** 2 + (points[:, 1] - 0.4) ** 2)


def _run_rotation(directory, *, reapply):
    """An exact steady incompressible flow with a nonzero pressure gradient."""
    patches = ("xmin", "xmax", "ymin", "ymax")
    setup = fvm.FVMSetup(
        case_name="rotation_pressure_history",
        logging=fvm.LoggingConfig(console=False),
        backup=fvm.BackupConfig(schedule=None, write_at_end=False),
        time=fvm.TimeConfig(
            time_step_size=0.008,
            end_time=0.2,
            output_schedule=fvm.RunSchedule(every_n_steps=1000),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="limitedLinear", gradient_scheme="lsq", time_scheme="backward"
        ),
        linear=fvm.LinearSolverConfig(
            linear_solver="spsolve",
            pressure_solver="spsolve",
            pressure_tolerance=1e-10,
            momentum_tolerance=1e-10,
        ),
        pimple=fvm.PimpleControl(
            n_correctors=2,
            n_outer_correctors=2,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=1 / 150),
        boundaries=[
            fvm.BoundaryConfig(
                name=name,
                velocity_type="fixedValue",
                velocity_value=[0, 0, 0],
                pressure_type="fixedFluxPressure",
            )
            for name in patches
        ]
        + [fvm.BoundaryConfig.slip("zmin"), fvm.BoundaryConfig.slip("zmax")],
        initial_velocity=[0, 0, 0],
    )
    with FVMSolver(setup, directory, mesh_data=structured_box(20, 20, 1, 0.8, 0.8, 0.04)) as solver:
        solver.auto_write = False
        count = solver.mesh_data["n_cells"]
        centre = solver.geo_data["cell_centre"][:count]
        jacobian = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 0]])
        for boundary in solver.boundaries:
            if boundary["name"] not in patches:
                continue
            faces = np.arange(boundary["start_face"], boundary["start_face"] + boundary["n_faces"])
            points = solver.geo_data["face_centre"][faces]
            areas = solver.geo_data["face_area_vector"][faces]
            normals = areas / np.linalg.norm(areas, axis=1)[:, None]
            solver.set_normal_velocity_tangential_gradient_boundary_condition(
                np.einsum("ij,ij->i", _rotation_velocity(points), normals),
                tangential_normal_velocity_gradient(
                    np.broadcast_to(jacobian, (len(faces), 3, 3)), normals
                ),
                boundary["name"],
            )
            owners = solver.mesh_data["owners"][faces]
            boundary["fixed_flux_pressure_delta"] = _rotation_pressure(points) - _rotation_pressure(
                centre[owners]
            )
        exact_velocity, exact_pressure = _rotation_velocity(centre), _rotation_pressure(centre)
        solver.set_initial_state(exact_velocity, exact_pressure)
        for _ in range(25):
            if reapply:
                for name in patches:
                    solver.set_flux_consistent_pressure_boundary_condition(name)
            solver.advance()
        velocity = solver.velocity[:count].copy()
        pressure = solver.kinematic_pressure[:count].copy()
        pressure_error = pressure - exact_pressure
        pressure_error -= pressure_error.mean()
        assert np.sqrt(np.mean(pressure_error**2)) < 5e-5
        assert np.sqrt(np.mean(np.sum((velocity - exact_velocity) ** 2, axis=1))) < 6e-5
        assert solver.last_diagnostics.max_continuity_error < 1e-10
        return velocity, pressure


def test_native_pressure_reapplication_preserves_manufactured_solution(tmp_path):
    uninterrupted = _run_rotation(tmp_path / "uninterrupted", reapply=False)
    reapplied = _run_rotation(tmp_path / "reapplied", reapply=True)
    for actual, expected in zip(reapplied, uninterrupted, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize(
    "old_type,external",
    [("fixedFluxPressure", True), ("fixedGradient", False), ("zeroGradient", False)],
)
def test_switch_to_native_pressure_discards_foreign_increment(old_type, external):
    patch = {
        "pressure_type": old_type,
        "fixed_flux_pressure_external": external,
        "fixed_flux_pressure_delta": np.array([0.1]),
        "fixed_gradient_delta": np.array([0.2]),
    }
    solver = SimpleNamespace(_optional_patch=lambda name: patch)
    CouplerInterfaceMixin.set_flux_consistent_pressure_boundary_condition(solver, "outer")
    assert patch == {"pressure_type": "fixedFluxPressure"}
