"""Pressure-free boundary drives must use the face location and one predictor."""

import numpy as np
import pytest

from openonda import fvm
from source.solvers.fvm.core.solver import FVMSolver
from source.solvers.fvm.fields.gradients import _resolve_gradient_fn
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.solve import simple_solver
from tests.support.fvm_mesh import structured_box

PATCHES = ("xmin", "xmax", "ymin", "ymax")


def _setup(dt=0.02, outer=2, nonorth=0):
    return fvm.FVMSetup(
        case_name="fixed_flux_face_drive",
        logging=fvm.LoggingConfig(console=False),
        backup=fvm.BackupConfig(schedule=None, write_at_end=False),
        time=fvm.TimeConfig(
            time_step_size=dt,
            end_time=0.5,
            output_schedule=fvm.RunSchedule(every_n_steps=10000),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="limitedLinear", gradient_scheme="lsq", time_scheme="backward"
        ),
        linear=fvm.LinearSolverConfig(
            linear_solver="spsolve",
            pressure_solver="spsolve",
            pressure_tolerance=1e-11,
            momentum_tolerance=1e-11,
        ),
        pimple=fvm.PimpleControl(
            n_outer_correctors=outer,
            n_correctors=2,
            n_nonorthogonal_correctors=nonorth,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        transport=fvm.TransportConfig(density=1, kinematic_viscosity=1 / 150),
        boundaries=[
            fvm.BoundaryConfig(
                name=name,
                velocity_type="fixedValue",
                velocity_value=[0, 0, 0],
                pressure_type="fixedFluxPressure",
            )
            for name in PATCHES
        ]
        + [fvm.BoundaryConfig.slip("zmin"), fvm.BoundaryConfig.slip("zmax")],
        initial_velocity=[0, 0, 0],
    )


def _field(name, points, time, dt, first):
    r = points - [0.4, 0.4, 0.02]
    if name == "accelerating_uniform":
        acceleration = 2 * (time - dt / 2) if first else 2 * time
        velocity = np.broadcast_to([0.3 + time**2, 0, 0], points.shape).copy()
        gradient = np.broadcast_to([-acceleration, 0, 0], points.shape).copy()
        return velocity, np.zeros((3, 3)), -acceleration * r[:, 0], gradient
    jacobian = np.diag([1, -1, 0])
    velocity = r @ jacobian.T
    gradient = -r @ (jacobian @ jacobian).T
    pressure = 0.5 * np.einsum("ij,ij->i", r, gradient)
    return velocity, jacobian, pressure, gradient


def _solve(directory, name, dt=0.02, outer=2, nonorth=0):
    with FVMSolver(
        _setup(dt, outer, nonorth), directory, mesh_data=structured_box(12, 12, 1, 0.8, 0.8, 0.04)
    ) as solver:
        solver.auto_write = False
        count = solver.mesh_data["n_cells"]
        centres = solver.geo_data["cell_centre"][:count]

        def impose(time, first, seed=False):
            for patch in solver.boundaries:
                if patch["name"] not in PATCHES:
                    continue
                faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
                points = solver.geo_data["face_centre"][faces]
                normal = (
                    solver.geo_data["face_area_vector"][faces]
                    / solver.geo_data["face_area"][faces, None]
                )
                velocity, jacobian, pressure, _ = _field(name, points, time, dt, first)
                derivative = normal @ jacobian.T
                tangent = derivative - np.einsum("ij,ij->i", derivative, normal)[:, None] * normal
                solver.set_normal_velocity_tangential_gradient_boundary_condition(
                    np.einsum("ij,ij->i", velocity, normal), tangent, patch["name"]
                )
                if seed:
                    owner = solver.mesh_data["owners"][faces]
                    patch["fixed_flux_pressure_delta"] = (
                        pressure - _field(name, centres[owner], time, dt, first)[2]
                    )
                else:
                    solver.set_flux_consistent_pressure_boundary_condition(patch["name"])

        impose(0, True, True)
        initial = _field(name, centres, 0, dt, True)
        solver.set_initial_state(initial[0], initial[2])
        steps = round(0.2 / dt)
        for step in range(steps):
            time = (step + 1) * dt
            impose(time, step == 0)
            solver.advance()
            assert solver.last_diagnostics.max_continuity_error < 1e-11
        expected = _field(name, centres, time, dt, False)
        velocity_error = np.sqrt(
            np.mean(np.sum((solver.velocity[:count] - expected[0]) ** 2, axis=1))
        )
        gradient = _resolve_gradient_fn(solver.geo_data)(
            solver.kinematic_pressure, solver.mesh_data, solver.geo_data
        )[:count, :, 0]
        gradient_error = np.sqrt(np.mean(np.sum((gradient - expected[3]) ** 2, axis=1)))
        return velocity_error, gradient_error


@pytest.mark.parametrize(
    ("dt", "velocity_limit", "gradient_limit"), [(0.02, 5e-5, 3e-4), (0.008, 2e-5, 1e-4)]
)
def test_stationary_strain_does_not_create_pressure_from_owner_face_velocity_offset(
    tmp_path, dt, velocity_limit, gradient_limit
):
    velocity_error, gradient_error = _solve(tmp_path, "strain", dt)
    # The former owner-sampled pressure boundary gives U errors 0.018/0.019
    # here, worsening as dt decreases. No exact pressure trace is prescribed.
    assert velocity_error < velocity_limit
    assert gradient_error < gradient_limit


@pytest.mark.parametrize(
    ("outer", "velocity_limit", "gradient_limit"), [(2, 6e-4, 0.04), (4, 7e-5, 0.008)]
)
def test_unsteady_native_pressure_converges_without_ghost_dependent_drive(
    tmp_path, outer, velocity_limit, gradient_limit
):
    velocity_error, gradient_error = _solve(tmp_path, "accelerating_uniform", outer=outer)
    # Both velocity AND pressure gradient improve over the former implementation;
    # a cell-only extrapolation without shared predictor rank assignment regressed p.
    assert velocity_error < velocity_limit
    assert gradient_error < gradient_limit


def test_frozen_predictor_and_face_drive_survive_all_piso_and_nonorthogonal_sweeps(
    tmp_path, monkeypatch
):
    assembly = simple_solver.assemble_pressure_correction_equation_rhie_chow
    calls = []

    def recorded(*args, **kwargs):
        result = assembly(*args, **kwargs)
        workspace = result[3]
        calls.append((workspace, kwargs["reuse_matrix"]))
        return result

    monkeypatch.setattr(simple_solver, "assemble_pressure_correction_equation_rhie_chow", recorded)
    _solve(tmp_path, "accelerating_uniform", nonorth=2)
    # Two PISO correctors each have three nonorthogonal sweeps, using one
    # momentum predictor/pressure-free face drive and one cached matrix.
    for start in range(0, len(calls), 6):
        group = calls[start : start + 6]
        assert len(group) == 6
        assert [reuse for _, reuse in group] == [False, True, True, True, True, True]
        for workspace, _ in group[1:]:
            assert workspace.velocity_h_over_a is group[0][0].velocity_h_over_a
            assert workspace.pressure_free_boundary_flux is group[0][0].pressure_free_boundary_flux
            assert workspace.matrix is group[0][0].matrix


@pytest.mark.parametrize("coefficient", [np.array([0.2, 0.4, 0.7]), 0.3])
def test_linear_pressure_on_oblique_face_includes_diagonal_cross_flux_and_skew_delta(
    monkeypatch, coefficient
):
    normal = np.array([0.6, 0.8, 0])
    sf = 2.5 * normal
    dr = 0.2 * normal + np.array([0.08, -0.06, 0.03])
    gradient = np.array([0.9, -0.4, 0.7])
    D = np.asarray(coefficient)
    D = np.array([D]) if D.ndim == 0 else D[None]
    target = np.array([0.7, -0.1, 0.2])
    B = target + np.asarray(coefficient) * gradient
    patch = {"start_face": 0, "n_faces": 1, "pressure_type": "fixedFluxPressure"}
    mesh = {"n_cells": 1, "n_faces": 1, "n_interior_faces": 0, "owners": np.array([0])}
    geometry = {"face_area_vector": sf[None], "cell_connection_vector": dr[None]}
    monkeypatch.setattr(
        simple_solver.gradients,
        "_resolve_gradient_fn",
        lambda _: lambda *args: np.tile(gradient, (2, 1)),
    )
    pressure = np.zeros(2)
    simple_solver._update_fixed_flux_pressure_boundaries(
        pressure,
        np.array([target, target]),
        D,
        mesh,
        geometry,
        [patch],
        kinematic_pressure_gradient=np.tile(gradient, (2, 1)),
        pressure_free_face_flux=np.array([B @ sf]),
    )
    assert pressure[1] == pytest.approx(gradient @ dr, abs=1e-14)


def test_nonpositive_normal_momentum_inverse_is_rejected(monkeypatch):
    monkeypatch.setattr(simple_solver.gradients, "_resolve_gradient_fn", lambda _: None)
    with pytest.raises(ValueError, match="finite positive"):
        simple_solver._update_fixed_flux_pressure_boundaries(
            np.zeros(2),
            np.zeros((2, 3)),
            np.zeros(1),
            {"n_cells": 1, "n_faces": 1, "n_interior_faces": 0, "owners": np.array([0])},
            {
                "face_area_vector": np.array([[1, 0, 0]]),
                "cell_connection_vector": np.array([[0.5, 0, 0]]),
            },
            [{"start_face": 0, "n_faces": 1, "pressure_type": "fixedFluxPressure"}],
            kinematic_pressure_gradient=np.zeros((2, 3)),
            pressure_free_face_flux=np.zeros(1),
        )


def test_thin_layer_fallback_closes_only_unresolved_direction_with_lagged_faces():
    mesh = structured_box(4, 4, 1)
    geometry = compute_mesh_geometry(mesh)
    faces = np.arange(mesh["n_interior_faces"], mesh["n_faces"])
    for patch in mesh["boundary"]:
        patch["pressure_type"] = "fixedFluxPressure"
    matrix = np.array([[0.2, -0.7, 1.3], [0.6, -0.1, -0.4], [-0.8, 1.2, -0.1]])
    real = geometry["cell_centre"] @ matrix.T + [0.3, -0.2, 0.1]
    face = geometry["face_centre"][faces] @ matrix.T + [0.3, -0.2, 0.1]
    native = np.concatenate((real, face))
    arguments = {
        "velocity_star": native,
        "pressure_velocity_coefficient": np.ones(mesh["n_cells"]),
        "predictor_pressure_gradient": np.zeros_like(native),
    }
    flux = simple_solver._fixed_flux_pressure_face_drive(
        real, mesh, geometry, mesh["boundary"], **arguments
    )
    expected = np.einsum("fi,fi->f", face, geometry["face_area_vector"][faces])
    np.testing.assert_allclose(flux[faces], expected, atol=2e-14)

    # All real neighbours are coplanar. Poisoning physical face values changes
    # the explicitly unresolved z closure but cannot change the identified
    # x/y extrapolation. No blanket ghost-inclusive fallback is permitted.
    dr = geometry["cell_connection_vector"][faces]
    in_plane = faces[np.abs(dr[:, 2]) < 1e-12]
    span = faces[np.abs(dr[:, 2]) > 1e-12]
    native[mesh["n_cells"] :] *= -7
    changed = simple_solver._fixed_flux_pressure_face_drive(
        real, mesh, geometry, mesh["boundary"], **arguments
    )
    np.testing.assert_array_equal(changed[in_plane], flux[in_plane])
    expected_change = (
        -8
        * matrix[2, 2]
        * geometry["cell_connection_vector"][span, 2]
        * geometry["face_area_vector"][span, 2]
    )
    np.testing.assert_allclose(changed[span] - flux[span], expected_change, atol=2e-14)
