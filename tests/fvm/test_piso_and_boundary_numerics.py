"""PISO correctors, slip-wall stress, cyclic linear-upwind and solver settings."""

import contextlib
import io

import numpy as np
import pytest

from openonda import fvm
from openonda.fvm.assemble import convection, diffusion, momentum
from openonda.fvm.fields.gradients import compute_gauss_gradient
from openonda.fvm.mesh.geometry import compute_mesh_geometry
from openonda.fvm.mesh.rectilinear import box_mesh_3d
import openonda.fvm.solve.pimple_solver as pimple_solver
from openonda.fvm.turbulence.smagorinsky import Smagorinsky, _compute_filter_width
from tests.support.fvm_mesh import structured_box

KINEMATIC_VISCOSITY = 0.05
CELLS = 12


def _periodic_square():
    mesh = box_mesh_3d(
        np.linspace(0.0, 2 * np.pi, CELLS + 1),
        np.linspace(0.0, 2 * np.pi, CELLS + 1),
        np.array([0.0, 2 * np.pi / CELLS]),
    )
    for patch, name in zip(
        mesh["boundary"], ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"), strict=True
    ):
        patch["name"] = name
    return mesh


def _cyclic(first, second):
    return [fvm.BoundaryConfig.cyclic(first, second), fvm.BoundaryConfig.cyclic(second, first)]


def _taylor_green(centres, time):
    x, y = centres[:, 0], centres[:, 1]
    decay = np.exp(-2 * KINEMATIC_VISCOSITY * time)
    velocity = np.column_stack(
        (np.sin(x) * np.cos(y) * decay, -np.cos(x) * np.sin(y) * decay, np.zeros_like(x))
    )
    return velocity, 0.25 * (np.cos(2 * x) + np.cos(2 * y)) * decay**2


def _taylor_green_velocity(tmp_path, n_correctors, n_outer_correctors=1):
    setup = fvm.FVMSetup(
        case_name="taylor_green",
        logging=fvm.LoggingConfig(console=False),
        backup=fvm.BackupConfig(schedule=None, write_at_end=False),
        time=fvm.TimeConfig(
            time_step_size=0.2, end_time=0.6, output_schedule=fvm.RunSchedule(every_n_steps=1000)
        ),
        schemes=fvm.DiscretizationConfig(convection_scheme="linear", time_scheme="backward"),
        linear=fvm.LinearSolverConfig(
            linear_solver="spsolve",
            pressure_solver="spsolve",
            pressure_tolerance=1e-12,
            momentum_tolerance=1e-12,
        ),
        pimple=fvm.PimpleControl(n_correctors=n_correctors, n_outer_correctors=n_outer_correctors),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=KINEMATIC_VISCOSITY),
        boundaries=_cyclic("xmin", "xmax") + _cyclic("ymin", "ymax") + _cyclic("zmin", "zmax"),
        initial_velocity=[0.0, 0.0, 0.0],
    )
    case = tmp_path / f"piso_{n_correctors}_{n_outer_correctors}"
    with (
        contextlib.redirect_stdout(io.StringIO()),
        fvm.create_fvm_solver(setup, case_dir=case, mesh=_periodic_square) as solver,
    ):
        solver.auto_write = False
        n = solver.mesh_data["n_cells"]
        solver.set_initial_state(*_taylor_green(solver.geo_data["cell_centre"][:n], 0.0))
        while solver.time < 0.6 - 1e-12:
            solver.advance()
        return np.array(solver.velocity[:n])


def test_second_piso_corrector_approaches_the_implicit_step(tmp_path):
    # Outer correctors converge each step to the implicit solution; the PISO
    # corrector must remove most of the single-corrector splitting error.
    implicit = _taylor_green_velocity(tmp_path, 1, n_outer_correctors=12)
    one = np.abs(_taylor_green_velocity(tmp_path, 1) - implicit).max()
    two = np.abs(_taylor_green_velocity(tmp_path, 2) - implicit).max()
    assert two < 0.3 * one


def test_piso_increment_is_the_off_diagonal_momentum_change():
    matrix = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -2.0], [0.0, -1.0, 3.0]])
    from scipy.sparse import csr_matrix

    operator = {name: csr_matrix(matrix) for name in "xyz"}
    change = np.arange(9.0).reshape(3, 3)
    increment = pimple_solver._piso_h_over_a_increment(operator, change, 3, None)
    off_diagonal = matrix - np.diag(np.diag(matrix))
    np.testing.assert_allclose(increment, -(off_diagonal @ change) / np.diag(matrix)[:, None])


def _slip_box():
    mesh = structured_box(3, 3, 2)
    mesh["boundary"][3]["velocity_type"] = "slip"  # ymax
    for patch in mesh["boundary"]:
        patch.setdefault("velocity_type", "zeroGradient")
    return mesh, compute_mesh_geometry(mesh, compute_lsq=False)


def test_slip_plane_diffuses_only_the_wall_normal_velocity():
    mesh, geometry = _slip_box()
    n_cells = mesh["n_cells"]
    n_total = n_cells + mesh["n_faces"] - mesh["n_interior_faces"]
    velocity = np.tile([0.3, 0.7, -0.2], (n_total, 1))
    patch = mesh["boundary"][3]
    faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
    owners = mesh["owners"][faces]
    area = np.linalg.norm(geometry["face_area_vector"][faces], axis=1)
    coefficient = KINEMATIC_VISCOSITY * area / geometry["wall_distance"][faces]
    for component in range(3):
        flux = diffusion.assemble_diffusion_term(
            velocity[:, component],
            np.zeros((n_total, 3)),
            KINEMATIC_VISCOSITY,
            mesh,
            geometry,
            mesh["boundary"],
            vector_field=velocity,
            component=component,
        )
        # Face flux of component i: nu A n_i (u_P . n) / d, positive out of
        # the owner; it vanishes for the tangential components.
        expected = coefficient * (component == 1) * velocity[owners, 1]
        np.testing.assert_allclose(flux["flux_tf"][faces], expected, atol=1e-14)


def test_slip_plane_keeps_tangential_components_on_one_matrix():
    mesh, geometry = _slip_box()
    mesh_boundaries = mesh["boundary"]
    n_total = mesh["n_cells"] + mesh["n_faces"] - mesh["n_interior_faces"]
    equations = momentum.assemble_momentum_equation(
        np.full((n_total, 3), 0.1),
        np.zeros(n_total),
        np.zeros(mesh["n_faces"]),
        1.0,
        KINEMATIC_VISCOSITY,
        mesh,
        geometry,
        mesh_boundaries,
        time_step_size=0.1,
        velocity_old=np.full((n_total, 3), 0.1),
    )
    assert equations["x"]["A"] is equations["z"]["A"]
    assert equations["y"]["A"] is not equations["x"]["A"]
    extra = equations["y"]["A"].diagonal() - equations["x"]["A"].diagonal()
    patch = mesh_boundaries[3]
    owners = mesh["owners"][patch["start_face"] : patch["start_face"] + patch["n_faces"]]
    assert np.all(extra[owners] > 0.0)
    assert np.count_nonzero(extra) == len(np.unique(owners))


def test_cyclic_linear_upwind_matches_an_interior_face(tmp_path):
    setup = fvm.FVMSetup(
        case_name="cyclic_lust",
        logging=fvm.LoggingConfig(console=False),
        backup=fvm.BackupConfig(schedule=None, write_at_end=False),
        time=fvm.TimeConfig(
            time_step_size=0.1, end_time=0.1, output_schedule=fvm.RunSchedule(every_n_steps=10)
        ),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=KINEMATIC_VISCOSITY),
        boundaries=_cyclic("xmin", "xmax") + _cyclic("ymin", "ymax") + _cyclic("zmin", "zmax"),
        initial_velocity=[1.0, 0.0, 0.0],
    )
    with (
        contextlib.redirect_stdout(io.StringIO()),
        fvm.create_fvm_solver(setup, case_dir=tmp_path, mesh=_periodic_square) as solver,
    ):
        mesh, geometry = solver.mesh_data, solver.geo_data
        n_cells, n_interior = mesh["n_cells"], mesh["n_interior_faces"]
        spacing = 2 * np.pi / CELLS
        flux = np.zeros(mesh["n_faces"])
        flux[: mesh["n_faces"]] = geometry["face_area_vector"][:, 0]
        xmin = next(patch for patch in solver.boundaries if patch["name"] == "xmin")
        cyclic_face = xmin["start_face"]
        y_face = geometry["face_centre"][cyclic_face, 1]
        interior = np.flatnonzero(
            (np.abs(geometry["face_area_vector"][:n_interior, 0]) > 0)
            & np.isclose(geometry["face_centre"][:n_interior, 1], y_face)
        )
        shift = geometry["face_centre"][interior[0], 0]

        def field(offset):
            values = np.zeros(n_cells + mesh["n_faces"] - n_interior)
            values[:n_cells] = np.sin(geometry["cell_centre"][:n_cells, 0] + offset)
            return values

        for scheme in ("LUST", "linearUpwind"):
            results = []
            for offset, face in ((0.0, interior[0]), (shift, cyclic_face)):
                values = field(offset)
                gradient = compute_gauss_gradient(values, mesh, geometry)[..., 0]
                result = convection.assemble_convection_term(
                    values,
                    flux,
                    mesh,
                    geometry,
                    solver.boundaries,
                    scheme=scheme,
                    scalar_field_gradient=gradient,
                )
                results.append(result["flux_vf"][face] * np.sign(flux[face]))
            assert results[1] == pytest.approx(results[0], rel=1e-10, abs=1e-14 * spacing)


def test_steady_pressure_solve_uses_the_configured_amg_settings(monkeypatch):
    from openonda.fvm.solve import linear_interface

    calls = []

    def recorded(A, b, amg_tolerance, amg_max_iterations, *args):
        calls.append((amg_tolerance, amg_max_iterations, args[-1]))
        return np.zeros(len(b)), {
            "preconditioner": "pyamg",
            "iterations": 0,
            "reason": "",
            "setup_seconds": 0.0,
            "solve_seconds": 0.0,
            "used_fallback": False,
        }

    monkeypatch.setattr(linear_interface, "_solve_pressure", recorded)
    from scipy.sparse import identity

    linear_interface.solve_linear_system(
        identity(4, format="csr"),
        np.zeros(4),
        method="amg",
        equation_type="kinematic_pressure",
        tol=1e-9,
        maxiter=50,
    )
    # Without explicit AMG settings, the pressure tolerance and cap apply.
    assert calls[-1][:2] == (1e-9, 50)


def test_smagorinsky_reports_the_filter_width_it_uses():
    mesh = structured_box(4, 4, 1, lz=0.1)
    for patch in mesh["boundary"][4:]:
        patch["velocity_type"] = "empty"
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    model = Smagorinsky(mesh, geometry)
    width = _compute_filter_width(geometry["cell_volume"], mesh, geometry)
    info = model.get_filter_info()
    assert info["mean_filter_width"] == pytest.approx(float(np.mean(width)))
    assert info["mean_filter_width"] == pytest.approx(0.25)


@pytest.mark.parametrize("n_correctors", [1, 2])
def test_uniform_freestream_is_preserved_by_every_corrector(tmp_path, n_correctors):
    from openonda.fvm.mesher import coupling_box_mesh

    # The first pressure corrector must see the prescribed inflow ghosts of a
    # freestream patch; a zero predictor ghost turns one step of uniform flow
    # into an O(1) pressure correction.
    velocity = [1.0, 0.0, 0.0]
    setup = fvm.FVMSetup(
        case_name="uniform_freestream",
        boundaries=[fvm.BoundaryConfig.freestream("numericalBoundary", velocity)],
        initial_velocity=velocity,
        pimple=fvm.PimpleControl(n_correctors=n_correctors),
        logging=fvm.LoggingConfig(console=False),
        backup=fvm.BackupConfig(schedule=None, write_at_end=False),
        time=fvm.TimeConfig(
            time_step_size=0.01, end_time=0.01, output_schedule=fvm.RunSchedule(every_n_steps=10)
        ),
    )
    with (
        contextlib.redirect_stdout(io.StringIO()),
        fvm.create_fvm_solver(
            setup,
            case_dir=tmp_path,
            mesh=lambda: coupling_box_mesh((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), 0.25),
        ) as solver,
    ):
        solver.auto_write = False
        solver.advance()
        n = solver.mesh_data["n_cells"]
        np.testing.assert_allclose(solver.velocity[:n], np.tile(velocity, (n, 1)), atol=1e-12)
