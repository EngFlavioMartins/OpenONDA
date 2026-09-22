"""Independent field and geometry checks for the production transfer path."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler.boundary import apply_fvm_boundary, tangential_normal_velocity_gradient
from source.coupler.config.types import CouplerSetup
from source.coupler.geometry import TriangulatedWall
from source.coupler.stable_renewal import (
    gaussian_represented_vortex_strength,
    recover_vortex_invariants,
    vortex_invariants,
    vortex_strength_from_velocity_trace,
)
from source.coupler.vorticity_transfer import VorticityTransfer
from source.solvers.fvm import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    LinearSolverConfig,
    PimpleControl,
    RunSchedule,
    TimeConfig,
    TransportConfig,
    create_fvm_solver,
)
from source.solvers.fvm.coupling.coupler_interface import CouplerInterfaceMixin
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d, coupling_box_mesh


def _taylor_green_fields(position, time, viscosity=0.01):
    """Return an advected, decaying Taylor--Green velocity, Jacobian and pressure.

    Positions have shape (N, 3) in metres and time is in seconds. The uniform
    flow is (1, 0, 0) m/s; the wavelength is 2 m. The Jacobian uses J[i,j] =
    du_i/dx_j, and pressure is kinematic pressure in m²/s². The field is
    invariant in z and solves the unforced incompressible Navier--Stokes
    equations for kinematic viscosity in m²/s.
    """
    x, y, _ = np.asarray(position).T
    k = np.pi
    x = k * (x - time)
    y = k * y
    decay = np.exp(-2.0 * viscosity * k**2 * time)
    velocity = np.column_stack(
        (1.0 - decay * np.cos(x) * np.sin(y), decay * np.sin(x) * np.cos(y), np.zeros(len(x)))
    )
    jacobian = np.zeros((len(x), 3, 3))
    jacobian[:, 0, 0] = k * decay * np.sin(x) * np.sin(y)
    jacobian[:, 0, 1] = -k * decay * np.cos(x) * np.cos(y)
    jacobian[:, 1, 0] = k * decay * np.cos(x) * np.cos(y)
    jacobian[:, 1, 1] = -k * decay * np.sin(x) * np.sin(y)
    pressure = -0.25 * decay**2 * (np.cos(2 * x) + np.cos(2 * y))
    return velocity, jacobian, pressure


def _taylor_green_pressure_gradient(position, time, viscosity=0.01):
    """Return the analytic kinematic-pressure gradient, shape (N, 3), in m/s²."""
    x, y, _ = np.asarray(position).T
    k = np.pi
    decay_squared = np.exp(-4.0 * viscosity * k**2 * time)
    return (
        0.5
        * k
        * decay_squared
        * np.column_stack((np.sin(2 * k * (x - time)), np.sin(2 * k * y), np.zeros(len(x))))
    )


def _solve_boundary_refinement(n, mode, output, dt=0.0025, steps=20):
    """Advance an n×n FVM mesh through 0.05 s using exact endpoint boundary data.

    Refinement holds dt fixed and measures combined space/time/boundary error.
    Direct linear solves isolate discretization error from iterative tolerance.
    The empty span has one cell of width 0.25 m. There is no body or VPM error.
    """
    axis = np.linspace(-1, 1, n + 1)
    mesh = coupling_box_mesh(
        (-1, 1, -1, 1, -0.125, 0.125),
        2 / n,
        nodes=(axis, axis, np.array([-0.125, 0.125])),
        empty_spanwise=True,
    )
    setup = FVMSetup(
        case_name=f"oracle_{mode}_{n}",
        time=TimeConfig(
            time_step_size=dt,
            end_time=steps * dt,
            output_schedule=RunSchedule(every_n_steps=100000),
        ),
        schemes=DiscretizationConfig(
            convection_scheme="linear", gradient_scheme="lsq", time_scheme="backward"
        ),
        linear=LinearSolverConfig(
            linear_solver="spsolve",
            pressure_solver="spsolve",
            pressure_tolerance=1e-12,
            momentum_tolerance=1e-12,
            pressure_relative_tolerance=0.0,
            momentum_relative_tolerance=0.0,
        ),
        pimple=PimpleControl(
            n_outer_correctors=3, n_correctors=2, velocity_relaxation=1.0, pressure_relaxation=1.0
        ),
        transport=TransportConfig(kinematic_viscosity=0.01),
        boundaries=[
            BoundaryConfig(
                name="numericalBoundary",
                velocity_type="fixedValue",
                velocity_value=[1, 0, 0],
                pressure_type="fixedFluxPressure",
            ),
            BoundaryConfig.empty("zmin"),
            BoundaryConfig.empty("zmax"),
        ],
        initial_velocity=[1, 0, 0],
    )
    with create_fvm_solver(setup, case_dir=output / f"{mode}_{n}", mesh=mesh) as solver:
        coupler = SimpleNamespace(
            fvm_solver=solver, setup=CouplerSetup(boundary_condition_mode=mode)
        )
        centre = solver.get_cell_centre_coordinates()
        face = solver.get_boundary_face_centre_coordinates("numericalBoundary")
        normal = solver.get_boundary_face_normal("numericalBoundary")
        initial, _, pressure = _taylor_green_fields(centre, 0)
        velocity, jacobian, _ = _taylor_green_fields(face, 0)
        if mode in {"vorticity_mixed", "vorticity_mixed_pressure_gradient"}:
            solver.set_normal_velocity_tangential_gradient_boundary_condition(
                np.einsum("ij,ij->i", velocity, normal),
                tangential_normal_velocity_gradient(jacobian, normal),
                "numericalBoundary",
            )
        else:
            solver.set_dirichlet_velocity_boundary_condition_vec(velocity, "numericalBoundary")
        if mode == "vorticity_mixed_pressure_gradient":
            solver.set_neumann_pressure_boundary_condition(
                _taylor_green_pressure_gradient(face, 0), "numericalBoundary"
            )
        solver.kinematic_pressure[: len(centre)] = pressure
        solver.set_initial_velocity(initial)
        for step in range(1, steps + 1):
            velocity, jacobian, _ = _taylor_green_fields(face, step * dt)
            apply_fvm_boundary(
                coupler,
                "numericalBoundary",
                velocity,
                normal_velocity=np.einsum("ij,ij->i", velocity, normal),
                tangential_gradient=tangential_normal_velocity_gradient(jacobian, normal),
                pressure_gradient=_taylor_green_pressure_gradient(face, step * dt),
            )
        assert solver.step == steps
        assert solver.time == pytest.approx(steps * dt, rel=0.0, abs=1e-14)
        expected, _, _ = _taylor_green_fields(centre, solver.time)
        error = solver.get_velocity_field() - expected
        volumes = solver.get_cell_volume()
        rms = np.sqrt(np.sum(volumes * np.sum(error**2, axis=1)) / volumes.sum())
        return {
            "n": n,
            "h": 2 / n,
            "mode": mode,
            "dt": dt,
            "steps": steps,
            "time": solver.time,
            "velocity_rms_error_over_Uinf": float(rms),
            "velocity_max_error_over_Uinf": float(np.linalg.norm(error, axis=1).max()),
        }


@pytest.mark.parametrize("strength_scale", [1.0e-18, 1.0, 1.0e18])
def test_invariant_recovery_resolves_small_and_large_circulation(strength_scale):
    rng = np.random.default_rng(20260913)
    position = rng.uniform(-1, 1, size=(120, 3))
    expected = rng.normal(size=(120, 3)) * strength_scale
    target = vortex_invariants(position, expected)
    perturbed = expected + rng.normal(size=(120, 3)) * strength_scale * 0.01
    corrected = recover_vortex_invariants(position, perturbed, target, volumes=np.ones(120))
    actual = vortex_invariants(position, corrected)
    tolerance = 32 * np.finfo(float).eps * np.linalg.norm(expected, axis=1).sum()
    assert np.linalg.norm(actual.total_vortex_strength - target.total_vortex_strength) < tolerance
    assert np.linalg.norm(actual.linear_impulse - target.linear_impulse) < tolerance


@pytest.mark.parametrize(
    "mode", ["dirichlet", "vorticity_mixed", "vorticity_mixed_pressure_gradient"]
)
def test_exact_unsteady_boundary_data_converges_under_fvm_refinement(
    tmp_path, mode, record_property
):
    errors = [
        _solve_boundary_refinement(n, mode, tmp_path)["velocity_rms_error_over_Uinf"]
        for n in (8, 16, 32)
    ]
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    for n, error in zip((8, 16, 32), errors, strict=True):
        record_property(f"n{n}_velocity_rms_error", error)
    record_property("minimum_observed_order", float(orders.min()))
    assert orders.min() > 1.7
    assert errors[-1] < 0.005


@pytest.mark.qualification
@pytest.mark.parametrize("cell_spacing", [(0.5, 0.5, 0.5), (0.125, 0.5, 1.0)])
def test_anisotropic_donors_preserve_constant_vorticity_target(cell_spacing, record_property):
    h = 0.0625
    box = np.array([-1.0, 1.0] * 3)
    axes = [np.arange(-1.0 + d / 2, 1.0, d) for d in cell_spacing]
    position = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    setup = CouplerSetup(
        transfer_method="buffered_m4_renewal",
        transfer_region_bounds=tuple(box),
        transfer_vorticity_cutoff=0.0,
    )
    coupler = SimpleNamespace(
        setup=setup,
        kinematic_viscosity=0.01,
        fvm_box=box,
        vpm_core_radius_ratio=1.0,
        vpm_particle_spacing=h,
        vpm_time_step_size=0.01,
        vpm_solver=SimpleNamespace(
            viscous_scheme="GBD",
            setup=SimpleNamespace(
                viscous=SimpleNamespace(
                    gbd_threshold_mode="absolute",
                    gbd_threshold=0.0,
                )
            ),
        ),
    )
    donors = SimpleNamespace(
        setup=SimpleNamespace(boundaries=[]),
        get_cell_centre_coordinates=lambda: position,
        get_cell_volume=lambda: np.full(len(position), np.prod(cell_spacing)),
    )
    transfer = VorticityTransfer(coupler)
    transfer.setup(donors)
    lattice = transfer._stable_renewal_lattice
    interior = np.all(np.abs(lattice.positions) < 0.75, axis=1)
    # Rigid rotation: divergence zero, curl exactly (0, 0, 1), regardless
    # of FVM/VPM spacing or lattice phase.
    gradient = np.zeros((len(position), 3, 3))
    gradient[:, 1, 0] = -0.5
    gradient[:, 0, 1] = 0.5
    velocity = position @ gradient[0]
    strength = vortex_strength_from_velocity_trace(
        lattice.positions,
        h,
        lambda points: transfer._velocity_trace.sample(points, velocity, gradient),
    )
    target = strength[interior] * lattice.mesh_weight[interior, None] / h**3
    error = float(np.max(np.abs(target - [0.0, 0.0, 1.0])))
    record_property("constant_vorticity_max_error", error)
    assert error < 3.0e-14
    np.testing.assert_array_equal(lattice.fvm_authority[interior], 1.0)


@pytest.mark.qualification
@pytest.mark.parametrize("ratio", [0.5, 1.0, 1.7])
def test_lattice_representation_matches_independent_gaussian_sum(ratio, record_property):
    shape = (13, 11, 9)
    h = 0.17
    sigma = ratio * h
    positions = np.indices(shape).reshape(3, -1).T * h
    strength = np.zeros((len(positions), 3))
    source_rows = np.array([0, 132, 777, len(positions) - 1])
    strength[source_rows] = [[1, -2, 0.5], [0.2, 1, -0.3], [-0.9, 0.3, 1.1], [0.4, 0.2, -1]]
    displacement = positions[:, None] - positions[source_rows]
    kernel = np.exp(-np.sum(displacement**2, axis=-1) / sigma**2) / (np.pi**1.5 * sigma**3)
    expected = h**3 * kernel @ strength[source_rows]
    actual = gaussian_represented_vortex_strength(strength, shape, h, core_radius=sigma)
    error = float(np.max(np.abs(actual - expected)))
    record_property("physical_gaussian_max_error", error)
    assert error < 8.0e-15


def _native_wall_view(mesh):
    view = CouplerInterfaceMixin()
    view.mesh_data = mesh
    view.boundaries = mesh["boundary"]
    view.setup = SimpleNamespace(boundaries=[SimpleNamespace(name="body", mesh_type="wall")])
    return view


def test_native_wall_triangles_preserve_orientation_and_area():
    axis = np.linspace(-1.0, 1.0, 9)
    mesh = box_mesh_3d(axis, axis, axis, hole_box=(-0.5, 0.5) * 3, wall_patch_name="body")
    triangles = _native_wall_view(mesh).get_wall_surface_triangles()
    area_vector = 0.5 * np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    np.testing.assert_allclose(np.linalg.norm(area_vector, axis=1).sum(), 6.0, atol=1e-14)
    assert np.all(np.einsum("ij,ij->i", area_vector, triangles.mean(axis=1)) < 0)
    wall = TriangulatedWall(triangles, [-1, 1] * 3)
    query = np.array([[0, 0, 0], [0.49, 0, 0], [0.75, 0, 0], [2, 0, 0]])
    np.testing.assert_array_equal(wall.contains(query), [True, True, False, False])
    np.testing.assert_allclose(
        wall.signed_distance(query)[:3], [-0.5, -0.01, 0.25], rtol=0.0, atol=2e-14
    )


@pytest.mark.parametrize("capped", [False, True])
def test_curved_cylinder_wall_is_recognized_without_a_box_approximation(capped):
    # Oriented polygonal cylinder; every normal points into the body from
    # the fluid. The open case represents walls crossing the span boundaries.
    angle = np.linspace(0, 2 * np.pi, 65)[:-1]
    lower = np.column_stack((0.5 * np.cos(angle), 0.5 * np.sin(angle), -np.ones(64)))
    upper = lower.copy()
    upper[:, 2] = 1
    triangles = []
    for i in range(64):
        j = (i + 1) % 64
        triangles.extend(([lower[i], upper[j], lower[j]], [lower[i], upper[i], upper[j]]))
        if capped:
            triangles.extend(([[0, 0, -1], lower[i], lower[j]], [[0, 0, 1], upper[j], upper[i]]))
    wall = TriangulatedWall(np.array(triangles), [-2, 2, -2, 2, -1, 1])
    if capped:
        assert wall.verified_cylinder_z is None
    else:
        assert wall.verified_cylinder_z is not None
        assert wall.contains(np.array([[0.0, 0.0, 1.5]]))[0]
        internal_wall = TriangulatedWall(np.array(triangles), [-2, 2, -2, 2, -2, 2])
        assert internal_wall.verified_cylinder_z is None
    query = np.array([[0, 0, 0], [0.3, 0.3, 0.4], [0.45, 0.45, 0.4], [0.6, 0, -0.4]])
    np.testing.assert_array_equal(wall.contains(query), [True, True, False, False])
    first = wall.signed_distance(query)
    first[:] = 99
    assert wall.signed_distance(query)[0] < 0  # cached geometry cannot be mutated by callers

    triangles = np.asarray(triangles)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    axis = np.arange(-0.875, 1, 0.25)
    donors = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    donors = donors[np.linalg.norm(donors[:, :2], axis=1) > 0.5]
    cfg = CouplerSetup(
        transfer_method="buffered_m4_renewal", transfer_region_bounds=(-0.9, 0.9) * 3
    )
    coupler = SimpleNamespace(
        setup=cfg,
        kinematic_viscosity=0.01,
        fvm_box=np.array([-1, 1] * 3),
        vpm_core_radius_ratio=1.0,
        vpm_particle_spacing=0.125,
        vpm_time_step_size=0.01,
        vpm_solver=SimpleNamespace(
            viscous_scheme="GBD",
            setup=SimpleNamespace(
                viscous=SimpleNamespace(
                    gbd_threshold_mode="absolute",
                    gbd_threshold=0.0,
                )
            ),
        ),
    )
    fvm = SimpleNamespace(
        setup=SimpleNamespace(boundaries=[SimpleNamespace(name="cylinder", mesh_type="wall")]),
        get_cell_centre_coordinates=lambda: donors,
        get_cell_volume=lambda: np.full(len(donors), 0.25**3),
        get_boundary_face_centre_coordinates=lambda patch: triangles.mean(axis=1),
        get_boundary_face_normal=lambda patch: normals,
        get_wall_surface_triangles=lambda: triangles,
    )
    transfer = VorticityTransfer(coupler)
    transfer.setup(fvm)
    assert transfer._body_bounds is None
    np.testing.assert_array_equal(
        transfer._points_in_solid(query, include_boundary=True), [True, True, False, False]
    )
