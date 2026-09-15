"""Independent filament, finite-wing, moving-frame and output regressions."""

from types import SimpleNamespace

from _flat_plate_geometry import (
    create_flat_plate,
    lifting_line_circulation,
    lifting_line_polar,
)
import numpy as np
import pytest
import pyvista as pv
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.coupling.kinematics import (
    SmoothRampVLM,
    TranslatingVLM,
)
from source.solvers.vpm.boundary_elements.vlm.kernels.biot_savart import (
    regularized_segment_velocity_and_gradient,
)
from source.solvers.vpm.boundary_elements.vlm.solver.linear_solvers import (
    ScipySolver,
    TaichiBiCGSTABSolver,
)
from source.solvers.vpm.boundary_elements.vlm.solver.loading_distribution import (
    VLMLoadingDistribution,
)
from source.solvers.vpm.boundary_elements.vlm.solver.restart import (
    restore_vlm_restart,
    validate_vlm_restart,
    write_vlm_restart,
)
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver
from source.solvers.vpm.config.artifacts import Backup, Samplers
from source.solvers.vpm.io.sampler import OutputEvent, OutputManager
from source.solvers.vpm.io.sampling import EverySteps, VLMSampler


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    yield
    ti.reset()


@ti.kernel
def _segment_probe(
    points: ti.template(), velocity: ti.template(), gradient: ti.template(), radius: float
):
    for i in range(points.shape[0]):
        velocity[i], gradient[i] = regularized_segment_velocity_and_gradient(
            points[i],
            ti.Vector([0.1, -0.7, 0.2]),
            ti.Vector([0.4, 0.8, -0.1]),
            1.3,
            radius,
        )


def _sample_segment(points, radius):
    x = ti.Vector.field(3, ti.f64, shape=len(points))
    v = ti.Vector.field(3, ti.f64, shape=len(points))
    g = ti.Matrix.field(3, 3, ti.f64, shape=len(points))
    x.from_numpy(np.asarray(points, dtype=float))
    _segment_probe(x, v, g, radius)
    return v.to_numpy(), g.to_numpy()


def test_filament_velocity_and_jacobian_match_independent_quadrature():
    a = np.array([0.1, -0.7, 0.2])
    b = np.array([0.4, 0.8, -0.1])
    points = np.array([[0.3, 0.2, 0.8], [0.1, -0.7, 0.2], (a + b) / 2, [2.0, -0.3, 0.4]])
    radius = 0.12
    nodes, weights = np.polynomial.legendre.leggauss(200)
    sources = a + (nodes[:, None] + 1) / 2 * (b - a)

    def quadrature(targets):
        r = targets[:, None] - sources
        return (
            1.3
            / (4 * np.pi)
            * np.sum(
                np.cross(b - a, r)
                / (np.sum(r * r, axis=2) + radius**2)[..., None] ** 1.5
                * weights[None, :, None]
                / 2,
                axis=1,
            )
        )

    velocity, gradient = _sample_segment(points, radius)
    np.testing.assert_allclose(velocity, quadrature(points), rtol=2e-11, atol=1e-12)
    for j in range(3):
        offset = np.eye(3)[j] * 1e-5
        reference = (quadrature(points + offset) - quadrature(points - offset)) / (2e-5)
        np.testing.assert_allclose(gradient[:, :, j], reference, rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(np.trace(gradient, axis1=1, axis2=2), 0.0, atol=2e-13)


def _plate(angle=5.0, motion=None, placement=None, name=None, nc=8, ns=14, wake_core_overlap=None):
    plate = create_flat_plate(
        chord=1.0,
        span=10.0,
        angle_of_attack_degrees=angle,
        n_chordwise_panels=nc,
        n_spanwise_panels=ns,
    )
    setup = VLMSurfaceSetup(plate, kinematics=motion, name=name, translation=placement)
    solver = VLMSolver(
        VLMSetup(
            surfaces=(setup,),
            dtype="f64",
            freestream_velocity=(10.0, 0.0, 0.0),
            sample_surface_forces=True,
            wake_core_overlap=wake_core_overlap,
        )
    )
    solver.generate_mesh()
    return solver


def _solve_steady(solver, incident):
    solver.advance_time(0.0125, 0.0125)
    velocity = np.broadcast_to(incident, (solver.lattice.n_panels, 3)).copy()
    solver.solve(velocity)
    solver.compute_postprocess(velocity, np.array([10.0, 0.0, 0.0]), 1.0)
    return solver.compute_forces(1.0, np.array([10.0, 0.0, 0.0]))


@pytest.mark.parametrize("moving", [False, True])
def test_conservation_tracker_records_dimensional_surface_force(moving, monkeypatch):
    from source.solvers.vpm.diagnostics.conservation import ConservationTracker

    motion = TranslatingVLM(velocity=[-10.0, 0.0, 0.0]) if moving else None
    vlm = _plate(nc=2, ns=3, motion=motion)
    _solve_steady(vlm, [10.0, 0.0, 0.0])
    density = 1.3
    expected = density * vlm.lattice.get_forces().sum(axis=0)
    assert expected[2] > 100.0
    flow = SimpleNamespace(
        time=0.0125,
        net_vortex_strength=np.zeros(3),
        total_linear_impulse=np.zeros(3),
        total_kinetic_energy=0.0,
        kinetic_energy_rate=0.6,
        viscous_kinetic_energy_rate=-0.4,
        particles=SimpleNamespace(n_particles_total=0),
        vlm_solver=vlm,
        freestream_velocity=np.array([0.0, 0.0, 0.0] if moving else [10.0, 0.0, 0.0]),
    )
    tracker = ConservationTracker(density)
    state = tracker.record_state(flow)
    np.testing.assert_allclose(state.kutta_joukowski_force, expected, rtol=1e-12)
    np.testing.assert_allclose(state.impulse_total, density * vlm.compute_bound_linear_impulse())
    assert state.viscous_kinetic_energy_rate == pytest.approx(-0.4 * density)

    def unavailable_force(**kwargs):
        raise RuntimeError("surface force unavailable")

    monkeypatch.setattr(vlm, "compute_forces", unavailable_force)
    with pytest.raises(RuntimeError, match="surface force unavailable"):
        tracker.record_state(flow)
    assert len(tracker.history) == 1


@pytest.mark.parametrize("coupled", [False, True])
def test_bound_impulse_matches_filament_quadrature_and_translation(coupled):
    solver = _plate(nc=3, ns=4)
    _solve_steady(solver, [10.0, 0.0, 0.0])
    # The standalone solve resets its mode; select the finite field being integrated afterwards.
    solver._coupled_mode = coupled
    lattice = solver.lattice
    n = lattice.n_panels
    vortex = lattice.vortex_point_position.to_numpy()
    corners = lattice.panel_corner_position.to_numpy()
    gamma = lattice.circulation.to_numpy()[:n]
    trailing = lattice.trailing_edge_index.to_numpy()[:n]
    nodes, weights = np.polynomial.legendre.leggauss(6)
    expected = np.zeros(3)
    for index in range(n):
        vertices = [vortex[index, 1], vortex[index, 2]]
        if coupled:
            vertices = [corners[trailing[index], 3], *vertices, corners[trailing[index], 2]]
        for start, end in zip(vertices[:-1], vertices[1:], strict=True):
            points = start + 0.5 * (nodes[:, None] + 1) * (end - start)
            expected += (
                0.25
                * gamma[index]
                * np.sum(weights[:, None] * np.cross(points, end - start), axis=0)
            )
    impulse = solver.compute_bound_linear_impulse()
    np.testing.assert_allclose(impulse, expected, atol=2e-12, rtol=2e-12)
    shift = np.array([1.3, -0.7, 0.9])
    expected_shift = 0.5 * np.cross(shift, solver.compute_total_bound_vortex_strength())
    lattice.vortex_point_position.from_numpy(vortex + shift)
    lattice.panel_corner_position.from_numpy(corners + shift)
    np.testing.assert_allclose(
        solver.compute_bound_linear_impulse(), impulse + expected_shift, atol=2e-12, rtol=2e-12
    )


def test_conservation_export_distinguishes_wake_bound_and_total_impulse(tmp_path):
    import pandas as pd

    from source.solvers.vpm.diagnostics.conservation import ConservationState, ConservationTracker

    tracker = ConservationTracker()
    tracker.history.append(
        ConservationState(
            time=1.0,
            impulse_wake=np.array([1.0, 2.0, 3.0]),
            impulse_bound=np.array([4.0, 5.0, 6.0]),
            impulse_total=np.array([5.0, 7.0, 9.0]),
        )
    )
    row = pd.read_csv(tracker.export_csv(tmp_path)).iloc[0]
    # Retain the established wake-only CSV columns; total has an explicit name.
    assert row.linear_impulse_z == 3.0
    assert row.bound_linear_impulse_z == 6.0
    assert row.total_linear_impulse_z == 9.0


def test_mesh_honors_initial_trailing_leg_length():
    from source.solvers.vpm.boundary_elements.vlm.solver.mesh import generate_vlm_mesh

    solver = _plate(nc=2, ns=2)
    generate_vlm_mesh(solver.aircraft, solver.lattice, trailing_edge_infty=23.0)
    vertices = solver.lattice.vortex_point_position.to_numpy()
    direction = solver.lattice.trailing_direction.to_numpy()
    np.testing.assert_allclose(vertices[:, 0] - vertices[:, 1], 23.0 * direction[:, 0])
    np.testing.assert_allclose(vertices[:, 3] - vertices[:, 2], 23.0 * direction[:, 1])


def test_coupled_bound_strength_includes_the_on_wing_trailing_legs(tmp_path):
    """A closing TE filament must cancel the entire bound horseshoe vector."""
    solver = _plate(nc=2, ns=2)
    lattice = solver.lattice
    n = lattice.n_panels
    corners = lattice.panel_corner_position.to_numpy()
    # Taper/sweep the downstream edge so it differs from the quarter chord.
    corners[:, 2, 0] += 0.2 * corners[:, 2, 1]
    corners[:, 2, 2] += 0.1 * corners[:, 2, 1]
    corners[:, 3, 0] += 0.2 * corners[:, 3, 1]
    corners[:, 3, 2] += 0.1 * corners[:, 3, 1]
    corners[:, 2, 1] *= 0.8
    corners[:, 3, 1] *= 0.8
    lattice.panel_corner_position.from_numpy(corners)
    gamma = np.linspace(0.5, 1.7, n)
    lattice.circulation.from_numpy(gamma)
    solver._solved = True
    solver._coupled_mode = True
    points = lattice.vortex_point_position.to_numpy()
    te = lattice.trailing_edge_index.to_numpy()
    closed = np.zeros(3)
    for i in range(n):
        vertices = np.array([corners[te[i], 3], points[i, 1], points[i, 2], corners[te[i], 2]])
        closed += gamma[i] * np.diff(vertices, axis=0).sum(axis=0)
    np.testing.assert_allclose(solver.compute_total_bound_vortex_strength(), closed, atol=1e-13)
    closing_wake = np.sum(gamma[:, None] * (corners[te, 3] - corners[te, 2]), axis=0)
    np.testing.assert_allclose(closed + closing_wake, 0.0, atol=1e-13)

    from collections import defaultdict

    from source.solvers.vpm.boundary_elements.vlm.solver.diagnostics import VLMDiagnostics

    history = defaultdict(list)
    solver._last_forces = {"lift_coefficient": 0.0, "drag_coefficient": 0.0}
    solver.logging_interval_steps = 2
    VLMDiagnostics.record_vlm_diagnostics(
        solver,
        SimpleNamespace(n_particles_total=1),
        closing_wake[None, :],
        history,
        step=1,
        time=0.1,
        case_dir=str(tmp_path),
    )
    assert history["vlm_bound_vortex_strength_y"] == pytest.approx([closed[1]])
    assert history["vlm_wake_vortex_strength_y"] == pytest.approx([-closed[1]])


def test_lifting_line_reference_matches_independent_full_span_collocation():
    n = np.arange(1, 161)
    theta = np.pi * n / 161
    mu = np.pi / 20
    matrix = np.sin(theta[:, None] * n) * (np.sin(theta[:, None]) + mu * n)
    a = np.linalg.solve(matrix, mu * np.radians(5) * np.sin(theta))
    cl, cd = lifting_line_polar(5.0, 10.0)
    np.testing.assert_allclose(
        [cl, cd], [np.pi * 10 * a[0], np.pi * 10 * np.sum(n * a * a)], rtol=2e-6
    )
    np.testing.assert_allclose(lifting_line_polar(5.0, 10.0, 40), [cl, cd], rtol=2e-5)
    cl_minus, cd_minus = lifting_line_polar(-5.0, 10.0)
    assert cl_minus == -cl and cd_minus == cd


@pytest.mark.parametrize("overlap", [None, 2.5])
def test_asymmetric_wing_halves_shed_one_shared_root_filament(overlap):
    """A newborn closed row must preserve strength under unequal half-wing loads."""
    solver = _plate(nc=1, ns=2, wake_core_overlap=overlap)
    lattice = solver.lattice
    gamma = np.array([1.0, 2.0, -0.4, -0.8])
    lattice.circulation.from_numpy(gamma)
    lattice.cumulative_circulation_old.fill(0.0)
    displacement = np.array([0.2, 0.05, 0.03])
    offsets = lattice.wake_offset.to_numpy()
    offsets[:] = displacement
    lattice.wake_offset.from_numpy(offsets)
    solver._solved = solver._coupled_mode = True
    solver._compute_wake_particles()
    count = lattice.n_wake_particles[None]
    strength = lattice.wake_vortex_strength.to_numpy()[:count]
    np.testing.assert_allclose(
        solver.compute_total_bound_vortex_strength() + strength.sum(axis=0),
        0.0,
        atol=2e-14,
    )
    corners = lattice.panel_corner_position.to_numpy()
    root_midpoint = corners[0, 3] + 0.5 * displacement
    position = lattice.wake_position.to_numpy()[:count]
    root = np.linalg.norm(position - root_midpoint, axis=1) < 1e-13
    assert root.sum() == 1
    np.testing.assert_allclose(strength[root][0], -(gamma[0] + gamma[2]) * displacement)
    if overlap is not None:
        # Every element in this uniform 2.5 m span grid must have the
        # requested overlap, including trailing, root and starting elements.
        np.testing.assert_allclose(lattice.wake_core_radius.to_numpy()[:count], overlap * 2.5)


@pytest.mark.parametrize("body_rotation", [0.0, 0.23])
def test_shedding_closes_a_deformed_previous_wake(body_rotation):
    """Changing bound loads must close against a geometrically deformed old wake."""
    solver = _plate(nc=2, ns=3)
    lattice = solver.lattice
    corners = lattice.panel_corner_position.to_numpy()
    n = lattice.n_panels
    old_gamma = np.linspace(0.3, 1.1, n)
    old_gamma[lattice.is_mirrored.to_numpy() == 1] *= -0.7
    lattice.circulation.from_numpy(old_gamma)
    solver._compute_cumulative_circulation_cpu()
    old_cumulative = lattice.cumulative_circulation.to_numpy()
    lattice.cumulative_circulation_old.from_numpy(old_cumulative)

    # Transport a closing polyline in a prescribed affine fluid deformation.
    # Summing its oriented segment integrals is independent of the emitter's
    # spanwise differencing and its particle layout.
    deformation = np.array([[1.0, 0.04, 0.0], [0.0, 1.0, 0.0], [0.0, -0.03, 1.0]])
    fluid_translation = np.array([0.3, -0.04, 0.02])
    old_wake_strength = np.zeros(3)
    trailing = np.flatnonzero(lattice.is_trailing_edge.to_numpy())
    for i in trailing:
        left, right = corners[i, 3], corners[i, 2]
        far = np.array([2.0, 0.1, -0.2])
        vertices = np.array([right, right + far, left + far, left])
        transported = vertices @ deformation.T + fluid_translation
        old_wake_strength += old_cumulative[i] * np.diff(transported, axis=0).sum(axis=0)

    cosine, sine = np.cos(body_rotation), np.sin(body_rotation)
    rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    current = corners @ rotation.T + np.array([0.02, -0.03, 0.01])
    lattice.panel_corner_position.from_numpy(current)
    offsets = lattice.wake_offset.to_numpy()
    for i in trailing:
        offsets[i] = corners[i, [3, 2]] @ deformation.T + fluid_translation - current[i, [3, 2]]
    lattice.wake_offset.from_numpy(offsets)
    new_gamma = old_gamma * (1.0 + 0.2 * np.sin(np.arange(n)))
    lattice.circulation.from_numpy(new_gamma)
    solver._solved = solver._coupled_mode = True
    solver._compute_wake_particles()
    count = lattice.n_wake_particles[None]
    newborn_strength = lattice.wake_vortex_strength.to_numpy()[:count].sum(axis=0)
    np.testing.assert_allclose(
        solver.compute_total_bound_vortex_strength() + old_wake_strength + newborn_strength,
        0.0,
        atol=3e-14,
    )


@pytest.mark.parametrize("overlap", [0, -1, float("inf"), float("nan")])
def test_wake_overlap_is_validated_by_the_solver_api(overlap):
    with pytest.raises(ValueError, match="wake_core_overlap"):
        VLMSetup(surfaces=(VLMSurfaceSetup(create_flat_plate()),), wake_core_overlap=overlap)


def test_steady_plate_lift_loading_and_drag_match_lifting_line():
    solver = _plate()
    forces = _solve_steady(solver, [10.0, 0.0, 0.0])
    cl, cd = lifting_line_polar(5.0, 10.0)
    assert forces["lift_coefficient"] == pytest.approx(cl, rel=0.05)
    assert forces["drag_coefficient"] == pytest.approx(cd, rel=0.12)
    data = VLMLoadingDistribution.extract_distributions(
        solver, "flat_plate", np.array([10.0, 0.0, 0.0]), 1.0
    )["spanwise"]
    theory = lifting_line_circulation(data.span_coordinate, 10.0, 1.0, np.radians(5), 10.0)
    relative_l2 = np.linalg.norm(
        data.section_lift_coefficient - theory.section_lift_coefficient
    ) / np.linalg.norm(theory.section_lift_coefficient)
    assert relative_l2 < 0.07
    assert np.sum(data.lift_per_span * data.spanwise_station_width) == pytest.approx(
        forces["lift"], rel=2e-12
    )
    assert solver.compute_per_surface_forces(1.0, np.array([10.0, 0.0, 0.0]))["flat_plate"][
        "lift"
    ] == pytest.approx(forces["lift"])


def test_static_and_translating_plate_have_same_forces_and_quarter_chord_moment():
    static = _plate()
    moving = _plate(motion=TranslatingVLM(velocity=np.array([-10.0, 0.0, 0.0])))
    stationary = _solve_steady(static, [10.0, 0.0, 0.0])
    translated = _solve_steady(moving, [0.0, 0.0, 0.0])
    for key in [
        "lift_coefficient",
        "drag_coefficient",
        "pitching_moment_coefficient_quarter_chord",
    ]:
        assert translated[key] == pytest.approx(stationary[key], rel=1e-8, abs=1e-9)
    assert abs(stationary["pitching_moment_coefficient_quarter_chord"]) < 0.003


def test_single_surface_placement_and_alias_apply_to_geometry_and_loading():
    reference = _plate(nc=2, ns=3)
    shifted = _plate(placement=(2.0, 3.0, 4.0), name="chosen", nc=2, ns=3)
    np.testing.assert_allclose(
        shifted.lattice.get_collocation_points() - reference.lattice.get_collocation_points(),
        np.tile([2.0, 3.0, 4.0], (12, 1)),
        atol=1e-12,
    )
    _solve_steady(shifted, [10.0, 0.0, 0.0])
    data = VLMLoadingDistribution.extract_distributions(
        shifted, "chosen", np.array([10.0, 0.0, 0.0]), 1.0
    )
    assert len(data["spanwise"]) == 6
    assert shifted.compute_per_surface_forces(1.0)["chosen"]["panel_count"] == 12


def test_ramp_geometry_reaches_the_requested_clock_without_advancing_one_step_extra():
    motion = SmoothRampVLM(final_velocity=[-10.0, 0.0, 0.0], acceleration_time=0.12)
    solver = _plate(motion=motion, nc=1, ns=2)
    before = solver.lattice.get_collocation_points()
    solver.advance_time(0.0125, 0.0125)
    distance = -5 * (0.0125 - 0.12 / np.pi * np.sin(np.pi * 0.0125 / 0.12))
    np.testing.assert_allclose(
        solver.lattice.get_collocation_points() - before,
        np.tile([distance, 0.0, 0.0], (4, 1)),
        atol=1e-12,
    )


def test_dense_solver_reuses_only_identical_matrices_and_rejects_singular_systems():
    solver = ScipySolver()
    matrix = np.array([[3.0, 1.0], [-1.0, 2.0]])
    rhs = np.array([2.0, 5.0])
    out = np.zeros(2)
    solver.solve(matrix, rhs, out, 2)
    factor = solver._factorization
    solver.solve(matrix, rhs * 2, out, 2)
    assert solver._factorization is factor
    np.testing.assert_allclose(matrix @ out, rhs * 2)
    matrix[0, 0] = 4.0
    solver.solve(matrix, rhs, out, 2)
    assert solver._factorization is not factor
    np.testing.assert_allclose(matrix @ out, rhs)
    with pytest.raises((np.linalg.LinAlgError, Warning)):
        solver.solve(np.zeros((2, 2)), rhs, out, 2)


def test_dense_solver_failed_factorization_cannot_poison_retry_or_output():
    solver = ScipySolver()
    rhs = np.array([1.0, 2.0])
    out = np.zeros(2)
    solver.solve(np.eye(2), rhs, out, 2)
    previous = out.copy()
    singular = np.ones((2, 2))
    for _ in range(2):
        with pytest.raises(np.linalg.LinAlgError):
            solver.solve(singular, rhs, out, 2)
        np.testing.assert_array_equal(out, previous)
    matrix = np.array([[3.0, 1.0], [-1.0, 2.0]])
    solver.solve(matrix, rhs, out, 2)
    np.testing.assert_allclose(matrix @ out, rhs, rtol=1e-14)
    solver.solve(np.eye(2), 2.0 * rhs, out, 2)
    np.testing.assert_allclose(out, 2.0 * rhs, rtol=1e-14)


@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_bicgstab_checks_relative_residual_even_for_small_rhs(scale):
    matrix = ti.field(ti.f64, shape=(3, 3))
    rhs = ti.field(ti.f64, shape=3)
    x = ti.field(ti.f64, shape=3)
    a = np.array([[4.0, 1.0, 0.0], [-1.0, 3.0, 1.0], [0.2, -1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0]) * scale
    matrix.from_numpy(a)
    rhs.from_numpy(b)
    solver = TaichiBiCGSTABSolver(max_n_panels=3)
    solver.solve(matrix, rhs, x, 3, tolerance=1e-10)
    np.testing.assert_allclose(a @ x.to_numpy(), b, rtol=1e-10, atol=scale * 1e-12)
    with pytest.raises(np.linalg.LinAlgError, match="did not converge"):
        solver.solve(matrix, rhs, x, 3, max_iterations=0, tolerance=1e-10)


def test_vlm_sampler_uses_owner_samples_path_and_resumable_polydata_index(tmp_path):
    solver = _plate(nc=2, ns=3)
    _solve_steady(solver, [10.0, 0.0, 0.0])
    samples = Samplers((VLMSampler(schedule=EverySteps(1)),), "case_a")
    runtime = SimpleNamespace(
        case_dir=tmp_path,
        _backup_path=tmp_path / "solution",
        step=1,
        time=0.1,
        time_step_size=0.1,
        case=SimpleNamespace(samplers=samples, backup=Backup()),
        vlm_solver=solver,
    )
    manager = OutputManager(runtime)
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    folder = tmp_path / "samples/case_a"
    assert (folder / "vlm_000001.vtp").is_file()
    assert not (tmp_path / "solution/vlm_000001.vtp").exists()
    runtime.step = 2
    runtime.time = 0.2
    OutputManager(runtime).dispatch(OutputEvent.ACCEPTED_STEP)
    assert (folder / "vlm.pvd").read_text().count("<DataSet") == 2
    import pyvista as pv

    vtk = pv.read(folder / "vlm_000002.vtp")
    expected = 2 * vtk["circulation"] / (10 * vtk["panel_chord"])
    np.testing.assert_allclose(vtk["circulation_pressure_jump_proxy"], expected)
    assert vtk.field_data["time"][0] == 0.2


def test_vlm_sampler_and_backup_use_distinct_owner_series(tmp_path, monkeypatch):
    solver = _plate(nc=2, ns=3)
    _solve_steady(solver, [10.0, 0.0, 0.0])

    class FieldSnapshot:
        file_name = "wake"
        schedule = EverySteps(1)
        vtk_extension = ".vts"

        def save_vtp(self, _solver, filepath, time=None):
            filepath.write_text(f"time={time}", encoding="utf-8")

    samples = Samplers((VLMSampler(schedule=EverySteps(2)), FieldSnapshot()), "case_a")
    runtime = SimpleNamespace(
        case_dir=tmp_path,
        _backup_path=tmp_path / "native_solution",
        step=2,
        time=0.2,
        time_step_size=0.1,
        case=SimpleNamespace(
            samplers=samples,
            backup=Backup(interval_steps=3),
        ),
        vlm_solver=solver,
    )
    calls = []
    original_save_results = solver.save_results

    def counted_save_results(*args, **kwargs):
        calls.append((runtime.step, runtime.time))
        return original_save_results(*args, **kwargs)

    monkeypatch.setattr(solver, "save_results", counted_save_results)

    def write_backup():
        from source.solvers.vpm.io.vlm_backup import write_vlm_backup

        write_vlm_backup(
            runtime.vlm_solver,
            runtime._backup_path,
            step=runtime.step,
            time=runtime.time,
        )

    runtime._write_backup = write_backup
    manager = OutputManager(runtime)
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    runtime.step = 3
    runtime.time = 0.3
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    runtime.step = 4
    runtime.time = 0.4
    manager.dispatch(OutputEvent.ACCEPTED_STEP)

    folder = tmp_path / "native_solution"
    sample_folder = tmp_path / "samples/case_a"
    assert calls == [(2, 0.2), (3, 0.3), (4, 0.4)]
    assert sorted(path.name for path in folder.glob("vlm_*.vtp")) == ["vlm_000003.vtp"]
    assert (folder / "vlm.pvd").read_text().count("<DataSet") == 1
    assert pv.get_reader(folder / "vlm.pvd").time_values == [0.3]
    assert sorted(path.name for path in sample_folder.glob("vlm_*.vtp")) == [
        "vlm_000002.vtp",
        "vlm_000004.vtp",
    ]
    assert pv.get_reader(sample_folder / "vlm.pvd").time_values == [0.2, 0.4]
    from source.solvers.vpm.io.vlm_backup import write_vlm_backup

    write_vlm_backup(runtime.vlm_solver, folder, step=4, time=0.4)
    assert pv.get_reader(folder / "vlm.pvd").time_values == [0.3, 0.4]
    with pytest.raises(ValueError, match="filename/time conflicts"):
        write_vlm_backup(runtime.vlm_solver, folder, step=5, time=0.4)
    assert list((tmp_path / "samples").rglob("vlm_*.vtp"))
    assert list((tmp_path / "samples").rglob("vlm.pvd"))
    assert all((tmp_path / f"samples/case_a/wake_{step:06d}.vts").is_file() for step in (2, 3, 4))
    assert (tmp_path / "samples/case_a/wake.pvd").read_text().count("<DataSet") == 3


def test_vlm_restart_restores_motion_circulation_and_next_solve(tmp_path):
    import h5py

    def make():
        return _plate(motion=TranslatingVLM(velocity=np.array([-10.0, 0.0, 0.0])), nc=2, ns=3)

    original = make()
    _solve_steady(original, [0.0, 0.0, 0.0])
    original._last_reference_velocity = np.array([10.0, 0.0, 0.0])
    path = tmp_path / "vlm.h5"
    with h5py.File(path, "w") as file:
        write_vlm_restart(original, file.create_group("vlm"))
    restored = make()
    with h5py.File(path, "r") as file:
        validate_vlm_restart(restored, file["vlm"])
        restore_vlm_restart(restored, file["vlm"])
    np.testing.assert_array_equal(
        restored.lattice.get_circulation(), original.lattice.get_circulation()
    )
    np.testing.assert_array_equal(
        restored.kinematics.current_position, original.kinematics.current_position
    )
    for solver in (original, restored):
        solver.advance_time(0.0125, 0.025)
        solver.solve(np.zeros((12, 3)), time_step_size=0.0125, coupled=True)
    np.testing.assert_array_equal(
        restored.lattice.get_circulation(), original.lattice.get_circulation()
    )
    np.testing.assert_array_equal(
        restored.lattice.get_collocation_points(), original.lattice.get_collocation_points()
    )
    with h5py.File(path, "a") as file:
        file["vlm/circulation"][0] = np.nan
        with pytest.raises(ValueError, match="circulation"):
            validate_vlm_restart(restored, file["vlm"])


def test_restart_rejects_unsupported_schema_before_restoring_fields(tmp_path):
    import h5py

    solver = _plate(nc=1, ns=2)
    path = tmp_path / "vlm.h5"
    with h5py.File(path, "w") as archive:
        group = archive.create_group("vlm")
        write_vlm_restart(solver, group)
        before = solver.lattice.panel_corner_position.to_numpy().copy()
        group.attrs["version"] = -1
        with pytest.raises(ValueError, match="Incompatible VLM restart version"):
            validate_vlm_restart(solver, group)
        np.testing.assert_array_equal(solver.lattice.panel_corner_position.to_numpy(), before)


def test_multisurface_loading_uses_wing_local_segment_ids_and_reports_output_failures(
    tmp_path, monkeypatch
):
    import warnings

    plate = create_flat_plate(
        chord=1.0,
        span=4.0,
        angle_of_attack_degrees=5.0,
        n_chordwise_panels=2,
        n_spanwise_panels=3,
    )
    solver = VLMSolver(
        VLMSetup(
            surfaces=(
                VLMSurfaceSetup(plate, name="front"),
                VLMSurfaceSetup(plate, name="rear", translation=(5.0, 0.0, 0.0)),
            ),
            dtype="f64",
            freestream_velocity=(10.0, 0.0, 0.0),
            sample_surface_forces=True,
        )
    )
    solver.generate_mesh()
    _solve_steady(solver, [10.0, 0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in ("front", "rear"):
            data = VLMLoadingDistribution.extract_distributions(
                solver, name, np.array([10.0, 0.0, 0.0]), 1.0
            )
            assert len(data["spanwise"]) == 6

    def fail_write(*args, **kwargs):
        raise OSError("sample write failed")

    monkeypatch.setattr(VLMLoadingDistribution, "export_distribution_csv", fail_write)
    with pytest.raises(OSError, match="sample write failed"):
        VLMLoadingDistribution.record_loading_distributions(
            solver, {}, solver.logging_interval_steps, 0.0125, str(tmp_path)
        )
