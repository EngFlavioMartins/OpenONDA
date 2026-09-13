"""Local bound/free exchange, RK acceptance and vector-valued wake deposition."""

from types import SimpleNamespace

from _flat_plate_geometry import create_flat_plate
import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver
from source.solvers.vpm.numerics.rk_tableaux import RK2, RK4, SSPRK3
from source.solvers.vpm.numerics.runge_kutta import RungeKutta
from source.solvers.vpm.physics.induction.base import StageRates, StageState
from source.solvers.vpm.physics.stage_rhs import StageRHS, VLMStageContribution


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    yield
    ti.reset()


def _plate():
    geometry = create_flat_plate(
        chord=1,
        span=4,
        angle_of_attack_degrees=8,
        n_chordwise_panels=2,
        n_spanwise_panels=2,
    )
    vlm = VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(geometry),), dtype="f64"))
    vlm.generate_mesh()
    lattice = vlm.lattice
    gamma = np.linspace(0.2, 1.0, lattice.n_panels)
    gamma[lattice.is_mirrored.to_numpy() == 1] *= -0.7
    lattice.circulation.from_numpy(gamma)
    vlm._compute_cumulative_circulation_cpu()
    vlm._solved = vlm._coupled_mode = True
    return vlm


def _fields():
    position = ti.Vector.field(3, ti.f64, shape=3)
    strength = ti.Vector.field(3, ti.f64, shape=3)
    radius = ti.field(ti.f64, shape=3)
    position.from_numpy(np.array([[0.8, 0.4, 0.3], [1.2, -0.5, -0.2], [9.0, 9.0, 9.0]]))
    strength.from_numpy(np.array([[0.1, 0.3, -0.2], [-0.2, 0.1, 0.4], [8.0, 8.0, 8.0]]))
    radius.from_numpy(np.array([0.3, 0.5, 0.1]))
    return position, strength, radius


def _provider(vlm, scheme="transposed"):
    physics = SimpleNamespace(
        accumulator_dtype=ti.f64,
        max_n_particles=3,
        induction=SimpleNamespace(stretching_scheme=scheme),
    )
    return VLMStageContribution(vlm, physics)


def _independent_strip_field(vlm, targets, radii):
    """Integrate oriented line elements without calling the production kernels."""
    lattice = vlm.lattice
    corners = lattice.panel_corner_position.to_numpy()
    points = lattice.vortex_point_position.to_numpy()
    edges, strip_by_panel = vlm._build_trailing_edge_strip_map()
    gamma = lattice.circulation.to_numpy()
    nodes, weights = np.polynomial.legendre.leggauss(80)
    result = np.zeros((len(edges), len(targets), 3))
    for panel, strip in enumerate(strip_by_panel):
        edge = edges[strip]
        vertices = [corners[edge, 3], points[panel, 1], points[panel, 2], corners[edge, 2]]
        for start, end in zip(vertices[:-1], vertices[1:], strict=True):
            segment = end - start
            sources = start + 0.5 * (nodes[:, None] + 1) * segment
            r = targets[:, None] - sources
            denominator = (np.sum(r * r, axis=-1) + radii[:, None] ** 2) ** 1.5
            result[strip] += (
                gamma[panel]
                / (8 * np.pi)
                * np.sum(
                    np.cross(segment, r) / denominator[..., None] * weights[None, :, None], axis=1
                )
            )
    return result


@pytest.mark.parametrize("scheme", ["direct", "transposed", "mixed"])
@pytest.mark.parametrize("with_gradient", [False, True])
def test_each_strip_reaction_matches_independent_filtered_line_integrals(scheme, with_gradient):
    vlm = _plate()
    position, strength, radius = _fields()
    velocity = ti.Vector.field(3, ti.f64, shape=3)
    rate = ti.Vector.field(3, ti.f64, shape=3)
    gradient = ti.Matrix.field(3, 3, ti.f64, shape=3) if with_gradient else None
    velocity.fill(2.0)
    rate.fill(3.0)
    if gradient is not None:
        gradient.fill(4.0)
    x, alpha, sigma = position.to_numpy()[:2], strength.to_numpy()[:2], radius.to_numpy()[:2]
    reference_velocity = _independent_strip_field(vlm, x, sigma)
    h = 1e-5
    jacobian = np.stack(
        [
            (
                _independent_strip_field(vlm, x + h * axis, sigma)
                - _independent_strip_field(vlm, x - h * axis, sigma)
            )
            / (2 * h)
            for axis in np.eye(3)
        ],
        axis=-1,
    )
    direct = np.einsum("spij,pj->spi", jacobian, alpha)
    transpose = np.einsum("spji,pj->spi", jacobian, alpha)
    expected = {"direct": direct, "transposed": transpose, "mixed": 0.5 * (direct + transpose)}[
        scheme
    ]
    provider = _provider(vlm, scheme)
    state = StageState(position, strength, radius, 2, 0.0)
    rates = StageRates(velocity, rate, gradient)
    with provider.integration_step(RK2(), 0.02, True):
        old = vlm._transported_bound.to_numpy().copy()
        provider.add_stage_rates(state, 0.0, rates)
        edges, _ = vlm._build_trailing_edge_strip_map()
        np.testing.assert_allclose(
            (vlm._transported_bound.to_numpy() - old)[edges],
            -0.01 * expected.sum(axis=1),
            atol=2e-12,
            rtol=2e-8,
        )
    np.testing.assert_allclose(
        velocity.to_numpy()[:2], 2 + reference_velocity.sum(axis=0), atol=2e-12
    )
    np.testing.assert_allclose(rate.to_numpy()[:2], 3 + expected.sum(axis=0), atol=2e-10)
    np.testing.assert_array_equal(velocity.to_numpy()[2], [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(rate.to_numpy()[2], [3.0, 3.0, 3.0])
    if gradient is not None:
        np.testing.assert_allclose(gradient.to_numpy()[:2], 4 + jacobian.sum(axis=0), atol=2e-9)
        np.testing.assert_array_equal(gradient.to_numpy()[2], np.full((3, 3), 4.0))


class _NoFreeInduction:
    def evaluate_stage(self, **fields):
        fields["velocity_out"].fill(0.0)
        fields["vortex_strength_rate_out"].fill(0.0)
        if fields["velocity_gradient_out"] is not None:
            fields["velocity_gradient_out"].fill(0.0)


@pytest.mark.parametrize("tableau", [RK2(), SSPRK3(), RK4()], ids=lambda t: t.name)
@pytest.mark.parametrize("strength_enabled", [False, True])
def test_rk_exchange_balances_actual_particle_increment_and_ignores_probes(
    tableau, strength_enabled
):
    vlm = _plate()
    position, strength, radius = _fields()
    rhs = StageRHS(_NoFreeInduction(), (_provider(vlm),), strength_enabled=strength_enabled)
    rk = RungeKutta(max_n_particles=3, dtype=ti.f64, tableau=tableau)
    old_strength = strength.to_numpy().copy()
    old_bound = vlm.compute_total_bound_vortex_strength()
    rk.advance(
        position=position,
        vortex_strength=strength,
        core_radius=radius,
        count=2,
        time=0.0,
        time_step_size=0.03,
        right_hand_side=rhs,
    )
    assert vlm._bound_transport_ready
    transported = vlm._transported_bound.to_numpy().copy()
    np.testing.assert_allclose(
        transported.sum(axis=0) - old_bound + (strength.to_numpy() - old_strength).sum(axis=0),
        0.0,
        atol=3e-15,
    )
    if not strength_enabled:
        np.testing.assert_array_equal(strength.to_numpy(), old_strength)
    # Health/diagnostic evaluations after integration must not count as RK work.
    rhs.evaluate(
        StageState(position, strength, radius, 2, 0.03),
        0.03,
        StageRates(rk.stage_velocity[0], rk.stage_strength_rate[0]),
    )
    np.testing.assert_array_equal(vlm._transported_bound.to_numpy(), transported)


def test_failed_stage_does_not_publish_exchange_or_change_accepted_particles():
    vlm = _plate()
    position, strength, radius = _fields()
    before = position.to_numpy().copy(), strength.to_numpy().copy()

    class FailSecondStage:
        calls = 0

        def add_stage_rates(self, state, time, rates):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("deliberate stage failure")

    provider = _provider(vlm)
    rhs = StageRHS(_NoFreeInduction(), (provider, FailSecondStage()))
    rk = RungeKutta(max_n_particles=3, dtype=ti.f64)
    with pytest.raises(RuntimeError, match="deliberate stage failure"):
        rk.advance(
            position=position,
            vortex_strength=strength,
            core_radius=radius,
            count=2,
            time=0.0,
            time_step_size=0.03,
            right_hand_side=rhs,
        )
    assert not vlm._bound_transport_ready
    assert provider._stage_weights is None
    np.testing.assert_array_equal(
        vlm._transported_bound.to_numpy(),
        np.zeros_like(vlm._transported_bound.to_numpy()),
    )
    np.testing.assert_array_equal(position.to_numpy(), before[0])
    np.testing.assert_array_equal(strength.to_numpy(), before[1])


def test_nonparallel_exchange_is_emitted_even_without_a_circulation_change():
    vlm = _plate()
    lattice = vlm.lattice
    lattice.cumulative_circulation_old.from_numpy(lattice.cumulative_circulation.to_numpy())
    offsets = lattice.wake_offset.to_numpy()
    offsets[:] = [0.2, 0.0, 0.0]
    lattice.wake_offset.from_numpy(offsets)
    edges, _ = vlm._build_trailing_edge_strip_map()
    reaction = np.array([0.003, -0.002, 0.004])
    with vlm.particle_transport_step():
        values = vlm._transported_bound.to_numpy()
        values[edges[0]] += reaction
        vlm._transported_bound.from_numpy(values)
    matrix, constant, _ = vlm._near_wake_particle_influence()
    vlm._compute_wake_particles()
    count = lattice.n_wake_particles[None]
    strengths = lattice.wake_vortex_strength.to_numpy()[:count]
    np.testing.assert_allclose(strengths.sum(axis=0), reaction, atol=2e-15)
    assert np.min(np.linalg.norm(strengths - reaction, axis=1)) < 2e-15
    targets = lattice.collocation_point.to_numpy()
    normals = lattice.normal.to_numpy()
    positions = lattice.wake_position.to_numpy()[:count]
    radii = lattice.wake_core_radius.to_numpy()[:count]
    actual_velocity = sum(
        vlm._wake_kernel.velocity_pair(targets - x, alpha, sigma, sigma)
        for x, alpha, sigma in zip(positions, strengths, radii, strict=True)
    )
    predicted = matrix @ lattice.cumulative_circulation.to_numpy()[edges] + constant
    np.testing.assert_allclose(
        np.einsum("ij,ij->i", actual_velocity, normals), predicted, atol=2e-15
    )
