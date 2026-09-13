"""Affine newborn-row history, moving geometry and finite-target stage transport."""

from types import SimpleNamespace

from _flat_plate_geometry import create_flat_plate
import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.coupling.kinematics import RotatingVLM
from source.solvers.vpm.boundary_elements.vlm.kernels.virtual_wake import make_virtual_wake_kernel
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver
from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.numerics.rk_tableaux import SSPRK3


@pytest.fixture(autouse=True)
def runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=2, offline_cache=False)
    yield
    ti.reset()


@pytest.mark.parametrize("moving", [False, True])
def test_identical_accepted_and_stage_rows_include_history(moving):
    """Independently supply accepted geometry/offsets; compare the complete affine row."""
    geometry = create_flat_plate(
        chord=1, span=2, angle_of_attack_degrees=6, n_chordwise_panels=1, n_spanwise_panels=2
    )
    motion = RotatingVLM(angular_speed=2.0, axis=[1.0, 0, 0]) if moving else None
    vlm = VLMSolver(
        VLMSetup(
            surfaces=(VLMSurfaceSetup(geometry, kinematics=motion),),
            dtype="f64",
            wake_core_overlap=2.5,
        )
    )
    vlm.generate_mesh()
    vlm._current_time = 0.0
    lattice, elapsed = vlm.lattice, 0.01
    count = lattice.n_panels
    old_gamma = np.linspace(0.1, 0.4, lattice.max_n_panels)
    lattice.cumulative_circulation.from_numpy(old_gamma)
    lattice.cumulative_circulation_old.from_numpy(old_gamma)
    initial_corners = lattice.panel_corner_position.to_numpy().copy()
    initial_bound = old_gamma[:, None] * (initial_corners[:, 2] - initial_corners[:, 3])

    class Uniform:
        @staticmethod
        def compute_target_velocity(particles, points, **kwargs):
            return np.tile([2.0, 0, 0], (len(points), 1))

    stage_geometry = vlm._ensure_stage_geometry_fields(elapsed)
    matrix, old_velocity = vlm._near_wake_stage_influence(stage_geometry, None, Uniform(), elapsed)
    assert np.linalg.norm(old_velocity) > 1e-4
    end_corners = stage_geometry[1].to_numpy()
    offsets = lattice.wake_offset.to_numpy()
    offsets[:] = initial_corners[:, [3, 2]] - end_corners[:, [3, 2]] + [2 * elapsed, 0, 0]
    lattice.wake_offset.from_numpy(offsets)
    lattice.panel_corner_position.from_numpy(end_corners)
    lattice.collocation_point.from_numpy(stage_geometry[2].to_numpy())
    lattice.normal.from_numpy(stage_geometry[3].to_numpy())
    vlm._transported_bound.from_numpy(initial_bound)
    vlm._bound_transport_ready = True
    accepted_matrix, accepted_old, strip_map = vlm._near_wake_particle_influence()
    np.testing.assert_allclose(matrix, accepted_matrix[:, strip_map], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(old_velocity, accepted_old, rtol=1e-12, atol=1e-13)
    if moving:
        assert np.max(np.abs(offsets[:count, :, 1:])) > 1e-3


def test_ssprk_midpoint_history_uses_stage_coefficients_not_final_weights():
    """SSPRK3 visits t+h before t+h/2: its midpoint must use a=(1/4,1/4)."""
    geometry = create_flat_plate(chord=1, span=1, n_chordwise_panels=1, n_spanwise_panels=1)
    vlm = VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(geometry),), dtype="f64"))
    vlm.generate_mesh()
    count = vlm.lattice.n_panels
    vlm._transport_tableau = SSPRK3()
    vlm._transport_dt = 0.02
    vlm._stage_bound_initial = np.tile([0.3, 0.4, 0.5], (count, 1))
    vlm._stage_exchange_history = {0: np.ones((count, 3)), 1: np.full((count, 3), 3.0)}
    stage = SimpleNamespace(stage_index=2)
    physics = SimpleNamespace(
        compute_target_velocity=lambda cloud, points, **kw: np.tile([2.0, 0, 0], (len(points), 1))
    )
    vlm._near_wake_stage_influence(vlm._ensure_stage_geometry_fields(0.01), stage, physics, 0.01)
    for _, _, _, constant, _ in vlm._stage_wake_sources:
        if np.any(constant):
            np.testing.assert_allclose(constant, [0.32, 0.42, 0.52], rtol=1e-13)


@pytest.mark.parametrize(
    "name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_virtual_row_rates_and_reaction_match_independent_pair_operator(name):
    """Check velocity, full Jacobian, transposed stretching and exact reaction balance."""
    rng = np.random.default_rng(44)
    count, sources = 9, 4
    x, gamma = rng.normal(size=(count, 3)), rng.normal(size=(count, 3))
    core = rng.uniform(0.1, 0.3, count)
    sx, sg, sc = (
        rng.normal(size=(sources, 3)),
        rng.normal(size=(sources, 3)),
        rng.uniform(0.1, 0.4, sources),
    )
    x[0] = sx[0]
    fields = [ti.Vector.field(3, dtype=ti.f64, shape=count) for _ in range(4)]
    position, strength, velocity, rate = fields
    radius = ti.field(ti.f64, shape=count)
    gradient = ti.Matrix.field(3, 3, ti.f64, shape=count)
    exchange = ti.Vector.field(3, ti.f64, shape=2)
    position.from_numpy(x)
    strength.from_numpy(gamma)
    radius.from_numpy(core)
    owners = np.array([0, 1, 1, 0], dtype=np.int32)
    kernel = make_virtual_wake_kernel(name, ti.f64)
    # The enum is obtained from the public formulation map, not assumed here.
    from source.solvers.vpm.physics.induction.base import _STRETCHING_MODES

    mode = _STRETCHING_MODES["TRANSPOSED"]
    kernel(
        position,
        strength,
        radius,
        velocity,
        rate,
        gradient,
        sx,
        sg,
        sc,
        owners,
        exchange,
        count,
        sources,
        mode,
        1,
        True,
    )
    reference = make_vortex_kernel(name)
    displacement = x[:, None, :] - sx[None, :, :]
    expected_u = reference.velocity_pair(displacement, sg[None], core[:, None], sc[None]).sum(
        axis=1
    )
    expected_j = reference.gradient_pair(displacement, sg[None], core[:, None], sc[None]).sum(
        axis=1
    )
    expected_rate = np.einsum("nji,nj->ni", expected_j, gamma)
    np.testing.assert_allclose(velocity.to_numpy(), expected_u, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(gradient.to_numpy(), expected_j, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(rate.to_numpy(), expected_rate, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        rate.to_numpy().sum(axis=0) + exchange.to_numpy().sum(axis=0), 0, atol=1e-11
    )
    before = exchange.to_numpy().copy()
    kernel(
        position,
        strength,
        radius,
        velocity,
        rate,
        gradient,
        sx,
        sg,
        sc,
        owners,
        exchange,
        count,
        sources,
        mode,
        0,
        True,
    )
    np.testing.assert_array_equal(exchange.to_numpy(), before)
    np.testing.assert_array_equal(position.to_numpy(), x)
    np.testing.assert_array_equal(strength.to_numpy(), gamma)
