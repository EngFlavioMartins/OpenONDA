"""Compact physical contracts for the baseline FVM--VPM exchange."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from source.coupler.boundary import advance_fvm_substeps
from source.coupler.interpolation import FVMVelocityInterpolator
from source.coupler.vorticity_transfer import (
    VorticityTransfer,
    build_transfer_lattice,
    cosine_eta,
    normalized_divergence,
    solenoidal_velocity_correction,
    vortex_strength_from_velocity_trace,
)

BOX = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
PARTICLE_SPACING = 0.1


def _lattice(box=BOX, particle_spacing=PARTICLE_SPACING):
    return build_transfer_lattice(box, particle_spacing)


def _scatter(lattice, result):
    field = np.zeros((len(lattice.position), 3))
    index = {tuple(position): i for i, position in enumerate(lattice.position)}
    for position, vortex_strength in zip(result.position, result.vortex_strength, strict=True):
        field[index[tuple(position)]] = vortex_strength
    return field


def _transfer(velocity_at, *, vpm_only_width=0.0):
    class VelocityTrace:
        @staticmethod
        def sample(position, _velocity, _velocity_gradient):
            return velocity_at(position)

    transfer = object.__new__(VorticityTransfer)
    transfer.step = 0
    transfer.diagnostic_interval = 100
    transfer._box = BOX.copy()
    transfer._lattice = _lattice()
    transfer._velocity_trace = VelocityTrace()
    transfer._cell_centre = np.zeros((1, 3))
    transfer._face_cells = {}
    transfer._body_bounds = None
    transfer._solid_bodies = ()
    transfer.particle_spacing = PARTICLE_SPACING
    transfer.authority_ramp_width = 4.0 * PARTICLE_SPACING
    transfer.vpm_only_width = vpm_only_width
    transfer.core_radius_ratio = 1.0
    transfer.kinematic_viscosity = 1.0e-3
    transfer.last_interface_flow = {}
    transfer.last_vortex_line_closure = {}
    transfer.last_transfer_diagnostics = {}
    return transfer


def test_uniform_velocity_and_solid_body_rotation_have_exact_curl():
    rng = np.random.default_rng(4)
    position = rng.uniform(-1.0, 1.0, (100, 3))
    h = 0.05
    uniform = vortex_strength_from_velocity_trace(
        position,
        h,
        lambda query: np.tile([1.3, -0.4, 0.7], (len(query), 1)),
    )
    np.testing.assert_array_equal(uniform, np.zeros_like(uniform))

    vorticity = np.array([0.3, -0.2, 1.1])
    rotating = vortex_strength_from_velocity_trace(
        position,
        h,
        lambda query: 0.5 * np.cross(vorticity, np.asarray(query)),
    )
    np.testing.assert_allclose(
        rotating,
        np.tile(vorticity * h**3, (len(position), 1)),
        rtol=0.0,
        atol=2.0e-18,
    )


def test_zero_authority_is_an_exact_identity():
    def forbidden(_position):
        raise AssertionError("a velocity evaluator was called where eta is zero")

    result = solenoidal_velocity_correction(
        _lattice(),
        PARTICLE_SPACING,
        fvm_velocity_at=forbidden,
        vpm_velocity_at=forbidden,
        authority_at=lambda position: np.zeros(len(position)),
        core_radius_ratio=1.0,
        n_existing_particles=7,
    )

    assert result.n_added_particles == 0
    assert result.n_updated_particles == 0
    assert result.n_total_particles == 7
    assert result.n_support_nodes == 0


class _ParticleStateVPM:
    np_dtype = np.float64
    particle_kernel = "HIGH_ORDER_GAUSSIAN"

    def __init__(self, position, velocity_at):
        count = len(position)
        self.velocity_at = velocity_at
        self.mutation_notifications = 0
        self.state = {
            "position": position.copy(),
            "velocity": np.arange(3 * count, dtype=float).reshape(count, 3) / 100.0,
            "vortex_strength": np.zeros((count, 3)),
            "core_radius": np.full(count, PARTICLE_SPACING),
            "particle_volume": np.full(count, PARTICLE_SPACING**3),
            "kinematic_viscosity": np.linspace(1.0e-4, 2.0e-4, count),
            "eddy_viscosity": np.linspace(2.0e-5, 3.0e-5, count),
            "effective_viscosity": np.linspace(1.2e-4, 2.3e-4, count),
            "group_id": np.arange(count, dtype=np.int32) % 7,
            "zone_id": np.arange(count, dtype=np.int32) % 5,
            "lineage": np.linspace(0.0, 1.0, count),
        }
        self.particles = SimpleNamespace(
            n_particles_total=count,
            capacity=2 * count,
            position_cpu=lambda: self.state["position"].copy(),
            core_radius_cpu=lambda: self.state["core_radius"].copy(),
        )

    def compute_velocity_at_points(self, position, **_kwargs):
        return self.velocity_at(position)

    def update_particle_vortex_strength(self, mask, increment):
        self.state["vortex_strength"][mask] += increment

    def add_vortex_particles(self, **_fields):
        raise AssertionError("an on-lattice correction unexpectedly added particles")

    def notify_external_particle_mutation(self):
        self.mutation_notifications += 1


def test_active_transfer_preserves_complete_state_where_eta_is_zero():
    def rotating_velocity(position):
        position = np.asarray(position)
        return np.column_stack((-position[:, 1], position[:, 0], np.zeros(len(position))))

    transfer = _transfer(rotating_velocity, vpm_only_width=2.0 * PARTICLE_SPACING)
    lattice = transfer._lattice
    assert lattice is not None
    vpm = _ParticleStateVPM(
        lattice.position,
        lambda position: np.zeros((len(position), 3)),
    )
    before = {name: value.copy() for name, value in vpm.state.items()}
    zero_authority = transfer._identity_authority(lattice.position) == 0.0

    result = transfer.transfer(vpm, np.zeros((1, 3)), np.zeros((1, 3, 3)))

    assert result.n_updated_particles > 0
    assert result.n_added_particles == 0
    assert vpm.mutation_notifications == 0
    np.testing.assert_array_equal(
        vpm.state["vortex_strength"][zero_authority],
        before["vortex_strength"][zero_authority],
    )
    for name in before.keys() - {"vortex_strength"}:
        np.testing.assert_array_equal(vpm.state[name], before[name])


def test_consistent_state_is_a_repeated_fixed_point_with_full_state_preserved():
    def velocity(position):
        position = np.asarray(position)
        return np.column_stack(
            (
                0.2 + 0.3 * position[:, 0] - 0.1 * position[:, 2],
                -0.4 + 0.2 * position[:, 1] + 0.5 * position[:, 2],
                0.1 - 0.3 * position[:, 0] + 0.4 * position[:, 1],
            )
        )

    transfer = _transfer(velocity)
    lattice = transfer._lattice
    assert lattice is not None
    vpm = _ParticleStateVPM(lattice.position, velocity)
    before = {name: value.copy() for name, value in vpm.state.items()}

    for _ in range(50):
        result = transfer.transfer(vpm, np.zeros((1, 3)), np.zeros((1, 3, 3)))
        assert result.n_added_particles == 0
        assert result.n_updated_particles == 0
        assert result.correction_vortex_strength_l1 == 0.0

    assert vpm.mutation_notifications == 0
    for name, value in before.items():
        np.testing.assert_array_equal(vpm.state[name], value)


def test_particle_evaluated_donor_is_idempotent_for_twenty_transfers(tmp_path):
    pytest.importorskip("taichi", reason="VPM requires taichi")
    from source.solvers.vpm import (
        AdvectionConfig,
        StretchingConfig,
        VelocityConfig,
        ViscousConfig,
        VPMSetup,
        VPMSolver,
    )

    vpm = VPMSolver(
        VPMSetup(
            compute_device="CPU",
            velocity=VelocityConfig.direct(),
            stretching=StretchingConfig.disabled(),
            viscous=ViscousConfig(scheme="NONE"),
            advection=AdvectionConfig(scheme="NONE"),
            checkpoint_interval_steps=0,
            logging_interval_steps=0,
            checkpoint_directory=str(tmp_path),
            max_n_particles=2048,
        )
    )
    position = np.array(
        [
            [-0.2, -0.1, 0.0],
            [0.2, -0.1, 0.0],
            [0.2, 0.1, 0.0],
            [-0.2, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    vpm.add_vortex_particles(
        position=position,
        velocity=np.arange(12, dtype=np.float32).reshape(4, 3) * 0.01,
        vortex_strength=np.array(
            [
                [0.0, 0.04, 0.0],
                [0.0, 0.0, 0.04],
                [0.0, -0.04, 0.0],
                [0.0, 0.0, -0.04],
            ],
            dtype=np.float32,
        ),
        core_radius=np.array([0.11, 0.12, 0.13, 0.14], dtype=np.float32),
        particle_volume=np.array([0.008, 0.009, 0.010, 0.011], dtype=np.float32),
        kinematic_viscosity=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * 1.0e-4,
        eddy_viscosity=np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32) * 1.0e-5,
        group_id=np.array([2, 3, 4, 5], dtype=np.int32),
        zone_id=np.array([6, 7, 8, 9], dtype=np.int32),
    )

    def physical_velocity(query):
        return vpm.compute_velocity_at_points(
            query,
            include_freestream=True,
            zone_mask=None,
            include_body=True,
        )

    transfer = _transfer(physical_velocity)
    field_names = (
        "particle_position",
        "particle_velocity",
        "particle_vortex_strength",
        "particle_core_radius",
        "particle_volume",
        "particle_kinematic_viscosity",
        "particle_eddy_viscosity",
        "particle_effective_viscosity",
        "particle_group_id",
        "particle_zone_id",
        "particle_velocity_gradient",
        "particle_strain_rate",
    )
    before = {name: np.asarray(getattr(vpm, name)).copy() for name in field_names}
    mutation_notifications = 0

    def record_mutation():
        nonlocal mutation_notifications
        mutation_notifications += 1

    vpm.notify_external_particle_mutation = record_mutation
    for _ in range(20):
        result = transfer.transfer(
            vpm,
            np.zeros((1, 3)),
            np.zeros((1, 3, 3)),
        )
        assert result.n_added_particles == 0
        assert result.n_updated_particles == 0
        assert result.correction_vortex_strength_l1 == 0.0

    assert mutation_notifications == 0
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(vpm, name), value)


def test_compatible_velocity_curl_is_discretely_solenoidal():
    box = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    h = 0.1
    lattice = _lattice(box, h)

    def velocity(position):
        x, y, z = np.asarray(position).T
        return np.column_stack((-y + 0.2 * y * z, x - 0.1 * x * z, 0.3 * x * y))

    result = solenoidal_velocity_correction(
        lattice,
        h,
        fvm_velocity_at=velocity,
        vpm_velocity_at=lambda position: np.zeros((len(position), 3)),
        authority_at=lambda position: cosine_eta(position, box, 0.3, 0.0),
        core_radius_ratio=1.0,
    )
    l2, linf = normalized_divergence(_scatter(lattice, result), lattice.shape, h)

    assert l2 < 5.0e-14
    assert linf < 5.0e-14


def _quadratic_velocity(position):
    x, y, z = np.asarray(position).T
    velocity = np.column_stack(
        (
            x**2 + 0.3 * y * z,
            y**2 - 0.2 * x * z,
            z**2 + 0.1 * x * y,
        )
    )
    gradient = np.empty((len(position), 3, 3))
    gradient[:, 0, :] = np.column_stack((2.0 * x, -0.2 * z, 0.1 * y))
    gradient[:, 1, :] = np.column_stack((0.3 * z, 2.0 * y, 0.1 * x))
    gradient[:, 2, :] = np.column_stack((0.3 * y, -0.2 * x, 2.0 * z))
    return velocity, gradient


def _graded_interpolation_error(node_count):
    uniform = np.linspace(-1.0, 1.0, node_count)
    graded = np.sign(uniform) * np.abs(uniform) ** 1.5
    mesh = np.meshgrid(graded, uniform, graded, indexing="ij")
    cell_centre = np.column_stack([component.ravel() for component in mesh])
    velocity, gradient = _quadratic_velocity(cell_centre)
    target_axis = np.linspace(-0.78, 0.78, 11)
    target_mesh = np.meshgrid(target_axis, target_axis, target_axis, indexing="ij")
    target = np.column_stack([component.ravel() for component in target_mesh])
    expected, _ = _quadratic_velocity(target)
    sampled = FVMVelocityInterpolator(
        cell_centre,
        cKDTree(cell_centre),
        neighbour_count=4,
    ).sample(target, velocity, gradient)
    return float(np.linalg.norm(sampled - expected) / np.linalg.norm(expected))


def test_fvm_velocity_interpolation_is_affine_exact_and_second_order_on_graded_meshes():
    axis = np.linspace(-1.0, 1.0, 7)
    mesh = np.meshgrid(axis, axis, axis, indexing="ij")
    cell_centre = np.column_stack([component.ravel() for component in mesh])
    gradient_matrix = np.array([[0.2, -0.1, 0.3], [0.4, 0.1, -0.2], [-0.3, 0.2, 0.15]])
    offset = np.array([0.8, -0.2, 0.1])
    velocity = offset + cell_centre @ gradient_matrix
    gradient = np.broadcast_to(gradient_matrix, (len(cell_centre), 3, 3)).copy()
    target = np.array([[-0.73, 0.18, 0.44], [0.02, -0.31, 0.11], [0.81, 0.62, -0.58]])
    interpolator = FVMVelocityInterpolator(
        cell_centre,
        cKDTree(cell_centre),
        neighbour_count=4,
    )
    np.testing.assert_allclose(
        interpolator.sample(target, velocity, gradient),
        offset + target @ gradient_matrix,
        rtol=0.0,
        atol=2.0e-15,
    )

    errors = np.array([_graded_interpolation_error(n) for n in (7, 13, 25)])
    orders = np.log2(errors[:-1] / errors[1:])
    assert np.all(orders > 1.8), (errors, orders)


def test_vortex_crossing_authority_ramp_remains_a_fixed_point():
    box = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    h = 0.1
    lattice = _lattice(box, h)

    for centre_x in np.linspace(-1.1, 0.2, 14):

        def velocity(position, centre=centre_x):
            x, y, z = np.asarray(position).T
            x = x - centre
            envelope = np.exp(-6.0 * (x**2 + y**2 + z**2))
            return np.column_stack((np.zeros(len(position)), -z * envelope, y * envelope))

        result = solenoidal_velocity_correction(
            lattice,
            h,
            fvm_velocity_at=velocity,
            vpm_velocity_at=velocity,
            authority_at=lambda position: cosine_eta(position, box, 0.3, 0.1),
            core_radius_ratio=1.0,
            n_existing_particles=4096,
        )
        assert result.n_total_particles == 4096
        assert result.correction_vortex_strength_l1 == 0.0


def test_vorticity_mixed_substeps_use_both_time_endpoints(monkeypatch):
    coupler = SimpleNamespace(
        n_fvm_substeps=4,
        fvm_time_step_size=0.005,
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        setup=SimpleNamespace(boundary_condition_mode="vorticity_mixed"),
    )
    recorded = []

    def record_step(
        _coupler,
        _patch,
        velocity,
        pressure_gradient=None,
        normal_velocity=None,
        tangential_gradient=None,
    ):
        recorded.append((velocity.copy(), normal_velocity.copy(), tangential_gradient.copy()))

    monkeypatch.setattr("source.coupler.boundary.apply_fvm_boundary", record_step)
    face_centre = np.zeros((3, 3))
    face_normal = np.tile([1.0, 0.0, 0.0], (3, 1))
    face_area = np.ones(3)
    previous_velocity = np.tile([1.0, 0.0, 0.0], (3, 1))
    next_velocity = np.tile([1.0, 0.4, 0.0], (3, 1))
    previous_normal_velocity = np.full(3, 0.2)
    next_normal_velocity = np.full(3, 0.6)
    previous_tangential_gradient = np.zeros((3, 3))
    next_tangential_gradient = np.full((3, 3), 0.8)

    advance_fvm_substeps(
        coupler,
        "numericalBoundary",
        face_centre,
        face_normal,
        face_area,
        previous_velocity,
        next_velocity,
        previous_normal_velocity=previous_normal_velocity,
        next_normal_velocity=next_normal_velocity,
        previous_tangential_gradient=previous_tangential_gradient,
        next_tangential_gradient=next_tangential_gradient,
    )

    for values, alpha in zip(recorded, (0.25, 0.5, 0.75, 1.0), strict=True):
        np.testing.assert_allclose(
            values[0], (1.0 - alpha) * previous_velocity + alpha * next_velocity
        )
        np.testing.assert_allclose(
            values[1],
            (1.0 - alpha) * previous_normal_velocity + alpha * next_normal_velocity,
        )
        np.testing.assert_allclose(values[2], alpha * next_tangential_gradient)
