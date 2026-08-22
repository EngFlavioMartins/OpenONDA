"""Fundamental properties required of the baseline FVM--VPM handoff."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler.vorticity_transfer import (
    TransferResult,
    VorticityTransfer,
    build_transfer_lattice,
    coalesce_lattice_corrections,
    cosine_eta,
    normalized_divergence,
    solenoidal_velocity_correction,
)
from source.solvers.VPM import VPMSetup, VPMSolver
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)
from source.solvers.VPM.runtime.backend import reset_taichi_backend

BOX = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
H = 0.1


def _lattice(box=BOX, h=H):
    return build_transfer_lattice(
        box,
        h,
    )


def _authority(points, box=BOX, h=H):
    return cosine_eta(points, box, 3.0 * h, 0.0)


def _scatter_result(lattice, result):
    field = np.zeros((len(lattice.positions), 3))
    lookup = {tuple(position): index for index, position in enumerate(lattice.positions)}
    for position, circulation in zip(result.pos, result.circ, strict=True):
        field[lookup[tuple(position)]] = circulation
    return field


def test_transfer_setup_worker_participates_in_collective_wall_query():
    """A worker must not leave setup before rank zero gathers wall faces."""

    class WorkerFVM:
        setup = SimpleNamespace(boundaries=[SimpleNamespace(name="cube", mesh_type="wall")])
        ibm = None

        def __init__(self):
            self.calls = []

        def get_cell_centre_coordinates(self):
            self.calls.append("cells")
            return np.empty((0, 3))

        def get_boundary_face_centre_coordinates(self, patch):
            self.calls.append(f"wall:{patch}")
            return np.empty((0, 3))

    coupler = SimpleNamespace(
        setup=SimpleNamespace(
            transfer_region_bounds=None,
            vpm_particle_spacing=H,
            authority_ramp_width=3.0 * H,
            vpm_only_width=0.0,
            vpm_core_radius_ratio=1.0,
            transfer_diagnostic_interval_steps=1,
        ),
        kinematic_viscosity=1.0e-3,
        vpm_time_step_size=0.01,
        fvm_box=BOX,
    )
    fvm = WorkerFVM()

    transfer = VorticityTransfer(coupler)
    transfer.setup(fvm)

    assert fvm.calls == ["cells", "wall:cube"]
    assert transfer._lattice is None


def test_eta_zero_is_exact_identity_inside_stencil_support():
    """A zero-authority stencil must not even evaluate or add a correction."""
    lattice = _lattice()

    def forbidden(_points):
        raise AssertionError("velocity was evaluated where authority is zero")

    result = solenoidal_velocity_correction(
        lattice,
        H,
        fvm_velocity_at=forbidden,
        vpm_velocity_at=forbidden,
        authority_at=lambda points: np.zeros(len(points)),
        core_radius_ratio=1.0,
        n_existing=3,
    )

    assert result.n_added == 0
    assert result.n_total == 3
    assert result.n_support == 0


def test_eta_zero_dead_zone_is_identity_next_to_active_curl_stencil():
    """A curl guard must not turn a stencil buffer into mutation support."""
    lattice = _lattice()
    dead_zone = 2.0 * H

    def identity_authority(points):
        return cosine_eta(points, BOX, 4.0 * H, dead_zone)

    def velocity_authority(points):
        return cosine_eta(points, BOX, 5.0 * H, dead_zone + H)

    def solid_body_velocity(points):
        points = np.asarray(points)
        return np.column_stack((-points[:, 1], points[:, 0], np.zeros(len(points))))

    result = solenoidal_velocity_correction(
        lattice,
        H,
        fvm_velocity_at=solid_body_velocity,
        vpm_velocity_at=lambda points: np.zeros((len(points), 3)),
        authority_at=velocity_authority,
        identity_authority_at=identity_authority,
        core_radius_ratio=1.0,
        blob_second_moment=0.0,
    )

    assert result.n_added > 0
    assert np.all(identity_authority(result.pos) > 0.0)
    field = _scatter_result(lattice, result)
    l2, linf = normalized_divergence(field, lattice.shape, H)
    assert l2 < 5.0e-14
    assert linf < 5.0e-14


def test_eta_zero_particle_is_not_selected_for_in_place_correction():
    lattice = _lattice()
    dead_zone = 2.0 * H

    def identity_authority(points):
        return cosine_eta(points, BOX, 4.0 * H, dead_zone)

    def velocity_authority(points):
        return cosine_eta(points, BOX, 5.0 * H, dead_zone + H)

    def velocity(points):
        points = np.asarray(points)
        return np.column_stack((-points[:, 1], points[:, 0], np.zeros(len(points))))

    result = solenoidal_velocity_correction(
        lattice,
        H,
        fvm_velocity_at=velocity,
        vpm_velocity_at=lambda points: np.zeros((len(points), 3)),
        authority_at=velocity_authority,
        identity_authority_at=identity_authority,
        core_radius_ratio=1.0,
        blob_second_moment=0.0,
        n_existing=2,
    )
    eta_zero_position = lattice.positions[identity_authority(lattice.positions) == 0.0][0]
    active_position = result.pos[0]
    coalesced = coalesce_lattice_corrections(
        result,
        np.vstack((eta_zero_position, active_position)),
        np.full(2, H),
        lattice,
        H,
        H,
    )

    np.testing.assert_array_equal(coalesced.updated_indices, np.array([1]))


def test_consistent_velocity_field_is_a_fixed_point_for_repeated_transfer():
    lattice = _lattice()

    def velocity(points):
        points = np.asarray(points)
        return np.column_stack(
            [
                0.2 + 0.3 * points[:, 0] - 0.1 * points[:, 2],
                -0.4 + 0.2 * points[:, 1] + 0.5 * points[:, 2],
                0.1 - 0.3 * points[:, 0] + 0.4 * points[:, 1],
            ]
        )

    for _ in range(50):
        result = solenoidal_velocity_correction(
            lattice,
            H,
            fvm_velocity_at=velocity,
            vpm_velocity_at=velocity,
            authority_at=_authority,
            core_radius_ratio=1.0,
            n_existing=7,
        )
        assert result.n_added == 0
        assert result.n_total == 7
        assert result.correction_vortex_strength_l1 == 0.0


def test_velocity_curl_correction_is_discretely_solenoidal():
    box = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    h = 0.1
    lattice = _lattice(box, h)

    def fvm_velocity(points):
        points = np.asarray(points)
        return np.column_stack(
            [
                -points[:, 1] + 0.2 * points[:, 1] * points[:, 2],
                points[:, 0] - 0.1 * points[:, 0] * points[:, 2],
                0.3 * points[:, 0] * points[:, 1],
            ]
        )

    result = solenoidal_velocity_correction(
        lattice,
        h,
        fvm_velocity_at=fvm_velocity,
        vpm_velocity_at=lambda points: np.zeros((len(points), 3)),
        authority_at=lambda points: cosine_eta(points, box, 0.3, 0.0),
        core_radius_ratio=1.0,
    )
    field = _scatter_result(lattice, result)
    l2, linf = normalized_divergence(field, lattice.shape, h)

    assert l2 < 5.0e-14
    assert linf < 5.0e-14


def test_wall_aligned_curl_excludes_solid_without_losing_solenoidality():
    box = np.array([-1.5, 1.5] * 3)
    body = np.array([-0.5, 0.5] * 3)
    h = 0.25

    def open_interior(points):
        points = np.asarray(points)
        return np.all((points > body[::2]) & (points < body[1::2]), axis=1)

    lattice = build_transfer_lattice(
        box,
        h,
        lattice_anchor=body[::2],
        interior_at_node=open_interior,
    )

    def authority(points):
        points = np.asarray(points)
        value = cosine_eta(points, box, 0.5, 0.0)
        value[np.all((points >= body[::2]) & (points <= body[1::2]), axis=1)] = 0.0
        return value

    def velocity(points):
        x, y, z = np.asarray(points).T
        return np.column_stack((-y + 0.2 * y * z, x - 0.1 * x * z, 0.3 * x * y))

    result = solenoidal_velocity_correction(
        lattice,
        h,
        fvm_velocity_at=velocity,
        vpm_velocity_at=lambda points: np.zeros((len(points), 3)),
        authority_at=authority,
        identity_authority_at=lambda points: cosine_eta(points, box, 0.5, 0.0),
        core_radius_ratio=1.0,
    )

    assert not open_interior(result.pos).any()
    assert result.divergence_correction_l2 < 5.0e-14
    assert result.divergence_correction_linf < 5.0e-14


def test_lattice_coalescing_changes_only_coincident_particle_strengths():
    lattice = _lattice()
    position = lattice.positions[[0, 1, 2]]
    result = TransferResult(
        pos=position.copy(),
        circ=np.arange(9, dtype=float).reshape(3, 3) / 100.0,
        vol=np.full(3, H**3),
        rad=np.full(3, H),
        n_existing=4,
    )
    existing_position = np.vstack((position[:2], [2.0, 2.0, 2.0], position[2]))
    existing_radius = np.array([H, H, H, 1.5 * H])

    coalesce_lattice_corrections(
        result,
        existing_position,
        existing_radius,
        lattice,
        H,
        H,
    )

    np.testing.assert_array_equal(result.updated_indices, [0, 1])
    np.testing.assert_array_equal(
        result.updated_circulation,
        np.arange(6, dtype=float).reshape(2, 3) / 100.0,
    )
    np.testing.assert_array_equal(result.pos, position[2:])
    assert result.n_updated == 2
    assert result.n_added == 1
    assert result.n_total == 5


def test_production_spacing_blob_moment_map_reduces_velocity_error(tmp_path):
    """Exercise the Gaussian realization at cubeFlow's production spacing."""
    reset_taichi_backend()
    h = 0.03125
    box = np.array([-0.25, 0.25] * 3)
    lattice = build_transfer_lattice(box, h)

    def make_solver(max_particles):
        return VPMSolver(
            VPMSetup(
                compute_device="CPU",
                particle_kernel="GAUSSIAN",
                velocity=VelocityConfig.direct(),
                stretching=StretchingConfig.disabled(),
                viscous=ViscousConfig(scheme="NONE"),
                advection=AdvectionConfig(scheme="NONE"),
                checkpoint_interval_steps=0,
                logging_interval_steps=0,
                checkpoint_directory=str(tmp_path),
                max_particles=max_particles,
            )
        )

    try:
        donor = make_solver(16)
        donor.add_vortex_particles(
            position=np.zeros((1, 3), dtype=np.float32),
            velocity=np.zeros((1, 3), dtype=np.float32),
            vortex_strength=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            core_radius=np.array([2.0 * h], dtype=np.float32),
            volume=np.array([h**3], dtype=np.float32),
            kinematic_viscosity=np.zeros(1, dtype=np.float32),
        )

        def donor_velocity(points):
            return donor.compute_target_velocities(points, include_freestream=False)

        result = {}
        for name, second_moment in (("raw", 0.0), ("compensated", 1.5)):
            transfer = solenoidal_velocity_correction(
                lattice,
                h,
                fvm_velocity_at=donor_velocity,
                vpm_velocity_at=lambda points: np.zeros((len(points), 3)),
                authority_at=lambda points: np.ones(len(points)),
                core_radius_ratio=1.05,
                blob_second_moment=second_moment,
            )
            receiver = make_solver(len(lattice.positions) + 16)
            receiver.add_vortex_particles(
                position=transfer.pos.astype(np.float32),
                velocity=np.zeros((transfer.n_added, 3), dtype=np.float32),
                vortex_strength=transfer.circ.astype(np.float32),
                core_radius=transfer.rad.astype(np.float32),
                volume=transfer.vol.astype(np.float32),
                kinematic_viscosity=np.zeros(transfer.n_added, dtype=np.float32),
            )
            result[name] = receiver.compute_target_velocities(
                lattice.positions, include_freestream=False
            )

        target = donor_velocity(lattice.positions)
        central = np.all(np.abs(lattice.positions) <= 0.125 + 1.0e-12, axis=1)
        errors = {
            name: np.linalg.norm(values[central] - target[central])
            / np.linalg.norm(target[central])
            for name, values in result.items()
        }
        assert errors["compensated"] < 0.55 * errors["raw"], errors
        assert errors["compensated"] < 0.11, errors
    finally:
        reset_taichi_backend()


def test_fixed_point_preserves_complete_existing_particle_state():
    """The coupler may append a real correction but must never rebuild a fixed point."""
    lattice = _lattice()

    def velocity(points):
        points = np.asarray(points)
        return np.column_stack((-points[:, 1], points[:, 0], np.zeros(len(points))))

    class ExactTrace:
        @staticmethod
        def sample(points, _velocity, _gradient):
            return velocity(points)

    class FakeVPM:
        np_dtype = np.float64

        def __init__(self):
            self.particles = SimpleNamespace(n_particles=3, capacity=32)
            self.state = {
                "position": np.arange(9, dtype=float).reshape(3, 3) / 10.0,
                "velocity": np.arange(9, 18, dtype=float).reshape(3, 3) / 10.0,
                "vortex_strength": np.arange(18, 27, dtype=float).reshape(3, 3) / 100.0,
                "core_radius": np.array([0.11, 0.12, 0.13]),
                "volume": np.array([0.01, 0.02, 0.03]),
                "kinematic_viscosity": np.array([1.0e-3, 2.0e-3, 3.0e-3]),
                "eddy_viscosity": np.array([4.0e-3, 5.0e-3, 6.0e-3]),
                "effective_viscosity": np.array([5.0e-3, 7.0e-3, 9.0e-3]),
                "group_id": np.array([2, 4, 6], dtype=np.int32),
                "zone_id": np.array([1, 3, 5], dtype=np.int32),
                "velocity_gradient": np.arange(27, dtype=float).reshape(3, 3, 3),
                "strain_rate": np.arange(27, 54, dtype=float).reshape(3, 3, 3),
                "lineage": np.array([0.7, 0.8, 0.9]),
            }

        @staticmethod
        def compute_target_velocities(points, **_kwargs):
            return velocity(points)

        def add_vortex_particles(self, **_kwargs):
            raise AssertionError("fixed-point transfer attempted to mutate the cloud")

    transfer = object.__new__(VorticityTransfer)
    transfer.step = 0
    transfer.diagnostic_interval = 2
    transfer._box = BOX.copy()
    transfer._lattice = lattice
    transfer._velocity_trace = ExactTrace()
    transfer._cell_centers = np.array([[0.0, 0.0, 0.0]])
    transfer._face_cells = {}
    transfer._body_bounds = None
    transfer._solid_bodies = ()
    transfer.particle_spacing = H
    transfer.authority_ramp_width = 3.0 * H
    transfer.vpm_only_width = 0.0
    transfer.core_radius_ratio = 1.0
    transfer.kinematic_viscosity = 1.0e-3
    transfer.last_interface_flow = {}
    transfer.last_vortex_line_closure = {}
    transfer.last_transfer_diagnostics = {}
    vpm = FakeVPM()
    before = {name: values.copy() for name, values in vpm.state.items()}

    result = transfer.transfer(vpm, np.zeros((1, 3)), np.zeros((1, 3, 3)))

    assert result.n_added == 0
    assert result.n_total == 3
    for name, values in before.items():
        np.testing.assert_array_equal(vpm.state[name], values)


def test_actual_vpm_manufactured_donor_is_idempotent_for_20_transfers(tmp_path):
    """Use the physical field evaluated from a real particle cloud as donor data."""
    reset_taichi_backend()
    try:
        vpm = VPMSolver(
            VPMSetup(
                compute_device="CPU",
                stretching=StretchingConfig.disabled(),
                viscous=ViscousConfig(scheme="NONE"),
                advection=AdvectionConfig(scheme="NONE"),
                checkpoint_interval_steps=0,
                logging_interval_steps=0,
                checkpoint_directory=str(tmp_path),
                max_particles=2048,
            )
        )
        position = np.array(
            [[-0.2, -0.1, 0.0], [0.2, -0.1, 0.0], [0.2, 0.1, 0.0], [-0.2, 0.1, 0.0]],
            dtype=np.float32,
        )
        strength = np.array(
            [[0.0, 0.04, 0.0], [0.0, 0.0, 0.04], [0.0, -0.04, 0.0], [0.0, 0.0, -0.04]],
            dtype=np.float32,
        )
        vpm.add_vortex_particles(
            position=position,
            velocity=np.arange(12, dtype=np.float32).reshape(4, 3) * 0.01,
            vortex_strength=strength,
            core_radius=np.array([0.11, 0.12, 0.13, 0.14], dtype=np.float32),
            volume=np.array([0.008, 0.009, 0.010, 0.011], dtype=np.float32),
            kinematic_viscosity=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * 1e-4,
            eddy_viscosity=np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32) * 1e-5,
            group_id=np.array([2, 3, 4, 5], dtype=np.int32),
            zone_id=np.array([6, 7, 8, 9], dtype=np.int32),
        )

        class ExactPhysicalTrace:
            @staticmethod
            def sample(points, _velocity, _gradient):
                return vpm.compute_target_velocities(
                    points, include_freestream=True, zone_mask=None, include_body=True
                )

        transfer = object.__new__(VorticityTransfer)
        transfer.step = 0
        transfer.diagnostic_interval = 100
        transfer._box = BOX.copy()
        transfer._lattice = _lattice()
        transfer._velocity_trace = ExactPhysicalTrace()
        transfer._cell_centers = np.array([[0.0, 0.0, 0.0]])
        transfer._face_cells = {}
        transfer._body_bounds = None
        transfer._solid_bodies = ()
        transfer.particle_spacing = H
        transfer.authority_ramp_width = 3.0 * H
        transfer.vpm_only_width = 0.0
        transfer.core_radius_ratio = 1.0
        transfer.kinematic_viscosity = 1.0e-3
        transfer.last_interface_flow = {}
        transfer.last_vortex_line_closure = {}
        transfer.last_transfer_diagnostics = {}
        fields = (
            "particles_positions",
            "particles_velocities",
            "particle_vortex_strength",
            "particle_core_radius",
            "particles_volumes",
            "particles_viscosities",
            "particles_viscosities_t",
            "particles_viscosities_eff",
            "particles_group_ids",
            "particles_zone_ids",
            "particles_velocity_gradients",
            "particles_strain_rate_tensors",
        )
        before = {name: np.asarray(getattr(vpm, name)).copy() for name in fields}
        mutation_notifications = 0

        def notify_external_particle_mutation():
            nonlocal mutation_notifications
            mutation_notifications += 1

        vpm.notify_external_particle_mutation = notify_external_particle_mutation

        for _ in range(20):
            result = transfer.transfer(vpm, np.zeros((1, 3)), np.zeros((1, 3, 3)))
            assert result.n_added == 0
            assert result.n_total == 4
        assert mutation_notifications == 0

        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(vpm, name), expected)

        class PhysicalCorrectionTrace:
            @staticmethod
            def sample(points, _velocity, _gradient):
                points = np.asarray(points)
                represented = vpm.compute_target_velocities(
                    points, include_freestream=True, zone_mask=None, include_body=True
                )
                return represented + np.column_stack(
                    (-points[:, 1], points[:, 0], np.zeros(len(points)))
                )

        transfer._velocity_trace = PhysicalCorrectionTrace()
        correction = transfer.transfer(vpm, np.zeros((1, 3)), np.zeros((1, 3, 3)))
        assert correction.n_added > 0
        assert mutation_notifications == 1
        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(vpm, name)[:4], expected)
    finally:
        reset_taichi_backend()


def test_capacity_exhaustion_fails_without_deleting_existing_particles():
    lattice = _lattice()

    class RotatingTrace:
        @staticmethod
        def sample(points, _velocity, _gradient):
            points = np.asarray(points)
            return np.column_stack((-points[:, 1], points[:, 0], np.zeros(len(points))))

    class CapacityLimitedVPM:
        np_dtype = np.float64

        def __init__(self):
            self.particles = SimpleNamespace(n_particles=2, capacity=2)
            self.sentinel = np.array([3.0, 4.0])

        @staticmethod
        def compute_target_velocities(points, **_kwargs):
            return np.zeros((len(points), 3))

        def add_vortex_particles(self, **_kwargs):
            raise AssertionError("capacity preflight did not run")

    transfer = object.__new__(VorticityTransfer)
    transfer.step = 0
    transfer.diagnostic_interval = 2
    transfer._box = BOX.copy()
    transfer._lattice = lattice
    transfer._velocity_trace = RotatingTrace()
    transfer._cell_centers = np.array([[0.0, 0.0, 0.0]])
    transfer._face_cells = {}
    transfer._body_bounds = None
    transfer._solid_bodies = ()
    transfer.particle_spacing = H
    transfer.authority_ramp_width = 3.0 * H
    transfer.vpm_only_width = 0.0
    transfer.core_radius_ratio = 1.0
    transfer.kinematic_viscosity = 1.0e-3
    transfer.last_interface_flow = {}
    transfer.last_vortex_line_closure = {}
    transfer.last_transfer_diagnostics = {}
    vpm = CapacityLimitedVPM()
    before = vpm.sentinel.copy()

    with pytest.raises(RuntimeError, match="will not delete wake particles"):
        transfer.transfer(vpm, np.zeros((1, 3)), np.zeros((1, 3, 3)))

    assert vpm.particles.n_particles == 2
    np.testing.assert_array_equal(vpm.sentinel, before)
