"""Pure, stage-responsive VLM boundary solves."""

from types import SimpleNamespace

from _flat_plate_geometry import create_flat_plate
import numpy as np
import taichi as ti

from source.solvers.vpm.boundary_elements.vlm.config import VLMSetup, VLMSurfaceSetup
from source.solvers.vpm.boundary_elements.vlm.solver.restart import restart_identity
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver
from source.solvers.vpm.numerics.rk_tableaux import RK2
from source.solvers.vpm.physics.induction.base import StageRates, StageState
from source.solvers.vpm.physics.stage_rhs import VLMStageContribution


def test_responsive_stage_solve_is_temporary_and_uses_stage_boundary_response():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    try:
        geometry = create_flat_plate(
            chord=1.0,
            span=1.0,
            n_chordwise_panels=1,
            n_spanwise_panels=1,
        )
        solver = VLMSolver(
            VLMSetup(
                surfaces=(VLMSurfaceSetup(geometry),),
                dtype="f64",
                boundary_response="responsive",
            )
        )
        solver.generate_mesh()
        solver._current_time = 0.0
        n_panels = solver.lattice.n_panels
        accepted_circulation = solver.lattice.circulation.to_numpy().copy()
        accepted_corners = solver.lattice.panel_corner_position.to_numpy().copy()

        count = 1
        position = ti.Vector.field(3, ti.f64, shape=count)
        strength = ti.Vector.field(3, ti.f64, shape=count)
        core_radius = ti.field(ti.f64, shape=count)
        position.from_numpy(np.array([[0.25, 0.2, 0.2]], dtype=np.float64))
        strength.from_numpy(np.array([[0.1, 0.2, 0.3]], dtype=np.float64))
        core_radius.from_numpy(np.array([0.05], dtype=np.float64))
        velocity = ti.Vector.field(3, ti.f64, shape=count)
        strength_rate = ti.Vector.field(3, ti.f64, shape=count)
        velocity.fill(0.0)
        strength_rate.fill(0.0)

        class _Physics:
            induction = SimpleNamespace(stretching_scheme="transposed")
            accumulator_dtype = ti.f64
            max_n_particles = count
            _zero_velocity = np.zeros(3)

            @staticmethod
            def compute_target_velocity(particles, target_position, include_freestream=True):
                assert len(particles) == count
                assert include_freestream is True
                stage_position = particles.position.to_numpy()[:count]
                stage_strength = particles.vortex_strength.to_numpy()[:count]
                incident = np.array(
                    [
                        0.0,
                        0.0,
                        1.0 + 0.5 * stage_position[:, 0].mean() + stage_strength[:, 1].mean(),
                    ]
                )
                return np.tile(incident, (len(target_position), 1))

        accepted_particles = SimpleNamespace(velocity_background_cpu=lambda: np.zeros(3))
        provider = VLMStageContribution(solver, _Physics(), accepted_particles)
        state = StageState(position, strength, core_radius, count, time=0.0, stage_index=0)
        rates = StageRates(velocity, strength_rate)

        with provider.integration_step(RK2(), 0.02, True):
            provider.add_stage_rates(state, 0.0, rates)
            first_transport = provider.vlm_solver._transported_bound.to_numpy().copy()
            provider.add_stage_rates(state, 0.0, rates)
            np.testing.assert_array_equal(
                provider.vlm_solver._transported_bound.to_numpy(), first_transport
            )
            position.from_numpy(np.array([[0.35, 0.2, 0.2]], dtype=np.float64))
            strength.from_numpy(np.array([[0.1, 0.4, 0.3]], dtype=np.float64))
            stage_one = StageState(
                position,
                strength,
                core_radius,
                count,
                time=0.01,
                stage_index=1,
            )
            provider.add_stage_rates(stage_one, 0.01, rates)
            assert solver._last_stage_near_wake_elapsed == 0.01
            assert solver._last_stage_near_wake_matrix_norm > 0.0
            # Accepted-state health/output refreshes query the field without
            # an RK stage identity.  They must not erase the temporal
            # partial-row evidence from the real stage evaluation.
            elapsed = solver._last_stage_near_wake_elapsed
            matrix_norm = solver._last_stage_near_wake_matrix_norm
            field_only = StageState(position, strength, core_radius, count, time=0.01)
            provider.add_stage_rates(field_only, 0.01, rates)
            assert solver._last_stage_near_wake_elapsed == elapsed
            assert solver._last_stage_near_wake_matrix_norm == matrix_norm

        assert solver._stage_response_active is True
        assert solver._last_stage_boundary_residual < 1.0e-10
        assert np.linalg.norm(solver._stage_circulation.to_numpy()[:n_panels]) > 0.0
        assert not np.allclose(solver._stage_external_velocity.to_numpy()[:n_panels], 0.0)
        np.testing.assert_array_equal(solver.lattice.circulation.to_numpy(), accepted_circulation)
        np.testing.assert_array_equal(
            solver.lattice.panel_corner_position.to_numpy(), accepted_corners
        )
        assert np.linalg.norm(velocity.to_numpy()) > 0.0
    finally:
        ti.reset()


def test_boundary_response_policy_is_part_of_restart_identity():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    try:
        geometry = create_flat_plate(
            chord=1.0,
            span=1.0,
            n_chordwise_panels=1,
            n_spanwise_panels=1,
        )
        lagged = VLMSolver(VLMSetup(surfaces=(VLMSurfaceSetup(geometry),), dtype="f64"))
        responsive = VLMSolver(
            VLMSetup(
                surfaces=(VLMSurfaceSetup(geometry),),
                dtype="f64",
                boundary_response="responsive",
            )
        )
        lagged.generate_mesh()
        responsive.generate_mesh()

        assert restart_identity(lagged) != restart_identity(responsive)
        assert lagged.field_contract.particle_target_radius(0.0) > 0.0
    finally:
        ti.reset()
