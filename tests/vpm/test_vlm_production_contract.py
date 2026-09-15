"""Physical backup fields and cross-surface wake response on real VPM states."""

from types import SimpleNamespace

from _flat_plate_geometry import create_flat_plate
import h5py
import numpy as np
import pyvista as pv

import openonda.vpm as vpm
from source.solvers.vpm.coupling.stepper import CouplingStepper
from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.induction.base import StageState
from source.solvers.vpm.physics.stage_rhs import _StageParticleView


def _case(directory, *, responsive=False):
    """Use two small moving wings with separate wake provenance groups."""
    plate = create_flat_plate(
        chord=1,
        span=2,
        angle_of_attack_degrees=6,
        n_chordwise_panels=1,
        n_spanwise_panels=2,
    )
    return vpm.VPMCase(
        directory=directory,
        backup=vpm.Backup(interval_steps=2, directory="solution", log_directory="solution"),
        numerics=vpm.Numerics(
            compute_device="CPU",
            precision="f64",
            time_step_size=0.01,
            max_n_particles=256,
            freestream_velocity=(2.0, 0.0, 0.0),
            induction=vpm.DirectInduction(),
            viscous=vpm.ViscousConfig(scheme="NONE"),
            stabilization=vpm.StabilizationConfig.disabled(),
            vlm=vpm.VLMSetup(
                surfaces=tuple(
                    vpm.VLMSurfaceSetup(
                        plate,
                        name=f"wing_{i}",
                        group_id=i,
                        translation=(1.6 * i, 0.0, 0.15 * i),
                        kinematics=vpm.RotatingVLM(angular_speed=0.3, axis=(1, 0, 0)),
                    )
                    for i in range(2)
                ),
                dtype="f64",
                kinematic_viscosity=0.0,
                wake_core_overlap=2.5,
                boundary_response="responsive" if responsive else "lagged",
                force=vpm.ForceConfig.kutta_joukowski(unsteady=True),
            ),
        ),
    )


def test_coupled_backup_velocity_contains_bound_induction_and_preserves_state(tmp_path):
    """Writing a backup must preserve the complete accepted transport velocity."""
    solver = vpm.VPMSolver(_case(tmp_path))
    try:
        solver.advance(defer_output=True)
        solver.stepper._update_velocity_and_gradients()
        expected_velocity = solver.particles.velocity_cpu(use_cache=False).copy()
        position = solver.particle_position.copy()
        strength = solver.particle_vortex_strength.copy()
        circulation = solver.vlm_solver.lattice.circulation.to_numpy().copy()
        # Poison same-revision diagnostic caches: serialization must use the
        # freshly evaluated device fields, regardless of prior reads.
        solver.particles.velocity.fill(-999.0)
        solver.particles.velocity_cpu(use_cache=False)
        solver.particles.vorticity.fill(-999.0)
        solver.particles.vorticity_cpu(use_cache=False)
        solver.save_backup()
        with h5py.File(tmp_path / "solution/vpm_000001.h5") as archive:
            np.testing.assert_allclose(
                archive["particles/velocity"][:],
                expected_velocity,
                rtol=2e-12,
                atol=2e-12,
            )
            np.testing.assert_array_equal(archive["particles/position"][:], position)
            np.testing.assert_array_equal(archive["particles/vortex_strength"][:], strength)
            np.testing.assert_array_equal(archive["solver/vlm/circulation"][:], circulation)
            np.testing.assert_allclose(
                archive["particles/vorticity"][:],
                solver.particles.vorticity_cpu(use_cache=False),
            )
        np.testing.assert_allclose(
            solver.particles.velocity_cpu(use_cache=False),
            expected_velocity,
            rtol=2e-12,
            atol=2e-12,
        )
    finally:
        solver.close()


def test_empty_wake_samplers_include_freestream_and_solved_bound_field(tmp_path):
    """A zero particle count does not imply a zero physical velocity field."""
    solver = vpm.VPMSolver(_case(tmp_path))
    try:
        samplers = (
            vpm.LineSampler([-1, 0, 1], [3, 0, 1], 0.5),
            vpm.SurfaceSampler(point=[1, 0, 1], normal=[0, 0, 1], bounds=[-1, 1, -1, 1], spacing=1),
        )
        for sampler in samplers:
            result = sampler.sample(solver)
            np.testing.assert_allclose(result["velocity_x"], 2.0)
            np.testing.assert_allclose(result["velocity_y"], 0.0)
            np.testing.assert_allclose(result["velocity_z"], 0.0)
        solver.vlm_solver.solve(np.array([2.0, 0.0, 0.0]))
        assert len(solver.particles) == 0
        for sampler in samplers:
            result = sampler.sample(solver)
            positions = np.column_stack([result[f"position_{axis}"] for axis in "xyz"])
            velocity = np.column_stack([result[f"velocity_{axis}"] for axis in "xyz"])
            expected = solver.compute_velocity_at_points(positions)
            np.testing.assert_allclose(velocity, expected, atol=1e-12)
            assert np.max(np.abs(velocity[:, 2])) > 1e-3
    finally:
        solver.close()


def test_periodic_and_final_backups_pair_exact_moving_surface_states(tmp_path):
    """The VPM clock alone selects HDF5 and matching physical VLM companions."""
    solver = vpm.VPMSolver(_case(tmp_path))
    try:
        for _ in range(3):
            solver.advance(defer_output=True)
            solver.execute_scheduled_samplers()
        solver.save_backup()
        solution = tmp_path / "solution"
        assert sorted(p.stem for p in solution.glob("vpm_*.h5")) == ["vpm_000002", "vpm_000003"]
        assert sorted(p.stem for p in solution.glob("vlm_*.vtp")) == ["vlm_000002", "vlm_000003"]
        for step in (2, 3):
            surface = pv.read(solution / f"vlm_{step:06d}.vtp")
            with h5py.File(solution / f"vpm_{step:06d}.h5") as archive:
                assert surface.field_data["TimeValue"][0] == archive["solver"].attrs["time"]
                np.testing.assert_array_equal(
                    surface.points,
                    archive["solver/vlm/panel_corner_position"][:].reshape(-1, 3),
                )
                for field in ("circulation", "normal", "panel_force", "bound_vortex_velocity"):
                    np.testing.assert_array_equal(surface[field], archive[f"solver/vlm/{field}"][:])
        assert set(solver.particle_group_id) == {0, 1}
        # Every surface receives induction from the entire currently emitted wake.
        expected = solver.physics.compute_target_velocity(
            solver.particles,
            solver.vlm_solver.lattice.get_collocation_points(),
        )
        np.testing.assert_allclose(
            solver.vlm_solver.lattice.external_velocity.to_numpy(),
            expected,
            rtol=2e-12,
            atol=2e-12,
        )
        # Independent host Biot-Savart evaluation of the accepted default
        # (lagged) solve, with a nonzero contribution from wing 0 onto wing 1.
        positions = solver.particle_position
        strength = solver.particle_vortex_strength
        radii = solver.particles.core_radius_cpu(use_cache=False)
        targets = solver.vlm_solver.lattice.get_collocation_points()
        pairs = make_vortex_kernel("GAUSSIAN").velocity_pair(
            targets[:, None, :] - positions[None, :, :],
            strength[None, :, :],
            radii[None, :],
            radii[None, :],
        )
        np.testing.assert_allclose(expected, pairs.sum(axis=1) + [2.0, 0.0, 0.0], atol=2e-12)
        assert (
            np.linalg.norm(pairs[len(targets) // 2 :, solver.particle_group_id == 0].sum(axis=1))
            > 1e-5
        )
    finally:
        solver.close()


def test_other_surface_wake_changes_receiving_circulation_without_group_filter(tmp_path):
    """Real particle induction reaches another surface irrespective of source tags."""
    solver = vpm.VPMSolver(_case(tmp_path, responsive=True))
    try:
        solver.add_vortex_particles(
            position=np.array([[1.4, 0.25, 0.5]]),
            velocity=np.zeros((1, 3)),
            vortex_strength=np.array([[0.0, 0.3, 0.1]]),
            core_radius=np.array([0.15]),
            particle_volume=np.array([0.15**3]),
            kinematic_viscosity=np.zeros(1),
            group_id=np.zeros(1, dtype=np.int32),
        )
        vlm = solver.vlm_solver
        state = StageState(
            solver.particles.position,
            solver.particles.vortex_strength,
            solver.particles.core_radius,
            1,
            time=0.0,
            stage_index=0,
        )

        def response():
            view = _StageParticleView(state, solver.particles, solver.physics)
            vlm.solve_stage_boundary(state, 0.0, solver.physics, view)
            return vlm._stage_circulation.to_numpy().copy()

        with_wake = response()
        groups = solver.particles.group_id.to_numpy()
        groups[0] = 1
        solver.particles.group_id.from_numpy(groups)
        np.testing.assert_array_equal(response(), with_wake)
        solver.particles.vortex_strength.fill(0.0)
        solver.particles.touch_state()
        without_wake = response()
        receiving = slice(vlm.lattice.n_panels // 2, vlm.lattice.n_panels)
        assert np.linalg.norm((with_wake - without_wake)[receiving]) > 1e-3
        matrix = vlm._stage_influence.to_numpy()
        assert np.linalg.norm(matrix[: receiving.start, receiving]) > 1e-6
        assert np.linalg.norm(matrix[receiving, : receiving.start]) > 1e-6
    finally:
        solver.close()


def test_coupling_stepper_runs_coupling_when_wake_release_is_disabled():
    calls = []

    def fake_advance_coupled(particles, physics, config, time_step_size, step, time, release_wake):
        calls.append((time_step_size, release_wake))
        if not release_wake:
            return None
        return {
            "_gpu_transfer_ready": True,
            "vertex_position": np.zeros((0, 3)),
            "velocity": np.zeros((0, 3)),
            "vortex_strength": np.zeros((0, 3)),
            "core_radius": np.zeros(0),
            "particle_volume": np.zeros(0),
        }

    added = []

    solver = SimpleNamespace(
        vlm_solver=SimpleNamespace(advance_coupled=fake_advance_coupled),
        particles=SimpleNamespace(),
        physics=SimpleNamespace(),
        setup=SimpleNamespace(),
        stepper=SimpleNamespace(time=0.25, step=3),
        _release_wake_particles=False,
        _release_interval=0.4,
        time_step_size=0.5,
        add_vortex_particles=lambda **kwargs: added.append(kwargs),
    )

    stepper = CouplingStepper(solver)
    stepper.advance_vlm(0.2)

    assert calls == [(0.4, False)]
    assert added == []

    solver._release_wake_particles = True
    stepper.advance_vlm(0.2)

    assert calls[-1] == (0.4, True)
    assert len(added) == 1
