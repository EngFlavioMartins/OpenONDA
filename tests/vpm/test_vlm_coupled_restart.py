"""VLM/VPM continuation and complete velocity sampling on real coupled states."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial import cKDTree

import openonda.vpm as vpm
from tutorials.vpm.flat_plate.assets.generate_surface import create_flat_plate


def test_native_metadata_preserves_loaded_geometry_when_the_input_file_is_removed(tmp_path):
    from openonda.plotting import read_vlm_surface
    from source.solvers.vpm.boundary_elements.vlm.geometry.surface_io import (
        save_surface,
        surface_to_dict,
    )
    from source.solvers.vpm.io.manifest import build_manifest

    plate = create_flat_plate(
        chord=1,
        span=4,
        angle_of_attack_degrees=5,
        n_chordwise_panels=2,
        n_spanwise_panels=2,
    )
    path = tmp_path / "plate.json"
    save_surface(plate, str(path))
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path / "run",
            numerics=vpm.Numerics(
                compute_device="CPU",
                max_n_particles=32,
                viscous=vpm.ViscousConfig.cs(kinematic_viscosity=0.01),
                vlm=vpm.VLMSetup(
                    surfaces=(vpm.VLMSurfaceSetup(str(path)),),
                    kinematic_viscosity=0.01,
                ),
            ),
        ),
    )
    try:
        path.unlink()
        record = build_manifest(solver)["configuration"]["numerics"]["vlm"]["surfaces"][0]
        assert record["geometry"] == surface_to_dict(plate)
        assert read_vlm_surface(record, tmp_path) == surface_to_dict(plate)
    finally:
        solver.close()


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
@pytest.mark.parametrize("scale", [1.0, 0.02])
@pytest.mark.parametrize("core_overlap", [None, 2.5])
def test_newborn_wake_satisfies_the_solved_boundary_condition(
    tmp_path, kernel, scale, core_overlap
):
    """Accepted particle induction must match the wake used in the VLM matrix."""
    plate = create_flat_plate(
        chord=scale,
        span=4 * scale,
        angle_of_attack_degrees=8,
        n_chordwise_panels=2,
        n_spanwise_panels=3,
    )
    viscosity = 0.01 * scale
    setup = vpm.VLMSetup(
        surfaces=(
            vpm.VLMSurfaceSetup(
                plate, kinematics=vpm.RotatingVLM(angular_speed=0.4 / scale, axis=[0, 0, 1])
            ),
        ),
        freestream_velocity=(10.0, 0.0, 0.0),
        dtype="f64",
        kinematic_viscosity=viscosity,
        wake_core_overlap=core_overlap,
    )
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path / f"{kernel}-{scale}",
            numerics=vpm.Numerics(
                vlm=setup,
                freestream_velocity=[10.0, 0.0, 0.0],
                time_step_size=0.01 * scale,
                compute_device="CPU",
                precision="f64",
                max_n_particles=256,
                particle_kernel=kernel,
                induction=vpm.DirectInduction(),
                viscous=vpm.ViscousConfig.cs(kinematic_viscosity=viscosity),
            ),
        )
    )
    try:
        for _ in range(4):
            solver.advance(defer_output=True)
            lattice = solver.vlm_solver.lattice
            velocity = lattice.get_velocity() - lattice.get_kinematic_velocity()
            normals = lattice.normal.to_numpy()[: lattice.n_panels]
            residual = np.einsum("ij,ij->i", velocity, normals)
            # The device Gaussian uses the A&S erf approximation; its host
            # native-kernel reference uses math.erf. Winckelmans is algebraic.
            tolerance = 2e-6 if kernel == "GAUSSIAN" else 2e-9
            np.testing.assert_allclose(residual, 0.0, atol=tolerance)
    finally:
        solver.close()


@pytest.mark.parametrize("core_overlap", [None, 2.5])
@pytest.mark.parametrize("restart_capacity", [256, 512])
def test_coupled_checkpoint_continues_particles_motion_and_sampled_velocity(
    tmp_path, core_overlap, restart_capacity
):
    plate = create_flat_plate(
        chord=1, span=4, angle_of_attack_degrees=5, n_chordwise_panels=2, n_spanwise_panels=3
    )
    vlm = vpm.VLMSetup(
        surfaces=(
            vpm.VLMSurfaceSetup(
                plate, kinematics=vpm.TranslatingVLM(velocity=np.array([-10.0, 0.0, 0.0]))
            ),
        ),
        freestream_velocity=(10.0, 0.0, 0.0),
        dtype="f64",
        kinematic_viscosity=0.01,
        wake_core_overlap=core_overlap,
        force=vpm.ForceConfig.kutta_joukowski(unsteady=True),
    )
    case = vpm.VPMCase(
        directory=tmp_path / "first",
        numerics=vpm.Numerics(
            vlm=vlm,
            time_step_size=0.01,
            compute_device="CPU",
            precision="f64",
            max_n_particles=256,
            induction=vpm.DirectInduction(),
            viscous=vpm.ViscousConfig.cs(kinematic_viscosity=0.01),
            stabilization=vpm.StabilizationConfig.disabled(),
        ),
    )
    original = vpm.VPMSolver(case)
    try:
        original.advance(defer_output=True)
        original.advance(defer_output=True)
        original.save_backup()
        saved_forces = original.vlm_solver.compute_forces(1.3)
        checkpoint = tmp_path / "first/solution/vpm_000002.h5"
        original.advance(defer_output=True)
        position = original.particle_position.copy()
        strength = original.particle_vortex_strength.copy()
        circulation = original.vlm_solver.lattice.get_circulation().copy()
        geometry = original.vlm_solver.lattice.get_collocation_points().copy()
        next_forces = original.vlm_solver.compute_forces(1.3)
        next_moments = original.vlm_solver.lattice.panel_moment_correction.to_numpy()[:12].copy()
    finally:
        original.close()
    resumed = vpm.VPMSolver(
        replace(
            case,
            directory=tmp_path / "resumed",
            numerics=replace(case.numerics, max_n_particles=restart_capacity),
        )
    )
    try:
        resumed.load_backup(checkpoint)
        restored_forces = resumed.vlm_solver.compute_forces(1.3)
        for key in ("force_z", "moment_y", "unsteady_force_z"):
            assert restored_forces[key] == pytest.approx(saved_forces[key], rel=1e-12, abs=1e-12)
        resumed.advance(defer_output=True)
        restored_next = resumed.vlm_solver.compute_forces(1.3)
        for key in ("force_z", "moment_y", "unsteady_force_z"):
            assert restored_next[key] == pytest.approx(next_forces[key], rel=1e-10, abs=1e-11)
        np.testing.assert_allclose(
            resumed.vlm_solver.lattice.panel_moment_correction.to_numpy()[:12],
            next_moments,
            rtol=1e-10,
            atol=1e-11,
        )
        distance, permutation = cKDTree(resumed.particle_position).query(position)
        np.testing.assert_allclose(distance, 0.0, atol=2e-12)
        np.testing.assert_allclose(
            resumed.particle_vortex_strength[permutation], strength, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            resumed.vlm_solver.lattice.get_circulation(), circulation, rtol=1e-11, atol=1e-12
        )
        np.testing.assert_allclose(
            resumed.vlm_solver.lattice.get_collocation_points(), geometry, atol=1e-12
        )
        assert (resumed.step, resumed.time) == (3, 0.03)
        points = np.array([[0.4, 0.0, 0.5], [1.5, 0.2, 0.8]])
        wake_only = resumed.compute_velocity_at_points(points, include_body=False)
        complete = resumed.compute_velocity_at_points(points)
        assert np.linalg.norm(complete - wake_only) > 0.01
        combined, _ = resumed.compute_velocity_and_gradient_at_points(points, particle_spacing=0.2)
        np.testing.assert_allclose(combined, complete, rtol=1e-10, atol=1e-12)
        forces = resumed.compute_forces()
        assert forces["dynamic_pressure"] == 50.0
        assert forces["lift_coefficient"] > 0.0
    finally:
        resumed.close()


def test_complete_coupled_state_is_galilean_invariant_after_wake_transport(tmp_path):
    """A moving body and stationary body agree at accepted times, including their wakes."""
    plate = create_flat_plate(
        chord=1, span=4, angle_of_attack_degrees=5, n_chordwise_panels=2, n_spanwise_panels=3
    )
    states = []
    for moving in (False, True):
        motion = vpm.TranslatingVLM(velocity=[-10.0, 0.0, 0.0]) if moving else vpm.StaticVLM()
        setup = vpm.VLMSetup(
            surfaces=(vpm.VLMSurfaceSetup(plate, kinematics=motion),),
            freestream_velocity=(10.0, 0.0, 0.0),
            dtype="f64",
            kinematic_viscosity=0.01,
            force=vpm.ForceConfig.kutta_joukowski(unsteady=True),
        )
        solver = vpm.VPMSolver(
            vpm.VPMCase(
                directory=tmp_path / str(moving),
                numerics=vpm.Numerics(
                    vlm=setup,
                    freestream_velocity=[0.0, 0.0, 0.0] if moving else [10.0, 0.0, 0.0],
                    time_step_size=0.01,
                    compute_device="CPU",
                    precision="f64",
                    max_n_particles=512,
                    induction=vpm.DirectInduction(),
                    viscous=vpm.ViscousConfig.cs(kinematic_viscosity=0.01),
                    stabilization=vpm.StabilizationConfig.disabled(),
                ),
            )
        )
        try:
            for _ in range(8):
                solver.advance(defer_output=True)
            positions = solver.particle_position.copy()
            if moving:
                positions[:, 0] += 10 * solver.time
            states.append(
                (
                    positions,
                    solver.particle_vortex_strength.copy(),
                    solver.vlm_solver.lattice.get_circulation().copy(),
                    solver.vlm_solver.lattice.get_forces().copy(),
                    solver.vlm_solver.lattice.panel_moment_correction.to_numpy().copy(),
                )
            )
        finally:
            solver.close()
    distance, permutation = cKDTree(states[1][0]).query(states[0][0])
    np.testing.assert_allclose(distance, 0.0, atol=2e-11)
    np.testing.assert_allclose(states[0][1], states[1][1][permutation], rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(states[0][2], states[1][2], rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(states[0][3], states[1][3], rtol=2e-10, atol=1e-11)
    np.testing.assert_allclose(states[0][4], states[1][4], rtol=2e-10, atol=1e-11)
