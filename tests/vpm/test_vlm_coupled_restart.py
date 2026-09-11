"""VLM/VPM continuation and complete velocity sampling on real coupled states."""

from dataclasses import replace
import shutil

from _flat_plate_geometry import create_flat_plate
import h5py
import numpy as np
import pytest
from scipy.spatial import cKDTree

import openonda.vpm as vpm


def _moving_coupled_case(directory):
    """Build a small translating coupled case for public restart-path tests."""
    plate = create_flat_plate(
        chord=1,
        span=4,
        angle_of_attack_degrees=5,
        n_chordwise_panels=2,
        n_spanwise_panels=3,
    )
    vlm = vpm.VLMSetup(
        surfaces=(
            vpm.VLMSurfaceSetup(
                plate,
                kinematics=vpm.TranslatingVLM(velocity=np.array([-10.0, 0.0, 0.0])),
            ),
        ),
        freestream_velocity=(10.0, 0.0, 0.0),
        dtype="f64",
        kinematic_viscosity=0.01,
        wake_core_overlap=2.5,
        force=vpm.ForceConfig.kutta_joukowski(unsteady=True),
    )
    return vpm.VPMCase(
        directory=directory,
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


def test_public_load_backup_v6_migration_requires_manifest_evidence_before_mutation(tmp_path):
    """The public coupled loader rejects absent/wrong v6 evidence transactionally."""
    from source.solvers.vpm.boundary_elements.vlm.solver.restart import restart_identity

    source = vpm.VPMSolver(_moving_coupled_case(tmp_path / "source"))
    target = vpm.VPMSolver(_moving_coupled_case(tmp_path / "target"))
    try:
        # Capture the legacy output-only identity before the moving surface advances.
        legacy_controls = {
            "logging_interval_steps": 4,
            "sample_surface_forces": True,
            "surface_sample_forces": [None],
        }
        legacy_identity = restart_identity(source.vlm_solver, output_controls=legacy_controls)
        source.advance(defer_output=True)
        source.save_backup()
        checkpoint = tmp_path / "source" / "solution" / "vpm_000001.h5"

        with h5py.File(checkpoint, "a") as archive:
            group = archive["solver/vlm"]
            group.attrs["version"] = 6
            group.attrs["identity"] = legacy_identity
            del group.attrs["physics_identity"]
            if "force_density" in group.attrs:
                del group.attrs["force_density"]
            for name in ("area", "relative_velocity", "bound_relative_velocity"):
                del group[name]

        target_step = target.step
        target_time = target.time
        target_geometry = target.vlm_solver.lattice.panel_corner_position.to_numpy().copy()

        # No sibling manifest means the output-only identity cannot be interpreted.
        (tmp_path / "source" / "solution" / "vpm_metadata.json").unlink()
        with pytest.raises(ValueError, match="geometry or configuration"):
            target.load_backup(checkpoint)
        assert (target.step, target.time) == (target_step, target_time)
        np.testing.assert_array_equal(
            target.vlm_solver.lattice.panel_corner_position.to_numpy(), target_geometry
        )

        # Wrong persisted controls also reject before particles or VLM state mutate.
        (tmp_path / "source" / "solution" / "vpm_metadata.json").write_text(
            '{"configuration": {"numerics": {"vlm": {'
            '"logging_interval_steps": 3, "sample_surface_forces": true, '
            '"surfaces": [{"sample_forces": null}]}}}}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="geometry or configuration"):
            target.load_backup(checkpoint)
        assert (target.step, target.time) == (target_step, target_time)
        np.testing.assert_array_equal(
            target.vlm_solver.lattice.panel_corner_position.to_numpy(), target_geometry
        )

        # The complete legacy evidence accepts the migration through VPMSolver.load_backup.
        (tmp_path / "source" / "solution" / "vpm_metadata.json").write_text(
            '{"configuration": {"numerics": {"vlm": {'
            '"logging_interval_steps": 4, "sample_surface_forces": true, '
            '"surfaces": [{"sample_forces": null}]}}}}',
            encoding="utf-8",
        )
        target.load_backup(checkpoint)
        assert target.step == 1
        assert target.time == pytest.approx(0.01)
        assert target._vlm_identity_migration["kind"] == "output_only"
        assert target.vlm_solver.kinematics.current_position[0] == pytest.approx(-0.1)
    finally:
        source.close()
        target.close()


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
        import pyvista as pv

        surface = pv.read(tmp_path / "first/solution/vlm_000002.vtp")
        lattice = original.vlm_solver.lattice
        np.testing.assert_array_equal(
            surface.points,
            lattice.panel_corner_position.to_numpy()[:12].reshape(-1, 3),
        )
        for field in ("circulation", "panel_force", "unsteady_panel_force", "wing_id"):
            np.testing.assert_array_equal(surface[field], getattr(lattice, field).to_numpy()[:12])
        assert pv.get_reader(tmp_path / "first/solution/vlm.pvd").time_values == [original.time]
        saved_forces = original.vlm_solver.compute_forces(1.3)
        checkpoint = tmp_path / "first/solution/vpm_000002.h5"
        # The solver-free export must reproduce the live companion, including
        # area-dependent normal loading, from the persisted VLM state alone.
        from source.solvers.vpm.io.vlm_backup import export_vlm_backup

        export_directory = tmp_path / "solver_free_export"
        export_directory.mkdir()
        export_checkpoint = export_directory / checkpoint.name
        shutil.copy2(checkpoint, export_checkpoint)
        regenerated = pv.read(export_vlm_backup(export_checkpoint))
        live = pv.read(tmp_path / "first/solution/vlm_000002.vtp")
        for field in (
            "area",
            "relative_velocity",
            "bound_relative_velocity",
            "panel_force",
            "panel_normal_load_coefficient",
        ):
            np.testing.assert_array_equal(regenerated[field], live[field])
        assert regenerated.field_data["force_density"][0] == pytest.approx(1.0)
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
        # A manual backup immediately after restore must retain the derived
        # frame fields and dimensional-force provenance, before another solve.
        resumed.save_backup()
        immediate = pv.read(tmp_path / "resumed/solution/vlm_000002.vtp")
        np.testing.assert_array_equal(
            immediate["relative_velocity"],
            resumed.vlm_solver.lattice.relative_velocity.to_numpy()[:12],
        )
        assert immediate.field_data["force_density"][0] == pytest.approx(1.0)
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


def test_changed_time_step_continuation_preserves_vlm_state_and_advances_coupled_loads(tmp_path):
    """An explicit smaller-step restart keeps the accepted VLM state coherent."""
    plate = create_flat_plate(
        chord=1, span=4, angle_of_attack_degrees=5, n_chordwise_panels=2, n_spanwise_panels=3
    )

    def make_case(directory, time_step_size):
        vlm = vpm.VLMSetup(
            surfaces=(
                vpm.VLMSurfaceSetup(
                    plate,
                    kinematics=vpm.TranslatingVLM(velocity=np.array([-10.0, 0.0, 0.0])),
                ),
            ),
            freestream_velocity=(10.0, 0.0, 0.0),
            dtype="f64",
            kinematic_viscosity=0.01,
            wake_core_overlap=2.5,
            force=vpm.ForceConfig.kutta_joukowski(unsteady=True),
        )
        return vpm.VPMCase(
            directory=directory,
            numerics=vpm.Numerics(
                vlm=vlm,
                freestream_velocity=[10.0, 0.0, 0.0],
                time_step_size=time_step_size,
                compute_device="CPU",
                precision="f64",
                max_n_particles=256,
                induction=vpm.DirectInduction(),
                viscous=vpm.ViscousConfig.cs(kinematic_viscosity=0.01),
                stabilization=vpm.StabilizationConfig.disabled(),
            ),
            samplers=vpm.Samplers(
                directory="changed",
            ),
        )

    original = vpm.VPMSolver(make_case(tmp_path / "original", 0.01))
    try:
        original.advance(defer_output=True)
        original.advance(defer_output=True)
        original.save_backup()
        checkpoint = tmp_path / "original/solution/vpm_000002.h5"
        expected_circulation = original.vlm_solver.lattice.get_circulation().copy()
        surface_name = next(iter(original.vlm_solver.surfaces))
        expected_position = original.vlm_solver.surfaces[surface_name][1].current_position.copy()
        assert (original.step, original.time) == (2, pytest.approx(0.02))
    finally:
        original.close()

    inconsistent_checkpoint = tmp_path / "original/solution/inconsistent.h5"
    shutil.copy2(checkpoint, inconsistent_checkpoint)
    with h5py.File(inconsistent_checkpoint, "r+") as file:
        file["solver/vlm"].attrs["time"] = 0.0205
    clock_reader = vpm.VPMSolver(make_case(tmp_path / "clock-reader", 0.005))
    try:
        with pytest.raises(ValueError, match="VLM restart time does not match"):
            clock_reader.load_backup(inconsistent_checkpoint, time_step_size=0.005)
        # Clock validation is part of the pre-mutation restart validation.
        assert clock_reader.step == 0
        assert clock_reader.time == pytest.approx(0.0)
        assert clock_reader.particles.n_particles_total == 0
    finally:
        clock_reader.close()

    resumed = vpm.VPMSolver(make_case(tmp_path / "resumed", 0.005))
    try:
        resumed.load_backup(checkpoint, time_step_size=0.005)
        assert resumed.time_step_size == pytest.approx(0.005)
        assert (resumed.step, resumed.time) == (2, pytest.approx(0.02))
        assert resumed.vlm_solver._current_time == pytest.approx(0.02)
        np.testing.assert_array_equal(
            resumed.vlm_solver.lattice.get_circulation(), expected_circulation
        )
        np.testing.assert_array_equal(
            resumed.vlm_solver.surfaces[surface_name][1].current_position,
            expected_position,
        )

        resumed.advance()
        resumed.save_backup()
        assert (resumed.step, resumed.time) == (3, pytest.approx(0.025))
        assert resumed.vlm_solver._current_time == pytest.approx(0.025)
        pvd = (tmp_path / "resumed/solution/vlm.pvd").read_text(encoding="utf-8")
        assert 'timestep="0.025" file="vlm_000003.vtp"' in pvd
        forces = resumed.vlm_solver.compute_forces(1.3)
        assert all(np.isfinite(value).all() for value in forces.values())
        assert resumed.particles.n_particles_total > 0
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
