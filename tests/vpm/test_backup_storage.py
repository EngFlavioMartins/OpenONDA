"""VPM backups stay compact, restartable, and readable by ParaView."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import multiprocessing
from pathlib import Path

import h5py
import numpy as np
import pytest

from source.solvers.vpm import (
    RK2,
    RK4,
    SSPRK3,
    Backup,
    DirectInduction,
    FilamentRefinementConfig,
    FMMInduction,
    Numerics,
    StabilizationConfig,
    TreecodeInduction,
    ViscousConfig,
    VPMCase,
    VPMSolver,
)


def _load_ring_metrics():
    path = Path(__file__).parents[2] / "tutorials/vpm/02_vortex_ring/assets/ring_metrics.py"
    spec = importlib.util.spec_from_file_location("vortex_ring_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(
    case_directory: Path,
    *,
    time_step_size: float = 0.01,
    max_n_particles: int = 64,
    integrator=None,
    viscous: ViscousConfig | None = None,
    induction=None,
    random_seed: int = 42,
    stabilization: StabilizationConfig | None = None,
) -> VPMCase:
    return VPMCase(
        directory=case_directory,
        backup=Backup(interval_steps=0, directory="solution", log_directory="solution"),
        numerics=Numerics(
            time_step_size=time_step_size,
            compute_device="CPU",
            max_n_particles=max_n_particles,
            domain_bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            write_precision="f16",
            verbose=False,
            integrator=SSPRK3() if integrator is None else integrator,
            viscous=(
                ViscousConfig.cs(kinematic_viscosity=0.01, particle_spacing=0.2)
                if viscous is None
                else viscous
            ),
            induction=DirectInduction() if induction is None else induction,
            random_seed=random_seed,
            stabilization=(
                StabilizationConfig.disabled() if stabilization is None else stabilization
            ),
        ),
    )


def _solver(case_dir, **case_options) -> VPMSolver:
    with contextlib.redirect_stdout(io.StringIO()):
        return VPMSolver(_case(case_dir, **case_options))


def _add_counter_rotating_pair(solver: VPMSolver) -> None:
    solver.add_vortex_particles(
        position=np.array([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        velocity=np.zeros((2, 3)),
        vortex_strength=np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
        core_radius=np.full(2, 0.2),
        particle_volume=np.full(2, 0.2**3),
        kinematic_viscosity=np.full(2, 0.01),
    )


def _advance(solver: VPMSolver, steps: int) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            solver.advance(defer_output=True)


def _write_rwm_process_result(case_directory: str, output_file: str) -> None:
    """Run one seeded RWM trajectory in a fresh spawned Python process."""
    solver = _solver(
        Path(case_directory),
        integrator=SSPRK3(),
        viscous=ViscousConfig.rwm(kinematic_viscosity=0.01, particle_spacing=0.2),
        induction=DirectInduction(),
        random_seed=42,
    )
    _add_counter_rotating_pair(solver)
    _advance(solver, 4)
    np.save(output_file, solver.particle_position)


def test_vpm_backup_has_one_fixed_restart_schema(tmp_path):
    solver = _solver(tmp_path / "writer")
    solver.stabilization.regularization_events = 1
    solver.stabilization.pedrizzetti_moment_transfer[:] = np.arange(9).reshape(3, 3) * 0.0123
    solver.stabilization.regularization_energy_transfer = -0.12
    solver.stabilization.regularization_enstrophy_transfer = -0.34
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0], [0.11, 0.0, 0.0]], dtype=np.float32),
        velocity=np.zeros((2, 3), dtype=np.float32),
        vortex_strength=np.array([[0.0, 0.0, 0.01], [0.0, 0.02, 0.0]], dtype=np.float32),
        core_radius=np.array([0.05, 0.05], dtype=np.float32),
        particle_volume=np.array([0.008, 0.008], dtype=np.float32),
        kinematic_viscosity=np.array([0.01, 0.01], dtype=np.float32),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver.save_backup()
    backup = tmp_path / "writer" / "solution" / "vpm_000000"

    with h5py.File(f"{backup}.h5", "r") as archive:
        particles = archive["particles"]
        assert particles["position"].dtype == np.float32
        assert particles["position"].compression == "gzip"
        assert particles["position"].shuffle
        assert "velocity_gradient" not in particles
        assert "strain_rate" not in particles
        assert "backup_store_velocity_gradient" not in archive["solver"].attrs
        assert archive["solver"].attrs["backup_format_version"] == "10.0"

    xdmf = Path(f"{backup}.xdmf").read_text(encoding="utf-8")
    assert 'Name="velocity_gradient"' not in xdmf

    import pyvista as pv

    visual = pv.read(f"{backup}.xdmf")
    assert "velocity" in visual.point_data
    assert "velocity_gradient" not in visual.point_data

    restored = _solver(tmp_path / "reader")
    with contextlib.redirect_stdout(io.StringIO()):
        restored.load_backup(str(backup))

    np.testing.assert_allclose(
        restored.particle_position,
        solver.particle_position,
        rtol=0.0,
        atol=4.0e-5,
    )
    assert np.isfinite(restored.particles.velocity_gradient_cpu()).all()
    assert restored.stabilization.regularization_energy_transfer == pytest.approx(-0.12)
    np.testing.assert_array_equal(
        restored.stabilization.pedrizzetti_moment_transfer,
        solver.stabilization.pedrizzetti_moment_transfer,
    )
    assert restored.stabilization.regularization_enstrophy_transfer == pytest.approx(-0.34)

    ring_data = _load_ring_metrics().load_ring_data([f"{backup}.h5"])
    assert len(ring_data[0]) == 1
    # Old checkpoints stored event counts without the transfer ledger.
    with h5py.File(f"{backup}.h5", "r+") as archive:
        del archive["solver"].attrs["regularization_cumulative_total_kinetic_energy_transfer"]
        del archive["solver"].attrs["regularization_cumulative_total_enstrophy_transfer"]
    with contextlib.redirect_stdout(io.StringIO()):
        restored.load_backup(str(backup))
    assert np.isnan(restored.stabilization.regularization_energy_transfer)
    assert np.isnan(restored.stabilization.regularization_enstrophy_transfer)


def test_vpm_restart_preserves_compute_precision_and_freestream(tmp_path):
    solver = _solver(tmp_path / "writer")
    position = np.array([[0.12345679, -0.2345679, 0.3456789]], dtype=np.float32)
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros((1, 3), dtype=np.float32),
        vortex_strength=np.array([[0.0, 0.0, 0.01]], dtype=np.float32),
        core_radius=np.array([0.05], dtype=np.float32),
        particle_volume=np.array([0.008], dtype=np.float32),
        kinematic_viscosity=np.array([0.01], dtype=np.float32),
    )
    solver._set_freestream_velocity([0.12345679, -0.25, 0.5])
    with contextlib.redirect_stdout(io.StringIO()):
        solver.save_backup()
    backup = tmp_path / "writer" / "solution" / "vpm_000000"

    with h5py.File(f"{backup}.h5", "r") as archive:
        assert archive["particles"]["position"].dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(archive["particles"]["position"][:], position)

    restored = _solver(tmp_path / "reader")
    restored._set_freestream_velocity([9.0, 8.0, 7.0])
    with contextlib.redirect_stdout(io.StringIO()):
        restored.load_backup(str(backup))

    np.testing.assert_array_equal(restored.particle_position, position)
    np.testing.assert_array_equal(restored.freestream_velocity, solver.freestream_velocity)


def test_restart_preserves_unbounded_filament_refinement(tmp_path):
    def solver(directory, factor):
        return _solver(
            tmp_path / directory,
            stabilization=StabilizationConfig(
                filament_refinement=FilamentRefinementConfig.adaptive(
                    interval_steps=1,
                    max_vortex_strength_factor=factor,
                    max_absolute_vortex_strength=1.0,
                )
            ),
        )

    writer = solver("writer", float("inf"))
    try:
        _add_counter_rotating_pair(writer)
        _advance(writer, 1)
        expected_position = writer.particle_position.copy()
        writer.save_backup()
    finally:
        writer.close()
    backup = tmp_path / "writer/solution/vpm_000001.h5"
    restored = solver("reader", np.float32("inf"))
    try:
        restored.load_backup(str(backup))
        assert restored.step == 1
        np.testing.assert_array_equal(restored.particle_position, expected_position)
    finally:
        restored.close()
    incompatible = solver("finite-threshold", 2.0)
    try:
        with pytest.raises(ValueError, match="max_vortex_strength_factor"):
            incompatible.load_backup(str(backup))
    finally:
        incompatible.close()


def test_vpm_restart_rejects_incompatible_format_with_versions(tmp_path):
    solver = _solver(tmp_path / "writer")
    with contextlib.redirect_stdout(io.StringIO()):
        solver.save_backup()
    backup = tmp_path / "writer" / "solution" / "vpm_000000.h5"
    with h5py.File(backup, "r+") as archive:
        archive["solver"].attrs["backup_format_version"] = "9.0"

    with pytest.raises(ValueError, match=r"9.0.*10.0"):
        solver.load_backup(str(backup))


def test_vpm_restart_reports_the_incompatible_configuration_path(tmp_path):
    solver = _solver(tmp_path / "writer")
    with contextlib.redirect_stdout(io.StringIO()):
        solver.save_backup()
    backup = tmp_path / "writer" / "solution" / "vpm_000000"

    case = _case(tmp_path / "reader", time_step_size=0.02)
    with contextlib.redirect_stdout(io.StringIO()):
        reader = VPMSolver(case)

    with pytest.raises(ValueError, match=r"numerical configuration mismatch at time_step_size"):
        reader.load_backup(str(backup))


def test_explicit_changed_time_step_restart_preserves_clock_and_checkpoint_state(tmp_path):
    writer = _solver(tmp_path / "writer", time_step_size=0.01)
    try:
        _add_counter_rotating_pair(writer)
        _advance(writer, 2)
        expected_position = writer.particle_position.copy()
        expected_strength = writer.particle_vortex_strength.copy()
        writer.save_backup()
    finally:
        writer.close()

    backup = tmp_path / "writer/solution/vpm_000002.h5"
    reader = _solver(tmp_path / "reader", time_step_size=0.005)
    try:
        reader.load_backup(backup, time_step_size=0.005)
        assert reader.time_step_size == pytest.approx(0.005)
        assert (reader.step, reader.time) == (2, pytest.approx(0.02))
        np.testing.assert_array_equal(reader.particle_position, expected_position)
        np.testing.assert_array_equal(reader.particle_vortex_strength, expected_strength)

        _advance(reader, 1)
        assert (reader.step, reader.time) == (3, pytest.approx(0.025))
        reader.save_backup()
    finally:
        reader.close()

    with h5py.File(tmp_path / "reader/solution/vpm_000003.h5", "r") as archive:
        assert archive["solver"].attrs["step"] == 3
        assert archive["solver"].attrs["time"] == pytest.approx(0.025)
        assert archive["solver"].attrs["time_step_size"] == pytest.approx(0.005)
    writer_metadata = json.loads(
        (tmp_path / "writer/solution/vpm_metadata.json").read_text(encoding="utf-8")
    )
    assert "restart" not in writer_metadata
    reader_metadata = json.loads(
        (tmp_path / "reader/solution/vpm_metadata.json").read_text(encoding="utf-8")
    )
    restart = reader_metadata["restart"]
    assert restart["source"]["accepted_step"] == 2
    assert restart["source"]["accepted_time"] == pytest.approx(0.02)
    assert restart["source"]["time_step_size"] == pytest.approx(0.01)
    assert restart["continuation"]["requested_time_step_size"] == pytest.approx(0.005)
    assert restart["continuation"]["accepted_time"] == pytest.approx(0.02)


def test_changed_time_step_restart_does_not_relax_other_configuration_checks(tmp_path):
    writer = _solver(tmp_path / "writer", time_step_size=0.01)
    try:
        writer.save_backup()
    finally:
        writer.close()
    backup = tmp_path / "writer/solution/vpm_000000.h5"
    reader = _solver(tmp_path / "reader", time_step_size=0.005, random_seed=43)
    try:
        _add_counter_rotating_pair(reader)
        before_position = reader.particle_position.copy()
        with pytest.raises(ValueError, match=r"numerical configuration mismatch at random_seed"):
            reader.load_backup(backup, time_step_size=0.005)
        assert reader.step == 0
        assert reader.time == 0.0
        np.testing.assert_array_equal(reader.particle_position, before_position)
    finally:
        reader.close()


@pytest.mark.parametrize(
    ("override", "error"),
    [
        (0.0, ValueError),
        (-0.001, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("0.001", TypeError),
    ],
)
def test_invalid_changed_time_step_override_fails_before_live_state_mutation(
    tmp_path, override, error
):
    reader = _solver(tmp_path / "reader", time_step_size=0.005)
    try:
        _add_counter_rotating_pair(reader)
        before_position = reader.particle_position.copy()
        before_strength = reader.particle_vortex_strength.copy()
        with pytest.raises(error):
            reader.load_backup(tmp_path / "missing.h5", time_step_size=override)
        assert reader.step == 0
        assert reader.time == 0.0
        assert reader.time_step_size == pytest.approx(0.005)
        np.testing.assert_array_equal(reader.particle_position, before_position)
        np.testing.assert_array_equal(reader.particle_vortex_strength, before_strength)
    finally:
        reader.close()


def test_changed_time_step_restart_rejects_dvh_history_remapping(tmp_path):
    reader = _solver(
        tmp_path / "reader",
        time_step_size=0.005,
        viscous=ViscousConfig.dvh(particle_spacing=0.1, kinematic_viscosity=0.01),
    )
    try:
        with pytest.raises(ValueError, match=r"not supported for DVH"):
            reader.load_backup(tmp_path / "missing.h5", time_step_size=0.001)
    finally:
        reader.close()


@pytest.mark.parametrize("induction_type", (DirectInduction, TreecodeInduction, FMMInduction))
def test_larger_restart_capacity_preserves_the_particle_trajectory(tmp_path, induction_type):
    writer = _solver(tmp_path / "writer", induction=induction_type())
    random = np.random.default_rng(21)
    count = 48
    try:
        writer.add_vortex_particles(
            position=random.uniform(-0.7, 0.7, (count, 3)),
            velocity=np.zeros((count, 3)),
            vortex_strength=random.normal(0.0, 1e-4, (count, 3)),
            core_radius=np.full(count, 0.08),
            particle_volume=np.full(count, 0.08**3),
            kinematic_viscosity=np.full(count, 0.01),
        )
        _advance(writer, 1)
        writer.save_backup()
        _advance(writer, 1)
        position = writer.particle_position.copy()
        strength = writer.particle_vortex_strength.copy()
    finally:
        writer.close()
    reader = _solver(tmp_path / "reader", max_n_particles=128, induction=induction_type())
    try:
        reader.load_backup(tmp_path / "writer/solution/vpm_000001.h5")
        _advance(reader, 1)
        np.testing.assert_allclose(reader.particle_position, position, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(reader.particle_vortex_strength, strength, rtol=1e-5, atol=1e-10)
        assert reader.particles.n_particles_total == count
        assert reader.step == 2
    finally:
        reader.close()


@pytest.mark.parametrize("policy", ("smaller", "filament", "remesh"))
def test_restart_keeps_capacity_checks_for_smaller_or_adaptive_allocations(tmp_path, policy):
    stabilization = StabilizationConfig.disabled()
    if policy == "filament":
        stabilization = StabilizationConfig(
            filament_refinement=FilamentRefinementConfig.adaptive(interval_steps=1)
        )
    elif policy == "remesh":
        stabilization = StabilizationConfig(
            regularization_interval_steps=1, regularization_grid_spacing=0.1
        )
    writer = _solver(tmp_path / "writer", stabilization=stabilization)
    try:
        _add_counter_rotating_pair(writer)
        writer.save_backup()
    finally:
        writer.close()
    reader = _solver(
        tmp_path / "reader",
        max_n_particles=32 if policy == "smaller" else 128,
        stabilization=stabilization,
    )
    try:
        with pytest.raises(ValueError, match="max_n_particles"):
            reader.load_backup(tmp_path / "writer/solution/vpm_000000.h5")
        assert reader.particles.n_particles_total == 0
    finally:
        reader.close()


def test_vpm_case_has_no_partial_configuration_serialization(tmp_path):
    case = _case(tmp_path / "custom-output")
    assert not hasattr(case, "to_dict")
    assert not hasattr(VPMCase, "from_dict")


def test_backup_refresh_does_not_compute_velocity_gradients(tmp_path, monkeypatch):
    solver = _solver(tmp_path / "refresh")
    calls = []
    monkeypatch.setattr(
        solver.stepper,
        "_update_velocity_and_gradients",
        lambda: calls.append("gradient"),
    )

    solver._refresh_backup_particle_fields()

    assert calls == []


def test_empty_vpm_backup_is_still_paraview_readable(tmp_path):
    solver = _solver(tmp_path / "writer")
    with contextlib.redirect_stdout(io.StringIO()):
        solver.save_backup()
    backup = tmp_path / "writer" / "solution" / "vpm_000000"

    import pyvista as pv

    visual = pv.read(f"{backup}.xdmf")

    assert visual.n_points == 0


def _assert_split_run_matches(tmp_path, options: dict[str, object]) -> None:
    uninterrupted = _solver(tmp_path / "uninterrupted", **options)
    interrupted = _solver(tmp_path / "interrupted", **options)
    _add_counter_rotating_pair(uninterrupted)
    _add_counter_rotating_pair(interrupted)

    _advance(uninterrupted, 4)
    _advance(interrupted, 2)
    with contextlib.redirect_stdout(io.StringIO()):
        interrupted.save_backup()

    resumed = _solver(tmp_path / "resumed", **options)
    resumed.load_backup(str(tmp_path / "interrupted" / "solution" / "vpm_000002"))
    _advance(resumed, 2)

    assert resumed.step == uninterrupted.step == 4
    assert resumed.time == uninterrupted.time
    np.testing.assert_allclose(
        resumed.particle_position,
        uninterrupted.particle_position,
        rtol=5.0e-7,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        resumed.particle_vortex_strength,
        uninterrupted.particle_vortex_strength,
        rtol=5.0e-7,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        resumed.particle_core_radius,
        uninterrupted.particle_core_radius,
        rtol=5.0e-7,
        atol=1.0e-10,
    )


@pytest.mark.parametrize("integrator", [RK2(), SSPRK3(), RK4()], ids=["rk2", "ssprk3", "rk4"])
def test_split_run_matches_each_deterministic_integrator(tmp_path, integrator):
    """Same-backend backups preserve each deterministic advection trajectory."""
    _assert_split_run_matches(
        tmp_path,
        {
            "integrator": integrator,
            "viscous": ViscousConfig.cs(kinematic_viscosity=0.01, particle_spacing=0.2),
            "induction": DirectInduction(),
        },
    )


@pytest.mark.qualification
def test_fmm_restart_and_repeated_run_match_over_twenty_accepted_steps(tmp_path):
    """Production FMM preserves state across a 10+10 restart and a repeated run."""
    uninterrupted = _solver(
        tmp_path / "uninterrupted",
        integrator=SSPRK3(),
        viscous=ViscousConfig.inviscid(particle_spacing=0.2),
        induction=FMMInduction(),
    )
    interrupted = _solver(
        tmp_path / "interrupted",
        integrator=SSPRK3(),
        viscous=ViscousConfig.inviscid(particle_spacing=0.2),
        induction=FMMInduction(),
    )
    repeated = _solver(
        tmp_path / "repeated",
        integrator=SSPRK3(),
        viscous=ViscousConfig.inviscid(particle_spacing=0.2),
        induction=FMMInduction(),
    )
    for solver in (uninterrupted, interrupted, repeated):
        _add_counter_rotating_pair(solver)

    _advance(uninterrupted, 20)
    _advance(repeated, 20)
    _advance(interrupted, 10)
    with contextlib.redirect_stdout(io.StringIO()):
        interrupted.save_backup()

    resumed = _solver(
        tmp_path / "resumed",
        integrator=SSPRK3(),
        viscous=ViscousConfig.inviscid(particle_spacing=0.2),
        induction=FMMInduction(),
    )
    resumed.load_backup(str(tmp_path / "interrupted" / "solution" / "vpm_000010"))
    _advance(resumed, 10)

    assert uninterrupted.step == repeated.step == resumed.step == 20
    assert uninterrupted.time == repeated.time == resumed.time
    for candidate in (repeated, resumed):
        np.testing.assert_allclose(
            candidate.particle_position,
            uninterrupted.particle_position,
            rtol=2.0e-6,
            atol=2.0e-9,
        )
        np.testing.assert_allclose(
            candidate.particle_vortex_strength,
            uninterrupted.particle_vortex_strength,
            rtol=2.0e-6,
            atol=2.0e-9,
        )
        np.testing.assert_array_equal(
            candidate.particle_core_radius,
            uninterrupted.particle_core_radius,
        )
    for solver in (uninterrupted, repeated, resumed):
        assert solver.induction.diagnostics.host_particle_transfers == 0
        assert solver.induction.diagnostics.direct_strength_rate_fallbacks == 0


@pytest.mark.parametrize(
    "viscous",
    [
        ViscousConfig.inviscid(particle_spacing=0.2),
        ViscousConfig.cs(kinematic_viscosity=0.01, particle_spacing=0.2),
        ViscousConfig.rwm(kinematic_viscosity=0.01, particle_spacing=0.2),
        ViscousConfig.dvh(
            particle_spacing=0.2,
            padding=3.0,
            threshold=1.0e-12,
            kinematic_viscosity=0.01,
            max_nodes=5_000,
        ),
        ViscousConfig.gbd(
            particle_spacing=0.2,
            padding=3.0,
            threshold=1.0e-12,
            kinematic_viscosity=0.01,
            max_nodes=5_000,
        ),
    ],
    ids=("none", "core-spreading", "rwm", "dvh", "gbd"),
)
def test_split_run_matches_each_deterministic_viscous_scheme(tmp_path, viscous):
    """Same-backend backups preserve every viscous state machine, including RWM."""
    _assert_split_run_matches(
        tmp_path,
        {
            "integrator": SSPRK3(),
            "viscous": viscous,
            "induction": DirectInduction(),
        },
    )


@pytest.mark.qualification
@pytest.mark.stochastic
def test_seeded_rwm_matches_across_fresh_processes(tmp_path):
    """The declared seed produces the same trajectory in fresh interpreters."""
    context = multiprocessing.get_context("spawn")
    outputs = []
    for index in range(2):
        output = tmp_path / f"rwm_{index}.npy"
        process = context.Process(
            target=_write_rwm_process_result,
            args=(str(tmp_path / f"case_{index}"), str(output)),
        )
        process.start()
        process.join(timeout=60.0)
        assert process.exitcode == 0
        outputs.append(np.load(output))

    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_truncated_backup_is_rejected_before_state_mutation(tmp_path):
    writer = _solver(tmp_path / "writer")
    _add_counter_rotating_pair(writer)
    with contextlib.redirect_stdout(io.StringIO()):
        writer.save_backup()
    valid = tmp_path / "writer" / "solution" / "vpm_000000.h5"
    corrupted = tmp_path / "truncated.h5"
    corrupted.write_bytes(valid.read_bytes()[:128])

    reader = _solver(tmp_path / "reader")
    with pytest.raises((OSError, ValueError, KeyError)):
        reader.load_backup(str(corrupted))
    assert reader.particles.n_particles_total == 0
    assert reader.step == 0


def test_atomic_backup_failure_preserves_the_last_complete_file(tmp_path, monkeypatch):
    solver = _solver(tmp_path / "writer")
    _add_counter_rotating_pair(solver)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.save_backup()
    destination = tmp_path / "writer" / "solution" / "vpm_000000.h5"
    original = destination.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("simulated publication failure")

    monkeypatch.setattr("source.solvers.vpm.io.backup.os.replace", fail_replace)
    with (
        pytest.raises(RuntimeError, match="simulated publication failure"),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        solver.save_backup()

    assert destination.read_bytes() == original
    assert not list(destination.parent.glob("*.tmp"))


def test_backup_storage_handles_more_than_fifty_thousand_particles(tmp_path, monkeypatch):
    """Exercise the storage path above the former hidden 50,000-particle branch."""
    count = 50_001
    writer = _solver(
        tmp_path / "writer",
        max_n_particles=count,
        induction=TreecodeInduction(),
        viscous=ViscousConfig.inviscid(particle_spacing=0.2),
    )
    index = np.arange(count, dtype=np.float32)
    position = np.column_stack(
        (
            -0.9 + 1.8 * index / count,
            0.1 * np.sin(index),
            0.1 * np.cos(index),
        )
    )
    writer.add_vortex_particles(
        position=position,
        velocity=np.zeros((count, 3), dtype=np.float32),
        vortex_strength=np.column_stack((np.zeros(count), np.zeros(count), np.full(count, 1.0e-8))),
        core_radius=np.full(count, 0.2),
        particle_volume=np.full(count, 0.2**3),
        kinematic_viscosity=np.zeros(count),
    )
    monkeypatch.setattr(writer, "_refresh_backup_particle_fields", lambda: None)
    with contextlib.redirect_stdout(io.StringIO()):
        writer.save_backup()

    reader = _solver(
        tmp_path / "reader",
        max_n_particles=count,
        induction=TreecodeInduction(),
        viscous=ViscousConfig.inviscid(particle_spacing=0.2),
    )
    monkeypatch.setattr(reader.stepper, "_update_velocity_and_gradients", lambda **_kwargs: None)
    reader.load_backup(str(tmp_path / "writer" / "solution" / "vpm_000000"))

    assert reader.particles.n_particles_total == count
    np.testing.assert_array_equal(reader.particle_position[[0, -1]], position[[0, -1]])


def test_backup_rejects_a_different_random_seed(tmp_path):
    writer = _solver(tmp_path / "writer", random_seed=7)
    with contextlib.redirect_stdout(io.StringIO()):
        writer.save_backup()

    reader = _solver(tmp_path / "reader", random_seed=8)
    with pytest.raises(ValueError, match="random_seed"):
        reader.load_backup(str(tmp_path / "writer" / "solution" / "vpm_000000"))


@pytest.mark.parametrize("induction_type", (DirectInduction, TreecodeInduction, FMMInduction))
def test_restart_matches_stretching_and_recovers_only_known_implicit_defaults(
    tmp_path, induction_type
):
    solver = _solver(tmp_path / "writer", induction=induction_type())
    solver.save_backup()
    backup = tmp_path / "writer" / "solution" / "vpm_000000"
    solver.close()
    forms = ("explicit", "implicit") if induction_type is not TreecodeInduction else ("explicit",)
    for form in forms:
        if form == "implicit":
            with h5py.File(f"{backup}.h5", "r+") as archive:
                attrs = archive["solver"].attrs
                configuration = json.loads(attrs["numerical_configuration"])
                del configuration["induction"]["stretching_scheme"]
                encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
                attrs["numerical_configuration"] = encoded
                attrs["numerical_configuration_sha256"] = hashlib.sha256(
                    encoded.encode()
                ).hexdigest()
        for scheme in ("DIRECT", "TRANSPOSED", "MIXED"):
            reader = _solver(
                tmp_path / f"reader-{form}-{scheme}",
                induction=induction_type(stretching_scheme=scheme),
            )
            try:
                if scheme == "TRANSPOSED":
                    reader.load_backup(str(backup))
                    assert reader.induction.stretching_scheme == scheme
                else:
                    with pytest.raises(ValueError, match="induction.stretching_scheme"):
                        reader.load_backup(str(backup))
            finally:
                reader.close()


def test_flow_integral_csv_preserves_energy_definitions_and_unknown_legacy_rows(tmp_path):
    import pandas as pd

    solver = _solver(tmp_path / "energy_measurement")
    csv = tmp_path / "flow_integrals.csv"
    try:
        solver._update_all_flow_integrals()
        assert solver._flow_integrals["energy_measurement"] == "empty_particle_field"
        _add_counter_rotating_pair(solver)
        solver._update_all_flow_integrals()
        assert solver._flow_integrals["energy_measurement"] == "unbounded_energy"
        solver.io.export_flow_integrals_csv(solver, csv)
        original = pd.read_csv(csv)
        assert original.loc[0, "energy_measurement"] == "unbounded_energy"

        # An older CSV has real numeric values but no saved energy definition.
        original.drop(columns="energy_measurement").to_csv(csv, index=False)
        solver.step = 1
        solver.time = 0.01
        solver._flow_integrals["energy_measurement"] = "periodic_fourier_energy"
        solver._flow_integrals["kinetic_energy_rate_source"] = "fourier_transition_viscous_rate"
        solver.io.export_flow_integrals_csv(solver, csv)
        result = pd.read_csv(csv)
        assert result.energy_measurement.tolist() == ["unknown", "periodic_fourier_energy"]
        pd.testing.assert_series_equal(
            result.total_kinetic_energy.iloc[:1], original.total_kinetic_energy
        )
        assert result.loc[1, "kinetic_energy_rate_source"] == "fourier_transition_viscous_rate"
    finally:
        solver.close()


def test_relaxation_transfer_is_native_sampled_and_restartable(tmp_path):
    import pandas as pd

    config = StabilizationConfig.pedrizzetti_relaxation(factor=0.3)
    solver = _solver(tmp_path / "relaxation_writer", stabilization=config)
    _add_counter_rotating_pair(solver)
    position = solver.particles.position_cpu().astype(float)
    original = solver.particles.vortex_strength_cpu().astype(float)
    gradient = np.zeros((64, 3, 3), dtype=np.float32)
    gradient[:2, 2, 1] = 1.0
    gradient[:2, 0, 2] = 0.5
    solver.particles.velocity_gradient.from_numpy(gradient)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.stabilization.apply_relaxation()
    change = solver.particles.vortex_strength_cpu(use_cache=False).astype(float) - original
    transfer = solver.stabilization.pedrizzetti_moment_transfer
    np.testing.assert_allclose(transfer[0], change.sum(axis=0), atol=1e-14)
    np.testing.assert_allclose(
        transfer[1], 0.5 * np.cross(position, change).sum(axis=0), atol=1e-14
    )
    assert np.linalg.norm(transfer) > 0
    csv = tmp_path / "flow_integrals.csv"
    solver.io.export_flow_integrals_csv(solver, csv)
    row = pd.read_csv(csv).iloc[0]
    for index, axis in enumerate("xyz"):
        assert row[f"pedrizzetti_cumulative_linear_impulse_transfer_{axis}"] == pytest.approx(
            transfer[1, index]
        )
    with contextlib.redirect_stdout(io.StringIO()):
        solver.save_backup()
    backup = tmp_path / "relaxation_writer/solution/vpm_000000.h5"
    restored = _solver(tmp_path / "relaxation_reader", stabilization=config)
    with contextlib.redirect_stdout(io.StringIO()):
        restored.load_backup(backup)
    np.testing.assert_array_equal(restored.stabilization.pedrizzetti_moment_transfer, transfer)
    # Missing historical transfer is unknown, even when reusing a solver that
    # previously loaded a modern checkpoint. No zero or stale total is allowed.
    with h5py.File(backup, "r+") as archive:
        for key in list(archive["solver"].attrs):
            if key.startswith("pedrizzetti_cumulative_"):
                del archive["solver"].attrs[key]
    with contextlib.redirect_stdout(io.StringIO()):
        restored.load_backup(backup)
    assert np.isnan(restored.stabilization.pedrizzetti_moment_transfer).all()
