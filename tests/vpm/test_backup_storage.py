"""VPM backups stay compact, restartable, and readable by ParaView."""

from __future__ import annotations

import contextlib
import importlib.util
import io
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
    Numerics,
    TreecodeInduction,
    ViscousConfig,
    VPMCase,
    VPMSolver,
)


def _load_ring_metrics():
    path = Path(__file__).parents[2] / "tutorials/vpm/vortex_ring/assets/ring_metrics.py"
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

    ring_data = _load_ring_metrics().load_ring_data([f"{backup}.h5"])
    assert len(ring_data[0]) == 1


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


def test_vpm_case_has_no_partial_configuration_serialization(tmp_path):
    case = _case(tmp_path / "custom-output")
    assert not hasattr(case, "to_dict")
    assert not hasattr(VPMCase, "from_dict")


def test_backup_refresh_does_not_compute_velocity_gradients(tmp_path, monkeypatch):
    solver = _solver(tmp_path / "refresh")
    calls = []
    monkeypatch.setattr(
        solver.stepper,
        "_update_velocity_gradients",
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
        induction=TreecodeInduction(theta=0.5),
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
        induction=TreecodeInduction(theta=0.5),
        viscous=ViscousConfig.inviscid(particle_spacing=0.2),
    )
    monkeypatch.setattr(reader.stepper, "_update_velocity_gradients", lambda **_kwargs: None)
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
