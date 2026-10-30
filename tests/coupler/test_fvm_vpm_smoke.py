"""Coupled FVM/VPM time stepping, event alignment and restart."""

from __future__ import annotations

import csv
import json
import shutil

import numpy as np
import pytest

from openonda import coupler, fvm, vpm
from openonda.core.vtk_output import read_collection
from tests.support.coupler import FREESTREAM_VELOCITY, uniform_stream_cases


def _solver(
    directory,
    *,
    end_time=0.45,
    time_step_size=0.05,
    vpm_time_step_size=None,
    time_step_control=None,
    vpm_output=None,
    fvm_samplers=None,
):
    flow, particles = uniform_stream_cases(
        directory,
        fvm_time_step_size=time_step_size,
        vpm_time_step_size=vpm_time_step_size,
        end_time=end_time,
        fvm_output=fvm.Output(),
        fvm_samplers=fvm_samplers,
        vpm_output=vpm_output or vpm.Output(schedule=vpm.EverySteps(3)),
        max_n_particles=50_000,
        fvm_time_step_control=time_step_control,
    )
    config = coupler.CouplerConfig(
        transfer_inset_spacings=0.0,
        blend_end_depth=0.0,
        freestream_velocity=FREESTREAM_VELOCITY,
        backup=coupler.Backup(schedule=coupler.EverySteps(2)),
    )
    return coupler.CoupledSolver(flow, particles, config)


def test_coupled_run_writes_matching_frames_and_restarts(tmp_path):
    with _solver(tmp_path) as solver:
        solver.run()
        expected_velocity = solver.fvm_solver.velocity.copy()
        expected_pressure = solver.fvm_solver.kinematic_pressure.copy()

        assert solver.fvm_solver.step == solver.vpm_solver.step == solver.step == 9
        assert solver.time == pytest.approx(0.45)
        assert np.allclose(expected_velocity.mean(axis=0), FREESTREAM_VELOCITY, atol=1e-6)
        assert solver.vpm_solver.particles.n_particles_total == 0

    solution = tmp_path / "solution"
    metadata = json.loads((solution / "coupler_metadata.json").read_text())
    assert metadata["solver"] == "coupler"
    assert metadata["run_status"] == {"status": "completed"}
    assert metadata["state"]["step"] == 9
    assert metadata["state"]["time"] == pytest.approx(0.45)

    backups = sorted(path.name for path in (solution / "coupled").iterdir())
    assert backups == [
        "coupled_000002",
        "coupled_000004",
        "coupled_000006",
        "coupled_000008",
        "coupled_000009",
    ]
    info = json.loads((solution / "coupled/coupled_000002/checkpoint_info.json").read_text())
    assert info["kind"] == "openonda.coupled_backup"
    assert (info["step"], info["fvm_step"], info["vpm_step"]) == (2, 2, 2)
    assert info["fvm_time_step_size"] == pytest.approx(0.05)
    assert info["vpm_time_step_size"] == pytest.approx(0.05)

    # Every particle frame has an FVM frame at exactly the same physical time.
    fvm_times = {repr(time) for time, _ in read_collection(solution / "fvm.pvd")}
    vpm_times = [repr(time) for time, _ in read_collection(solution / "vpm.pvd")]
    assert len(vpm_times) == 7
    assert set(vpm_times) <= fvm_times

    history = (solution / "coupler_diagnostics.jsonl").read_text().splitlines()
    assert [json.loads(row)["step"] for row in history] == list(range(1, 10))

    # Resume from the step-8 backup after removing the final checkpoint.
    restart = tmp_path / "restart"
    restart.mkdir()
    shutil.copytree(solution, restart / "solution")
    shutil.rmtree(restart / "solution/coupled/coupled_000009")
    with _solver(restart) as restored:
        restored.run(start_from="latest")
        np.testing.assert_allclose(
            restored.fvm_solver.velocity, expected_velocity, rtol=0.0, atol=1e-13
        )
        np.testing.assert_allclose(
            restored.fvm_solver.kinematic_pressure, expected_pressure, rtol=0.0, atol=1e-13
        )
    history = (restart / "solution/coupler_diagnostics.jsonl").read_text().splitlines()
    assert [json.loads(row)["step"] for row in history] == list(range(1, 10))


def test_adaptive_coupling_lands_on_events_and_reselects_fvm_steps(tmp_path):
    sample = fvm.LineSampler(
        (-0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        n_points=3,
        file_name="centreline",
        schedule=fvm.EveryTime(0.07),
    )
    particles_output = vpm.Output(schedule=vpm.EveryTime(0.13))
    control = fvm.MaximumCourantTimeStep(maximum=0.9, maximum_time_step_size=0.07)
    flow, particles = uniform_stream_cases(
        tmp_path,
        fvm_time_step_size=0.07,
        vpm_time_step_size=0.14,
        end_time=0.35,
        fvm_samplers=fvm.Samplers((sample,)),
        vpm_output=particles_output,
        fvm_time_step_control=control,
        max_n_particles=50_000,
    )
    config = coupler.CouplerConfig(
        transfer_inset_spacings=0.0,
        blend_end_depth=0.0,
        freestream_velocity=FREESTREAM_VELOCITY,
        backup=coupler.Backup(schedule=coupler.EveryTime(0.1)),
        max_coupling_iterations=1,
    )

    accepted_steps = []
    with coupler.CoupledSolver(flow, particles, config) as solver:
        advance = solver.advance
        fvm_advance = solver.fvm_solver.advance
        fvm_step_sizes = []
        fvm_courant_numbers = []

        def record_fvm_advance(*args, **kwargs):
            fvm_advance(*args, **kwargs)
            fvm_step_sizes.append(solver.fvm_solver._accepted_time_step_size)
            fvm_courant_numbers.append(solver.fvm_solver.max_courant_number)

        def record_advance():
            first_substep = len(fvm_step_sizes)
            advance()
            accepted_steps.append(
                (
                    solver.fvm_solver.time,
                    solver.vpm_solver.time_step_size,
                    tuple(fvm_step_sizes[first_substep:]),
                )
            )

        solver.fvm_solver.advance = record_fvm_advance
        solver.advance = record_advance
        solver.run()

    vpm_metadata = json.loads((tmp_path / "solution/vpm_metadata.json").read_text())
    assert vpm_metadata["run_status"] == {"status": "completed"}
    assert vpm_metadata["state"]["target_step"] is None
    assert vpm_metadata["state"]["target_time"] == pytest.approx(0.35)

    times, vpm_steps, substep_sizes = zip(*accepted_steps, strict=True)
    times = np.asarray(times)
    vpm_steps = np.asarray(vpm_steps)
    for interval, sizes in zip(vpm_steps, substep_sizes, strict=True):
        assert len(sizes) > 0
        assert sum(sizes) == pytest.approx(interval, abs=1.0e-14)
    assert max(map(len, substep_sizes)) > 1
    assert len(np.unique(np.round(fvm_step_sizes, 12))) > 1
    assert max(fvm_courant_numbers) <= control.maximum + 1.0e-12
    for event_time in (0.1, 0.2, 0.3):
        assert np.any(np.isclose(times, event_time, rtol=0.0, atol=1.0e-12))

    with (tmp_path / "samples/centreline.csv").open(newline="") as stream:
        sample_times = {float(row["time"]) for row in csv.DictReader(stream)}
    np.testing.assert_allclose(
        sorted(sample_times), [0.0, 0.07, 0.14, 0.21, 0.28, 0.35], rtol=0.0, atol=1.0e-12
    )
    assert not np.any(np.isclose(times, 0.07, rtol=0.0, atol=1.0e-12))

    backup_times = sorted(
        json.loads((path / "checkpoint_info.json").read_text())["time"]
        for path in (tmp_path / "solution/coupled").iterdir()
    )
    np.testing.assert_allclose(backup_times, [0.1, 0.2, 0.3, 0.35], rtol=0.0, atol=1.0e-12)
    first_backup = json.loads(
        (tmp_path / "solution/coupled/coupled_000001/checkpoint_info.json").read_text()
    )
    assert (first_backup["step"], first_backup["fvm_step"], first_backup["vpm_step"]) == (
        1,
        2,
        1,
    )
    assert first_backup["fvm_time_step_size"] == pytest.approx(substep_sizes[0][-1])
    assert first_backup["vpm_time_step_size"] == pytest.approx(0.1)

    particle_frame_times = [time for time, _ in read_collection(tmp_path / "solution/vpm.pvd")]
    for event_time in (0.13, 0.26):
        assert np.any(np.isclose(particle_frame_times, event_time, rtol=0.0, atol=1.0e-12))

    with coupler.CoupledSolver(flow, particles, config) as restored:
        restored.load_backup(tmp_path / "solution/coupled/coupled_000001")
        assert restored.time == pytest.approx(0.1)
        assert restored.fvm_solver.time_step_size == pytest.approx(substep_sizes[0][-1])
        assert restored.vpm_solver.time_step_size == pytest.approx(0.1)


def test_shared_cfl_step_grows_to_its_configured_maximum(tmp_path):
    control = fvm.MaximumCourantTimeStep(maximum=0.9, maximum_time_step_size=0.1)
    flow, particles = uniform_stream_cases(
        tmp_path,
        fvm_time_step_size=0.05,
        vpm_time_step_size=0.2,
        end_time=0.83,
        fvm_time_step_control=control,
    )
    config = coupler.CouplerConfig(
        transfer_inset_spacings=0.0,
        blend_end_depth=0.0,
        freestream_velocity=FREESTREAM_VELOCITY,
    )
    accepted_time_steps = []
    with coupler.CoupledSolver(flow, particles, config) as solver:
        advance = solver.advance

        def record_advance():
            advance()
            accepted_time_steps.append(solver.fvm_solver._accepted_time_step_size)

        solver.advance = record_advance
        solver.run()

    assert max(accepted_time_steps) > 0.05
    assert max(accepted_time_steps) <= 0.1


def test_particle_strain_shortens_the_coupling_interval(tmp_path):
    with _solver(
        tmp_path,
        end_time=0.36,
        time_step_size=0.05,
        vpm_time_step_size=0.2,
    ) as solver:
        solver._vpm_strain_time_step_limit = lambda: 0.12
        solver._select_coupling_interval()
        assert solver.vpm_time_step_size == pytest.approx(0.12)
        assert solver.vpm_time_step_size <= 0.12

        solver._vpm_strain_time_step_limit = lambda: 0.03
        solver._select_coupling_interval()
        assert solver.vpm_time_step_size == pytest.approx(0.03)
        assert solver.vpm_time_step_size <= 0.03


def test_fixed_step_checkpoint_can_continue_with_cfl_control(tmp_path):
    with _solver(tmp_path, end_time=0.2, time_step_size=0.05) as solver:
        solver.run()
    checkpoint = tmp_path / "solution/coupled/coupled_000002"
    legacy_info_path = checkpoint / "checkpoint_info.json"
    info = json.loads(legacy_info_path.read_text())
    info.pop("fvm_time_step_size")
    info.pop("vpm_time_step_size")
    info["format_version"] = 16
    info["n_fvm_substeps"] = 1
    legacy_info_path.write_text(json.dumps(info, indent=2) + "\n")
    shutil.rmtree(tmp_path / "solution/coupled/coupled_000004")

    adaptive = fvm.MaximumCourantTimeStep(maximum=0.9, maximum_time_step_size=0.05)
    with _solver(
        tmp_path,
        end_time=0.2,
        time_step_size=0.05,
        time_step_control=adaptive,
    ) as resumed:
        resumed.run(start_from="latest")
        assert resumed.time == pytest.approx(0.2)
        assert resumed.fvm_solver.step == resumed.vpm_solver.step == resumed.step == 4


def test_current_backup_rejects_a_changed_cfl_target(tmp_path):
    original_control = fvm.MaximumCourantTimeStep(maximum=0.9, maximum_time_step_size=0.05)
    with _solver(
        tmp_path,
        end_time=0.2,
        time_step_size=0.05,
        time_step_control=original_control,
    ) as solver:
        solver.run()

    changed_control = fvm.MaximumCourantTimeStep(maximum=0.8, maximum_time_step_size=0.05)
    with (
        _solver(
            tmp_path,
            end_time=0.2,
            time_step_size=0.05,
            time_step_control=changed_control,
        ) as resumed,
        pytest.raises(RuntimeError, match="configuration hash"),
    ):
        resumed.load_backup(tmp_path / "solution/coupled/coupled_000004")


def test_time_schedules_need_not_be_multiples_of_the_initial_step(tmp_path):
    flow, particles = uniform_stream_cases(
        tmp_path,
        fvm_time_step_size=0.05,
        end_time=0.3,
        vpm_output=vpm.Output(schedule=vpm.EveryTime(0.15, start_time=0.1)),
    )
    config = coupler.CouplerConfig(
        transfer_inset_spacings=0.0,
        blend_end_depth=0.0,
        backup=coupler.Backup(schedule=coupler.EveryTime(0.2)),
    )
    with coupler.CoupledSolver(flow, particles, config) as solver:
        assert solver.maximum_coupling_time_step_size == pytest.approx(0.05)
