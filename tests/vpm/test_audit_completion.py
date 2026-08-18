"""Regression tests for the VPM memory and output audit."""

from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

from source.solvers.VPM.boundary_elements.vlm.solver.diagnostics import VLMDiagnostics
from source.solvers.VPM.config.types import VPMSetup
from source.solvers.VPM.core.solver import Solver
from source.solvers.VPM.io.backup import BackupSystem
from source.solvers.VPM.io.logging import Logging, _TeeLogStream
from source.solvers.VPM.io.sampler import SamplerExecutor
from source.solvers.VPM.io.sampling import SAMPLER_CSV_COLUMNS
from source.solvers.VPM.particles.container import Particles
from source.solvers.VPM.physics.base import PhysicsBase


def test_output_and_target_configuration_round_trip():
    config = VPMSetup(
        max_targets=1234,
        export_flow_integrals=False,
        log_mode="tee",
        sample_subdirectory="test_case",
    )

    restored = VPMSetup.from_dict(config.to_dict())

    assert restored.max_targets == 1234
    assert restored.export_flow_integrals is False
    assert restored.log_mode == "tee"
    assert restored.sample_subdirectory == "test_case"


def test_solver_setup_is_immutable_after_construction():
    setup = VPMSetup()

    with pytest.raises(FrozenInstanceError):
        setup.logging_frequency = 5


@pytest.mark.parametrize("name", ["results/run", "run.h5", "vpm_run"])
def test_backup_name_is_a_safe_infix(name):
    with pytest.raises(ValueError, match="filename-safe infix"):
        VPMSetup(backup_file_name=name)


class _BackupParticles:
    number_of_particles = 1

    def __getattr__(self, name):
        if not name.endswith("_cpu"):
            raise AttributeError(name)
        field = name.removesuffix("_cpu")
        if field in {"position", "velocity", "circulation", "vorticity"}:
            return lambda: np.zeros((1, 3), dtype=np.float32)
        if field == "group_id":
            return lambda: np.zeros(1, dtype=np.int32)
        if field in {"radius", "volume", "viscosity", "viscosity_turbulent"}:
            return lambda: np.ones(1, dtype=np.float32)
        raise AttributeError(name)


def test_vpm_snapshot_and_checkpoint_names_are_unambiguous(tmp_path):
    solver = SimpleNamespace(
        flow_time=0.123456789012345,
        time_step=7,
        time_step_size=0.05,
        particles=_BackupParticles(),
        freestream_velocity=np.zeros(3),
    )

    BackupSystem.backup_solver(solver, str(tmp_path / "vpm"))
    snapshot = tmp_path / "vpm_000007.h5"
    assert snapshot.is_file()
    assert (tmp_path / "vpm_000007.xdmf").is_file()
    ET.parse(tmp_path / "vpm_000007.xdmf")
    with h5py.File(snapshot, "r") as handle:
        assert handle["solver"].attrs["flow_time"] == solver.flow_time

    BackupSystem.backup_solver(
        solver,
        str(tmp_path / "checkpoint" / "vpm"),
        append_step=False,
    )
    assert (tmp_path / "checkpoint" / "vpm.h5").is_file()
    assert not (tmp_path / "checkpoint" / "vpm_000007.h5").exists()


@pytest.mark.parametrize("field, value", [("max_targets", 0), ("max_particles", 0)])
def test_fixed_capacity_configuration_rejects_nonpositive_values(field, value):
    with pytest.raises(ValueError, match=field):
        VPMSetup(**{field: value})


def test_target_queries_fail_instead_of_reallocating_taichi_fields():
    fields = SimpleNamespace(_target_field_size=16)

    PhysicsBase._resize_target_fields(fields, 16)
    with pytest.raises(ValueError, match="max_targets=16"):
        PhysicsBase._resize_target_fields(fields, 17)


def test_particle_capacity_fails_instead_of_reallocating_taichi_fields():
    particles = SimpleNamespace(_max_particles=32)

    Particles._grow_capacity(particles, 32)
    with pytest.raises(ValueError, match="max_particles=32"):
        Particles._grow_capacity(particles, 33)


class _Sampler:
    def sample(self, _solver):
        return {
            name: np.array([index, index + 0.5], dtype=float)
            for index, name in enumerate(SAMPLER_CSV_COLUMNS)
        }


def test_sampler_csv_appends_all_events_to_one_time_aware_file(tmp_path):
    sampler = _Sampler()
    solver = SimpleNamespace(
        config=SimpleNamespace(samplers=[(sampler, "probe")]),
        particles=SimpleNamespace(number_of_particles=2),
        particles_circulation=np.ones((2, 3)),
        backup_directory=str(tmp_path),
        flow_time=0.1,
        time_step=1,
    )

    SamplerExecutor.execute(solver)
    solver.flow_time = 0.2
    solver.time_step = 2
    SamplerExecutor.execute(solver)

    output = tmp_path / "samples" / "probe.csv"
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))

    assert rows[0] == ["flow_time", "time_step", *SAMPLER_CSV_COLUMNS]
    assert len(rows) == 5
    assert [row[:2] for row in rows[1:]] == [
        ["0.1", "1"],
        ["0.1", "1"],
        ["0.2", "2"],
        ["0.2", "2"],
    ]
    assert list((tmp_path / "samples").iterdir()) == [output]


def test_sampler_executor_supports_csv_samplers_without_step_keyword(tmp_path):
    class LegacyCSVSampler:
        file_name = "profile"

        def save_csv(self, _solver, filepath, time=None):
            Path(filepath).write_text(f"time={time}\n", encoding="utf-8")

    solver = SimpleNamespace(
        config=SimpleNamespace(samplers=[LegacyCSVSampler()]),
        particles=SimpleNamespace(number_of_particles=2),
        particles_circulation=np.ones((2, 3)),
        backup_directory=str(tmp_path),
        flow_time=0.3,
        time_step=3,
    )

    SamplerExecutor.execute(solver)

    assert (tmp_path / "samples" / "profile.csv").read_text(encoding="utf-8") == "time=0.3\n"


def test_sampler_executor_appends_opted_in_csv_time_series(tmp_path):
    class TimeSeriesSampler:
        file_name = "profile"
        csv_time_series = True

        def sample(self, _solver):
            return {
                name: np.asarray([index, index + 0.5])
                for index, name in enumerate(SAMPLER_CSV_COLUMNS)
            }

        def save_csv(self, *_args, **_kwargs):
            raise AssertionError("time-series samplers must use the executor's append path")

    solver = SimpleNamespace(
        config=SimpleNamespace(samplers=[TimeSeriesSampler()]),
        particles=SimpleNamespace(number_of_particles=2),
        particles_circulation=np.ones((2, 3)),
        backup_directory=str(tmp_path),
        flow_time=0.1,
        time_step=1,
    )

    SamplerExecutor.execute(solver)
    solver.flow_time = 0.2
    solver.time_step = 2
    SamplerExecutor.execute(solver)

    with (tmp_path / "samples" / "profile.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))

    assert rows[0] == ["flow_time", "time_step", *SAMPLER_CSV_COLUMNS]
    assert [row[:2] for row in rows[1:]] == [
        ["0.1", "1"],
        ["0.1", "1"],
        ["0.2", "2"],
        ["0.2", "2"],
    ]


def test_surface_sampler_preserves_and_replaces_pvd_steps_after_restart(tmp_path):
    pvd_path = tmp_path / "surface.pvd"
    SamplerExecutor._write_pvd(tmp_path, "surface", [(0.1, "surface_000001.vts")])

    entries = SamplerExecutor._read_pvd(tmp_path, "surface")
    entries = [entry for entry in entries if entry[1] != "surface_000001.vts"]
    entries.append((0.1, "surface_000001.vts"))
    SamplerExecutor._write_pvd(tmp_path, "surface", entries)

    datasets = ET.parse(pvd_path).getroot().findall(".//DataSet")
    assert [(item.get("timestep"), item.get("file")) for item in datasets] == [
        ("0.1", "surface_000001.vts")
    ]


def test_sampler_subdirectory_stays_below_the_root_samples_directory(tmp_path):
    sampler = _Sampler()
    solver = SimpleNamespace(
        config=SimpleNamespace(samplers=[(sampler, "probe")], sample_subdirectory="dipole_cs"),
        particles=SimpleNamespace(number_of_particles=2),
        particles_circulation=np.ones((2, 3)),
        backup_directory=str(tmp_path / "solution"),
        flow_time=0.1,
        time_step=1,
    )

    SamplerExecutor.execute(solver)

    assert (tmp_path / "samples" / "dipole_cs" / "probe.csv").is_file()
    assert not (tmp_path / "solution" / "samples").exists()


def test_vlm_diagnostics_use_the_same_sample_subdirectory(tmp_path):
    VLMDiagnostics.export_forces_csv(
        vlm_solver=None,
        forces={"CL": 0.5},
        gamma_bound=1.0,
        gamma_wake=-1.0,
        lesp_max=0.0,
        n_p=10,
        flow_time=0.2,
        time_step=2,
        backup_directory=str(tmp_path / "solution"),
        sample_subdirectory="flat_plate",
    )

    output = tmp_path / "samples" / "flat_plate" / "vlm_forces.csv"
    assert output.is_file()
    assert not (tmp_path / "solution" / "samples").exists()


def test_flow_integral_export_is_configurable(monkeypatch):
    exports: list[bool] = []
    monkeypatch.setattr(Logging, "flow_diagnostics", lambda _solver: None)
    solver = SimpleNamespace(
        config=SimpleNamespace(export_flow_integrals=False),
        LES=None,
        _export_flow_integrals_csv=lambda: exports.append(True),
        _execute_samplers=lambda: None,
    )

    Solver.log_diagnostics(solver)
    assert exports == []

    solver.config.export_flow_integrals = True
    Solver.log_diagnostics(solver)
    assert exports == [True]


def test_tee_log_stream_writes_to_file_and_console():
    logfile = StringIO()
    console = StringIO()
    stream = _TeeLogStream(logfile, console)

    assert stream.write("visible in both\n") == len("visible in both\n")
    assert logfile.getvalue() == "visible in both\n"
    assert console.getvalue() == "visible in both\n"


def test_log_name_uses_snapshot_prefix(tmp_path):
    solver = SimpleNamespace(
        config=SimpleNamespace(log_mode="file"),
        backup_file_name="wake",
        backup_directory=str(tmp_path),
    )

    Logging.setup_output_redirection(solver)
    try:
        assert solver.log_file_path == str(tmp_path / "vpm_wake.log")
    finally:
        solver._restore_output_streams()
