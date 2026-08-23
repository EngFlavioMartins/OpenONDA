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

from source.solvers.vpm.boundary_elements.vlm.solver.diagnostics import VLMDiagnostics
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver
from source.solvers.vpm.config.types import VPMSetup
from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.diagnostics.conservation import ConservationTracker
from source.solvers.vpm.diagnostics.offline import FlowIntegrals, OfflineFlowDiagnostics
from source.solvers.vpm.io.checkpoint import CheckpointManager
from source.solvers.vpm.io.logging import Logging, _TeeLogStream
from source.solvers.vpm.io.sampler import SamplerExecutor
from source.solvers.vpm.io.sampling import SAMPLER_CSV_COLUMNS
from source.solvers.vpm.particles.container import Particles
from source.solvers.vpm.physics.base import PhysicsBase


def test_output_and_target_configuration_round_trip():
    config = VPMSetup(
        max_evaluation_points=1234,
        export_flow_integrals=False,
        log_mode="tee",
        sample_subdirectory="test_case",
    )

    restored = VPMSetup.from_dict(config.to_dict())

    assert restored.max_evaluation_points == 1234
    assert restored.export_flow_integrals is False
    assert restored.log_mode == "tee"
    assert restored.sample_subdirectory == "test_case"


def test_solver_setup_is_immutable_after_construction():
    setup = VPMSetup()

    with pytest.raises(FrozenInstanceError):
        setup.logging_interval_steps = 5


@pytest.mark.parametrize("name", ["results/run", "run.h5", "vpm_run"])
def test_checkpoint_name_is_a_safe_infix(name):
    with pytest.raises(ValueError, match="filename-safe infix"):
        VPMSetup(checkpoint_name=name)


class _CheckpointParticles:
    n_particles_total = 1

    def __getattr__(self, name):
        if not name.endswith("_cpu"):
            raise AttributeError(name)
        field = name.removesuffix("_cpu")
        if field in {"position", "velocity", "vortex_strength", "vorticity"}:
            return lambda: np.zeros((1, 3), dtype=np.float32)
        if field in {"group_id", "zone_id"}:
            return lambda: np.zeros(1, dtype=np.int32)
        if field in {
            "core_radius",
            "particle_volume",
            "kinematic_viscosity",
            "eddy_viscosity",
            "effective_viscosity",
        }:
            return lambda: np.ones(1, dtype=np.float32)
        if field in {"velocity_gradient", "strain_rate"}:
            return lambda: np.zeros((1, 3, 3), dtype=np.float32)
        raise AttributeError(name)


def test_vpm_snapshot_and_checkpoint_names_are_unambiguous(tmp_path):
    solver = SimpleNamespace(
        time=0.123456789012345,
        step=7,
        time_step_size=0.05,
        particles=_CheckpointParticles(),
        freestream_velocity=np.zeros(3),
        np_dtype=np.float32,
        precision="f32",
    )

    CheckpointManager.write_checkpoint(solver, str(tmp_path / "vpm"))
    snapshot = tmp_path / "vpm_000007.h5"
    assert snapshot.is_file()
    assert (tmp_path / "vpm_000007.xdmf").is_file()
    ET.parse(tmp_path / "vpm_000007.xdmf")
    with h5py.File(snapshot, "r") as handle:
        assert handle["solver"].attrs["time"] == solver.time

    CheckpointManager.write_checkpoint(
        solver,
        str(tmp_path / "checkpoint" / "vpm"),
        append_step=False,
    )
    assert (tmp_path / "checkpoint" / "vpm.h5").is_file()
    assert not (tmp_path / "checkpoint" / "vpm_000007.h5").exists()


@pytest.mark.parametrize("field, value", [("max_evaluation_points", 0), ("max_n_particles", 0)])
def test_fixed_capacity_configuration_rejects_nonpositive_values(field, value):
    with pytest.raises(ValueError, match=field):
        VPMSetup(**{field: value})


def test_target_queries_fail_instead_of_reallocating_taichi_fields():
    fields = SimpleNamespace(_target_field_size=16)

    PhysicsBase._resize_target_fields(fields, 16)
    with pytest.raises(ValueError, match="max_evaluation_points=16"):
        PhysicsBase._resize_target_fields(fields, 17)


def test_particle_capacity_fails_instead_of_reallocating_taichi_fields():
    particles = SimpleNamespace(_max_particles=32)

    Particles._grow_capacity(particles, 32)
    with pytest.raises(ValueError, match="max_n_particles=32"):
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
        setup=SimpleNamespace(samplers=[(sampler, "probe")]),
        particles=SimpleNamespace(n_particles_total=2),
        particle_vortex_strength=np.ones((2, 3)),
        case_dir=tmp_path,
        checkpoint_directory=str(tmp_path),
        time=0.1,
        step=1,
    )

    SamplerExecutor.execute(solver)
    solver.time = 0.2
    solver.step = 2
    SamplerExecutor.execute(solver)

    output = tmp_path / "samples" / "probe.csv"
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))

    assert rows[0] == ["time", "step", *SAMPLER_CSV_COLUMNS]
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
        setup=SimpleNamespace(samplers=[LegacyCSVSampler()]),
        particles=SimpleNamespace(n_particles_total=2),
        particle_vortex_strength=np.ones((2, 3)),
        case_dir=tmp_path,
        checkpoint_directory=str(tmp_path),
        time=0.3,
        step=3,
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
        setup=SimpleNamespace(samplers=[TimeSeriesSampler()]),
        particles=SimpleNamespace(n_particles_total=2),
        particle_vortex_strength=np.ones((2, 3)),
        case_dir=tmp_path,
        checkpoint_directory=str(tmp_path),
        time=0.1,
        step=1,
    )

    SamplerExecutor.execute(solver)
    solver.time = 0.2
    solver.step = 2
    SamplerExecutor.execute(solver)

    with (tmp_path / "samples" / "profile.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))

    assert rows[0] == ["time", "step", *SAMPLER_CSV_COLUMNS]
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
        setup=SimpleNamespace(samplers=[(sampler, "probe")], sample_subdirectory="dipole_cs"),
        particles=SimpleNamespace(n_particles_total=2),
        particle_vortex_strength=np.ones((2, 3)),
        case_dir=tmp_path,
        checkpoint_directory=str(tmp_path / "solution"),
        time=0.1,
        step=1,
    )

    SamplerExecutor.execute(solver)

    assert (tmp_path / "samples" / "dipole_cs" / "probe.csv").is_file()
    assert not (tmp_path / "solution" / "samples").exists()


def test_vlm_diagnostics_use_the_same_sample_subdirectory(tmp_path):
    VLMDiagnostics.export_forces_csv(
        vlm_solver=None,
        forces={"lift_coefficient": 0.5},
        bound_vortex_strength=1.0,
        wake_vortex_strength=-1.0,
        max_leading_edge_suction_parameter=0.0,
        n_p=10,
        time=0.2,
        step=2,
        case_dir=str(tmp_path),
        sample_subdirectory="flat_plate",
    )

    output = tmp_path / "samples" / "flat_plate" / "vlm_forces.csv"
    assert output.is_file()
    assert not (tmp_path / "solution" / "samples").exists()
    with output.open(newline="", encoding="utf-8") as stream:
        columns = next(csv.reader(stream))
    assert "bound_vortex_strength_y" in columns
    assert "wake_vortex_strength_y" in columns
    assert not any(name.startswith("gamma_") for name in columns)


class _ArrayField:
    def __init__(self, values):
        self.values = np.asarray(values)

    def to_numpy(self):
        return self.values.copy()


def test_vlm_bound_vector_strength_includes_oriented_leg_length():
    vortex_points = np.zeros((2, 4, 3))
    vortex_points[0, 2] = [0.0, 1.0, 0.0]
    vortex_points[1, 2] = [0.0, 2.0, 0.0]
    solver = SimpleNamespace(
        _solved=True,
        lattice=SimpleNamespace(
            n_panels=2,
            circulation=_ArrayField([2.0, 3.0]),
            vortex_points=_ArrayField(vortex_points),
        ),
    )

    result = VLMSolver.compute_total_bound_vortex_strength(solver)

    np.testing.assert_allclose(result, [0.0, 8.0, 0.0])
    assert not hasattr(VLMSolver, "compute_total_circulation")


def test_conservation_tracker_compares_dimensionally_equal_vector_strengths(tmp_path):
    vlm_solver = SimpleNamespace(
        _solved=True,
        compute_total_bound_vortex_strength=lambda: np.array([0.0, 8.0, 0.0]),
        compute_forces=lambda **_kwargs: {"force_x": 1.0, "force_y": 2.0, "force_z": 3.0},
    )
    solver = SimpleNamespace(
        time=0.25,
        net_vortex_strength=np.array([0.0, -8.0, 0.0]),
        total_linear_impulse=np.array([1.0, 2.0, 3.0]),
        total_kinetic_energy=4.0,
        kinetic_energy_rate=-0.5,
        particles=SimpleNamespace(n_particles_total=12),
        vlm_solver=vlm_solver,
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
    )
    tracker = ConservationTracker(density=2.0)

    state = tracker.record_state(solver)

    np.testing.assert_allclose(state.net_vortex_strength, 0.0)
    np.testing.assert_allclose(state.impulse_wake, [2.0, 4.0, 6.0])
    assert state.total_kinetic_energy == pytest.approx(8.0)
    assert state.viscous_kinetic_energy_rate == pytest.approx(-1.0)
    assert state.vortex_strength_closure_error_percent == pytest.approx(0.0)

    solver.net_vortex_strength = np.array([0.0, -6.0, 0.0])
    nonclosing_state = tracker.record_state(solver)
    np.testing.assert_allclose(nonclosing_state.net_vortex_strength, [0.0, 2.0, 0.0])
    assert nonclosing_state.vortex_strength_closure_error_percent == pytest.approx(25.0)

    output = tracker.export_csv(tmp_path)
    assert output == tmp_path / "samples" / "vpm_conservation.csv"
    with output.open(newline="", encoding="utf-8") as stream:
        columns = next(csv.reader(stream))
    assert columns[1:5] == [
        "bound_vortex_strength_magnitude",
        "wake_vortex_strength_magnitude",
        "net_vortex_strength_magnitude",
        "vortex_strength_closure_error_percent",
    ]
    tracker.print_summary()


def test_offline_flow_integral_output_uses_vortex_strength_contract(tmp_path):
    diagnostics = OfflineFlowDiagnostics.__new__(OfflineFlowDiagnostics)
    diagnostics.xdmf_path = tmp_path / "vpm_temporal.xdmf"
    diagnostics.base_dir = tmp_path
    diagnostics.results = [
        FlowIntegrals(
            time=0.0,
            total_kinetic_energy=1.0,
            total_helicity=2.0,
            total_enstrophy=3.0,
            viscous_kinetic_energy_rate=4.0,
            vortex_strength_magnitude_sum=5.0,
            net_vortex_strength=np.array([1.0, 2.0, 3.0]),
            linear_impulse=np.array([4.0, 5.0, 6.0]),
            angular_impulse=np.array([7.0, 8.0, 9.0]),
            n_particles_total=2,
        )
    ]

    output = diagnostics.save()
    contents = output.read_text(encoding="utf-8")

    assert "vortex_strength_magnitude_sum" in contents
    assert "Total circulation magnitude" not in contents


def test_offline_flow_integral_reader_uses_canonical_result_key(
    tmp_path,
    minimal_solver_config,
):
    VPMSolver(setup=VPMSetup(**minimal_solver_config))
    diagnostics = OfflineFlowDiagnostics.__new__(OfflineFlowDiagnostics)
    diagnostics._load_particle_data = lambda _path: {
        "time": 0.5,
        "n_particles_total": 1,
        "position": np.zeros((1, 3)),
        "vortex_strength": np.array([[1.0, 2.0, 3.0]]),
        "core_radius": np.ones(1),
        "effective_viscosity": np.zeros(1),
    }
    diagnostics.evaluator = SimpleNamespace(
        compute_flow_integrals=lambda _particles, _time: {
            "total_kinetic_energy": 1.0,
            "total_helicity": 2.0,
            "total_enstrophy": 3.0,
            "viscous_kinetic_energy_rate": 4.0,
            "vortex_strength_magnitude_sum": 5.0,
            "vortex_strength": np.array([1.0, 2.0, 3.0]),
            "linear_impulse": np.array([4.0, 5.0, 6.0]),
            "angular_impulse": np.array([7.0, 8.0, 9.0]),
        }
    )

    result = diagnostics._compute_single_time_step(tmp_path / "vpm_000001.h5")

    np.testing.assert_allclose(result.net_vortex_strength, [1.0, 2.0, 3.0])


def test_flow_integral_export_is_configurable(monkeypatch):
    exports: list[bool] = []
    monkeypatch.setattr(Logging, "flow_diagnostics", lambda _solver: None)
    solver = SimpleNamespace(
        setup=SimpleNamespace(export_flow_integrals=False),
        turbulence_model=None,
        _export_flow_integrals_csv=lambda: exports.append(True),
        _execute_samplers=lambda: None,
    )

    VPMSolver.log_diagnostics(solver)
    assert exports == []

    solver.setup.export_flow_integrals = True
    VPMSolver.log_diagnostics(solver)
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
        setup=SimpleNamespace(log_mode="file"),
        checkpoint_name="wake",
        case_dir=tmp_path,
        checkpoint_directory=str(tmp_path),
    )

    Logging.setup_output_redirection(solver)
    try:
        assert solver.log_file_path == str(tmp_path / "vpm_wake.log")
    finally:
        solver._restore_output_streams()


def test_vpm_log_redirection_closes_previous_owner(tmp_path):
    first = SimpleNamespace(
        setup=SimpleNamespace(log_mode="file"),
        checkpoint_name="first",
        checkpoint_directory=str(tmp_path),
    )
    second = SimpleNamespace(
        setup=SimpleNamespace(log_mode="file"),
        checkpoint_name="second",
        checkpoint_directory=str(tmp_path),
    )

    Logging.setup_output_redirection(first)
    first_handle = first._log_file_handle
    try:
        assert not first_handle.closed

        Logging.setup_output_redirection(second)
        assert first_handle.closed
        assert not second._log_file_handle.closed
        assert second.log_file_path == str(tmp_path / "vpm_second.log")
    finally:
        second._restore_output_streams()

    assert second._log_file_handle.closed
