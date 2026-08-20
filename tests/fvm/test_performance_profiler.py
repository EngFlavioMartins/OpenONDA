"""Low-overhead FVM phase-profiler contracts."""

from __future__ import annotations

import errno
import json
from types import SimpleNamespace

import numpy as np

from source.solvers.FVM.config.types import LoggingConfig
from source.solvers.FVM.io.logging import Logging, Timer
from source.solvers.FVM.io.profiling import PerformanceProfiler, numpy_allocation_inventory
from source.solvers.FVM.solve.linear_interface import LinearSolveResult


def _serial_context():
    return SimpleNamespace(rank=0, size=1, is_parallel=False, is_root=True, comm=None)


def _debug_logger(tmp_path):
    return Logging(tmp_path, config=LoggingConfig(mode="debug", console=False))


def test_profiler_writes_phase_linear_and_memory_telemetry(tmp_path):
    logger = _debug_logger(tmp_path)
    profiler = PerformanceProfiler(tmp_path, _serial_context(), logger, enabled=True)
    logger.profiler = profiler
    profiler.begin_step(step=7, time=0.07, time_step_size=0.01)

    profiler.record("Pressure Assembly", 0.2)
    profiler.record("Pressure Assembly", 0.3)
    linear = LinearSolveResult(
        backend="petsc-partitioned",
        method="amg",
        preconditioner="gamg",
        nullspace="constant",
        converged=True,
        reason="converged",
        iterations=5,
        initial_residual=1.0,
        final_residual=1.0e-7,
        setup_seconds=0.1,
        solve_seconds=0.4,
        equation="pressure",
    )
    record = profiler.finish_step(0.8, (linear,))
    logger.close()

    assert record is not None
    assert record["step"] == 7
    assert record["ranks"] == 1
    assembly = next(phase for phase in record["phases"] if phase["name"] == "Pressure Assembly")
    assert assembly["calls"]["max"] == 2
    assert assembly["seconds"]["max"] == 0.5
    assert assembly["critical_path_fraction"] == 0.625
    assert record["linear"][0]["equation"] == "pressure"
    assert record["linear"][0]["solve_seconds"]["max"] == 0.4
    assert record["memory"]["aggregate_rss_end_bytes"] > 0

    written = [json.loads(line) for line in profiler.output_path.read_text().splitlines()]
    assert written == [record]
    log = (tmp_path / "solution/fvm.log").read_text()
    assert "PERFORMANCE PROFILE" in log
    assert "Pressure Assembly" in log


def test_timer_records_phases_without_emitting_standalone_lines(tmp_path):
    logger = _debug_logger(tmp_path)
    profiler = PerformanceProfiler(tmp_path, _serial_context(), logger, enabled=True)
    logger.profiler = profiler
    profiler.begin_step(step=1, time=0.01, time_step_size=0.01)

    Timer.start("Momentum Predictor")
    Timer.log("Momentum Predictor", sink=logger)
    record = profiler.finish_step(0.01)
    logger.close()

    assert record is not None
    phase_names = {phase["name"] for phase in record["phases"]}
    assert "Momentum Predictor" in phase_names
    log = (tmp_path / "solution/fvm.log").read_text()
    assert "  Momentum Predictor          :" not in log
    assert "PERFORMANCE PROFILE" in log


def test_simple_mode_keeps_the_json_record_but_prints_no_table(tmp_path):
    logger = Logging(tmp_path, config=LoggingConfig(mode="simple", console=False))
    profiler = PerformanceProfiler(tmp_path, _serial_context(), logger, enabled=True)
    logger.profiler = profiler
    profiler.begin_step(step=1, time=0.01, time_step_size=0.01)

    profiler.record("Pressure Solve", 0.5)
    record = profiler.finish_step(0.6)
    logger.close()

    assert record is not None
    assert profiler.output_path.is_file()
    assert "PERFORMANCE PROFILE" not in (tmp_path / "solution/fvm.log").read_text()


def test_profiler_keeps_log_tracing_when_json_disk_is_full(tmp_path, monkeypatch):
    logger = _debug_logger(tmp_path)
    profiler = PerformanceProfiler(tmp_path, _serial_context(), logger, enabled=True)
    logger.profiler = profiler

    def disk_full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "test disk full")

    monkeypatch.setattr("source.solvers.FVM.io.profiling.append_line_recoverably", disk_full)
    profiler.begin_step(step=1, time=0.01, time_step_size=0.01)
    profiler.record("Pressure Solve", 0.5)
    record = profiler.finish_step(0.6)
    logger.close()

    assert record is not None
    assert profiler._output_disabled
    log = (tmp_path / "solution/fvm.log").read_text()
    assert "Performance output disabled" in log
    assert "PERFORMANCE PROFILE" in log


def test_numpy_inventory_deduplicates_views_and_shared_references():
    base = np.zeros(40, dtype=np.float64)
    solver = SimpleNamespace(
        mesh_data={"owners": base[:20], "shared": base},
        geo_data={"volume": np.ones(5)},
        velocity=base[20:],
        p=None,
        face_flux=None,
        velocity_old=None,
        velocity_older=None,
        face_flux_old=None,
        face_flux_older=None,
        eddy_viscosity=None,
        turbulence=None,
        algorithm=None,
        _derived_fields={},
        vtk_exporter=None,
        _buffered_vtk_writer=None,
    )

    inventory = numpy_allocation_inventory(solver)

    assert inventory["mesh_topology"] == base.nbytes
    assert inventory["solution_fields_history"] == 0
    assert inventory["geometry"] == 5 * np.dtype(np.float64).itemsize
    assert inventory["numpy_unique_total"] == base.nbytes + 5 * np.dtype(np.float64).itemsize
