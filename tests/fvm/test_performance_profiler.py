"""Low-overhead FVM phase-profiler contracts."""

from __future__ import annotations

import errno
import json
from types import SimpleNamespace

import numpy as np

from source.solvers.FVM.io.logging import Logging, Timer
from source.solvers.FVM.io.profiling import PerformanceProfiler, numpy_allocation_inventory
from source.solvers.FVM.solve.linear_interface import LinearSolveResult


def _serial_context():
    return SimpleNamespace(rank=0, size=1, is_parallel=False, is_root=True, comm=None)


def test_profiler_writes_phase_linear_and_memory_telemetry(tmp_path):
    logger = Logging(tmp_path, console=False)
    profiler = PerformanceProfiler(tmp_path, _serial_context(), logger, enabled=True)
    logger.profiler = profiler
    profiler.begin_step(step=7, flow_time=0.07, dt=0.01)

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


def test_detailed_timer_still_records_when_raw_timing_lines_are_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("FVM_DETAILED_TIMING", raising=False)
    logger = Logging(tmp_path, console=False)
    profiler = PerformanceProfiler(tmp_path, _serial_context(), logger, enabled=True)
    logger.profiler = profiler
    profiler.begin_step(step=1, flow_time=0.01, dt=0.01)

    Timer.start("Momentum Predictor")
    Timer.log("Momentum Predictor", sink=logger, detailed=True)
    record = profiler.finish_step(0.01)
    logger.close()

    assert record is not None
    phase_names = {phase["name"] for phase in record["phases"]}
    assert "Momentum Predictor" in phase_names
    log = (tmp_path / "solution/fvm.log").read_text()
    assert "  Momentum Predictor          :" not in log
    assert "PERFORMANCE PROFILE" in log


def test_profiler_keeps_log_tracing_when_json_disk_is_full(tmp_path, monkeypatch):
    logger = Logging(tmp_path, console=False)
    profiler = PerformanceProfiler(tmp_path, _serial_context(), logger, enabled=True)
    logger.profiler = profiler

    def disk_full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "test disk full")

    monkeypatch.setattr("source.solvers.FVM.io.profiling.append_line_recoverably", disk_full)
    profiler.begin_step(step=1, flow_time=0.01, dt=0.01)
    profiler.record("Pressure Solve", 0.5)
    record = profiler.finish_step(0.6)
    logger.close()

    assert record is not None
    assert profiler._output_disabled
    log = (tmp_path / "solution/fvm.log").read_text()
    assert "Performance JSON output disabled" in log
    assert "PERFORMANCE PROFILE" in log


def test_numpy_inventory_deduplicates_views_and_shared_references():
    base = np.zeros(40, dtype=np.float64)
    solver = SimpleNamespace(
        mesh_data={"owners": base[:20], "shared": base},
        geo_data={"volume": np.ones(5)},
        U=base[20:],
        p=None,
        phi=None,
        U_old=None,
        U_old_old=None,
        phi_old=None,
        phi_old_old=None,
        nut=None,
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
