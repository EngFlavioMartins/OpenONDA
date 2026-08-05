"""Low-overhead, MPI-aware performance and memory telemetry.

The profiler deliberately avoids collectives in timed regions.  Every rank
records local phase totals and samples its own resident set; one gather at the
end of an accepted step produces the critical-path summary written by rank
zero.  This keeps the measurements useful without perturbing Krylov or halo
communication timings with profiler barriers.
"""

from __future__ import annotations

from collections.abc import Iterable
import errno
import json
import os
from pathlib import Path
import resource
import sys
from typing import Any
import weakref

import numpy as np

from .storage import append_line_recoverably

_MIB = 1024.0 * 1024.0


def resident_set_bytes() -> int:
    """Return the process resident set, or zero when the platform cannot."""
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return 0


def peak_resident_set_bytes() -> int:
    """Return the process-wide maximum resident set in bytes."""
    try:
        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return 0
    # Linux and the BSDs report KiB; macOS reports bytes.
    return maximum if sys.platform == "darwin" else maximum * 1024


def _stats(values: Iterable[float | int]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(samples),
        "mean": sum(samples) / len(samples),
        "max": max(samples),
    }


_INVENTORY_SKIP_KEYS = {
    "_logger",
    "_parallel_context",
    "boundaries",
    "geo_data",
    "mesh_data",
    "params",
}


def numpy_allocation_inventory(solver: Any) -> dict[str, int]:
    """Return deduplicated stable NumPy allocation bytes by solver subsystem."""
    seen_allocations: set[int] = set()
    seen_objects: set[int] = set()

    def allocation_bytes(value: Any) -> int:
        if value is None or isinstance(value, str | bytes | int | float | bool):
            return 0
        identity = id(value)
        if identity in seen_objects:
            return 0
        seen_objects.add(identity)
        if isinstance(value, np.ndarray):
            root = value
            while isinstance(root.base, np.ndarray):
                root = root.base
            allocation = id(root)
            if allocation in seen_allocations:
                return 0
            seen_allocations.add(allocation)
            return int(root.nbytes)
        if isinstance(value, dict):
            return sum(
                allocation_bytes(item)
                for key, item in value.items()
                if str(key) not in _INVENTORY_SKIP_KEYS
            )
        if isinstance(value, list | tuple | set):
            return sum(allocation_bytes(item) for item in value)
        module = type(value).__module__
        if module.startswith(("source.solvers.FVM", "scipy.sparse")) and hasattr(value, "__dict__"):
            return sum(
                allocation_bytes(item)
                for key, item in vars(value).items()
                if key not in _INVENTORY_SKIP_KEYS
            )
        return 0

    fields = [
        getattr(solver, name, None)
        for name in (
            "U",
            "p",
            "phi",
            "U_old",
            "U_old_old",
            "phi_old",
            "phi_old_old",
            "nut",
        )
    ]
    sources = (
        ("mesh_topology", getattr(solver, "mesh_data", None)),
        ("geometry", getattr(solver, "geo_data", None)),
        ("solution_fields_history", fields),
        ("turbulence_model", getattr(solver, "turbulence", None)),
        ("linear_algorithm_workspaces", getattr(solver, "algorithm", None)),
        ("derived_diagnostics", getattr(solver, "_derived_fields", None)),
        (
            "output_buffers",
            [
                getattr(solver, "vtk_exporter", None),
                getattr(solver, "_buffered_vtk_writer", None),
            ],
        ),
    )
    inventory = {name: allocation_bytes(value) for name, value in sources}
    inventory["numpy_unique_total"] = sum(inventory.values())
    return inventory


class PerformanceProfiler:
    """Accumulate phase timings locally and publish one MPI summary per step."""

    schema_version = 1

    def __init__(
        self,
        case_dir: str | Path,
        parallel: Any,
        logger: Any,
        *,
        enabled: bool | None = None,
        solver: Any | None = None,
    ) -> None:
        if enabled is None:
            enabled = os.environ.get("FVM_PROFILE", "1") != "0"
        self.enabled = bool(enabled)
        self.parallel = parallel
        self.logger = logger
        self.output_path = Path(case_dir).resolve() / "solution" / "performance.jsonl"
        self._active = False
        self._metadata: dict[str, float | int] = {}
        self._phase_seconds: dict[str, float] = {}
        self._phase_calls: dict[str, int] = {}
        self._phase_rss_end: dict[str, int] = {}
        self._rss_start = 0
        self._rss_max_observed = 0
        self._peak_rss_start = 0
        self._output_disabled = False
        self._solver_ref = None if solver is None else weakref.ref(solver)
        self._inventory_reported = False
        petsc_log = os.environ.get("FVM_PETSC_LOG")
        self._petsc_log_path = None if not petsc_log else Path(petsc_log).resolve()
        if self._petsc_log_path is not None:
            try:
                from petsc4py import PETSc as _PETSc
            except ImportError as error:
                raise RuntimeError("FVM_PETSC_LOG requires petsc4py") from error
            PETSc: Any = _PETSc
            PETSc.Log.begin()

    def begin_step(self, *, step: int, flow_time: float, dt: float) -> None:
        """Reset local counters at the beginning of a time step."""
        if not self.enabled:
            return
        self._active = True
        self._metadata = {
            "step": int(step),
            "time": float(flow_time),
            "dt": float(dt),
        }
        self._phase_seconds.clear()
        self._phase_calls.clear()
        self._phase_rss_end.clear()
        self._rss_start = resident_set_bytes()
        self._rss_max_observed = self._rss_start
        self._peak_rss_start = peak_resident_set_bytes()

    def record(self, name: str, elapsed: float) -> None:
        """Record one completed local phase without communicating."""
        if not self.enabled or not self._active or elapsed <= 0.0:
            return
        phase = str(name).strip(" -")
        self._phase_seconds[phase] = self._phase_seconds.get(phase, 0.0) + float(elapsed)
        self._phase_calls[phase] = self._phase_calls.get(phase, 0) + 1
        rss = resident_set_bytes()
        self._phase_rss_end[phase] = rss
        self._rss_max_observed = max(self._rss_max_observed, rss)

    @staticmethod
    def _linear_payload(linear_results: Iterable[Any]) -> dict[str, dict[str, float | int]]:
        payload: dict[str, dict[str, float | int]] = {}
        for result in linear_results:
            equation = str(getattr(result, "equation", None) or "unknown")
            entry = payload.setdefault(
                equation,
                {"calls": 0, "iterations": 0, "setup_seconds": 0.0, "solve_seconds": 0.0},
            )
            entry["calls"] = int(entry["calls"]) + 1
            entry["iterations"] = int(entry["iterations"]) + int(result.iterations)
            entry["setup_seconds"] = float(entry["setup_seconds"]) + float(result.setup_seconds)
            entry["solve_seconds"] = float(entry["solve_seconds"]) + float(result.solve_seconds)
        return payload

    def _local_payload(
        self,
        step_seconds: float,
        linear_results: Iterable[Any],
    ) -> dict[str, Any]:
        rss_end = resident_set_bytes()
        self._rss_max_observed = max(self._rss_max_observed, rss_end)
        phases = {
            name: {
                "seconds": seconds,
                "calls": self._phase_calls[name],
                "rss_end_bytes": self._phase_rss_end.get(name, 0),
            }
            for name, seconds in self._phase_seconds.items()
        }
        attributed = sum(self._phase_seconds.values())
        phases["Unattributed / profiler gap"] = {
            "seconds": max(0.0, float(step_seconds) - attributed),
            "calls": 1,
            "rss_end_bytes": rss_end,
        }
        inventory = None
        if not self._inventory_reported and self._solver_ref is not None:
            solver = self._solver_ref()
            if solver is not None:
                inventory = numpy_allocation_inventory(solver)
        return {
            "rank": int(self.parallel.rank),
            "step_seconds": float(step_seconds),
            "phases": phases,
            "linear": self._linear_payload(linear_results),
            "memory": {
                "rss_start_bytes": self._rss_start,
                "rss_end_bytes": rss_end,
                "rss_max_observed_bytes": self._rss_max_observed,
                "peak_rss_start_bytes": self._peak_rss_start,
                "peak_rss_end_bytes": peak_resident_set_bytes(),
            },
            "allocation_inventory": inventory,
        }

    def _aggregate(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        step_stats = _stats(payload["step_seconds"] for payload in payloads)
        phase_names = sorted(
            {name for payload in payloads for name in payload["phases"]},
            key=lambda name: (
                name == "Unattributed / profiler gap",
                -max(
                    float(payload["phases"].get(name, {}).get("seconds", 0.0))
                    for payload in payloads
                ),
                name,
            ),
        )
        phases = []
        for name in phase_names:
            seconds = _stats(
                payload["phases"].get(name, {}).get("seconds", 0.0) for payload in payloads
            )
            calls = _stats(payload["phases"].get(name, {}).get("calls", 0) for payload in payloads)
            rss_end = _stats(
                payload["phases"].get(name, {}).get("rss_end_bytes", 0) for payload in payloads
            )
            phases.append(
                {
                    "name": name,
                    "seconds": seconds,
                    "calls": calls,
                    "critical_path_fraction": (
                        seconds["max"] / step_stats["max"] if step_stats["max"] > 0.0 else 0.0
                    ),
                    "rank_imbalance": (
                        seconds["max"] / seconds["mean"] if seconds["mean"] > 0.0 else 0.0
                    ),
                    "rss_end_bytes": rss_end,
                }
            )

        equations = sorted({name for payload in payloads for name in payload["linear"]})
        linear = []
        for equation in equations:
            linear.append(
                {
                    "equation": equation,
                    "calls": _stats(
                        payload["linear"].get(equation, {}).get("calls", 0) for payload in payloads
                    ),
                    "iterations": _stats(
                        payload["linear"].get(equation, {}).get("iterations", 0)
                        for payload in payloads
                    ),
                    "setup_seconds": _stats(
                        payload["linear"].get(equation, {}).get("setup_seconds", 0.0)
                        for payload in payloads
                    ),
                    "solve_seconds": _stats(
                        payload["linear"].get(equation, {}).get("solve_seconds", 0.0)
                        for payload in payloads
                    ),
                }
            )

        memory_keys = (
            "rss_start_bytes",
            "rss_end_bytes",
            "rss_max_observed_bytes",
            "peak_rss_start_bytes",
            "peak_rss_end_bytes",
        )
        memory: dict[str, Any] = {
            name: _stats(payload["memory"][name] for payload in payloads) for name in memory_keys
        }
        memory["aggregate_rss_end_bytes"] = float(
            sum(payload["memory"]["rss_end_bytes"] for payload in payloads)
        )
        memory["aggregate_rss_max_observed_bytes"] = float(
            sum(payload["memory"]["rss_max_observed_bytes"] for payload in payloads)
        )
        memory["aggregate_peak_rss_end_bytes"] = float(
            sum(payload["memory"]["peak_rss_end_bytes"] for payload in payloads)
        )

        allocation_inventory = None
        if all(payload.get("allocation_inventory") is not None for payload in payloads):
            names = sorted(payloads[0]["allocation_inventory"])
            allocation_inventory = {
                name: {
                    "per_rank_bytes": _stats(
                        payload["allocation_inventory"][name] for payload in payloads
                    ),
                    "aggregate_bytes": float(
                        sum(payload["allocation_inventory"][name] for payload in payloads)
                    ),
                }
                for name in names
            }
            numpy_total = allocation_inventory["numpy_unique_total"]["aggregate_bytes"]
            allocation_inventory["native_python_petsc_rss_remainder"] = {
                "aggregate_bytes": max(
                    0.0,
                    memory["aggregate_rss_end_bytes"] - numpy_total,
                )
            }

        return {
            "schema_version": self.schema_version,
            **self._metadata,
            "ranks": len(payloads),
            "step_seconds": step_stats,
            "phases": phases,
            "linear": linear,
            "memory": memory,
            "allocation_inventory": allocation_inventory,
        }

    def _render(self, record: dict[str, Any]) -> str:
        lines = [
            "",
            "-" * 88,
            "PERFORMANCE PROFILE (phase max is the MPI critical path)",
            "-" * 88,
            f"  {'Phase':<31} {'calls':>5} {'mean s':>10} {'max s':>10} {'step %':>8} {'imb.':>7}",
        ]
        for phase in record["phases"]:
            lines.append(
                f"  {phase['name']:<31} "
                f"{phase['calls']['max']:5.0f} "
                f"{phase['seconds']['mean']:10.3f} "
                f"{phase['seconds']['max']:10.3f} "
                f"{100.0 * phase['critical_path_fraction']:7.1f}% "
                f"{phase['rank_imbalance']:7.3f}"
            )
        if record["linear"]:
            lines.extend(["-" * 88, "  Linear solvers (included in phases above)"])
            for linear in record["linear"]:
                lines.append(
                    f"  {linear['equation']:<31} "
                    f"calls={linear['calls']['max']:.0f}, "
                    f"iterations(max)={linear['iterations']['max']:.0f}, "
                    f"setup(max)={linear['setup_seconds']['max']:.3f} s, "
                    f"solve(max)={linear['solve_seconds']['max']:.3f} s"
                )
        memory = record["memory"]
        lines.extend(
            [
                "-" * 88,
                f"  Aggregate RSS now           : {memory['aggregate_rss_end_bytes'] / _MIB:.1f} MiB",
                f"  Aggregate RSS max observed  : {memory['aggregate_rss_max_observed_bytes'] / _MIB:.1f} MiB",
                f"  Aggregate process peak RSS  : {memory['aggregate_peak_rss_end_bytes'] / _MIB:.1f} MiB",
                f"  Slowest rank step           : {record['step_seconds']['max']:.3f} s",
            ]
        )
        inventory = record.get("allocation_inventory")
        if inventory is not None:
            lines.extend(["-" * 88, "  Stable allocation inventory"])
            for name, values in inventory.items():
                aggregate = values["aggregate_bytes"]
                lines.append(f"  {name:<31}: {aggregate / _MIB:.1f} MiB aggregate")
        lines.append("-" * 88)
        return "\n".join(lines)

    def finish_step(
        self,
        step_seconds: float,
        linear_results: Iterable[Any] = (),
    ) -> dict[str, Any] | None:
        """Gather local counters, write JSONL, and emit the rank-zero table."""
        if not self.enabled or not self._active:
            return None
        local = self._local_payload(step_seconds, linear_results)
        if self.parallel.is_parallel:
            payloads = self.parallel.comm.gather(local, root=0)
        else:
            payloads = [local]
        self._active = False
        self._inventory_reported = True
        if not self.parallel.is_root:
            return None
        record = self._aggregate(payloads)
        if not self._output_disabled:
            line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            try:
                append_line_recoverably(self.output_path, line)
            except OSError as error:
                if error.errno != errno.ENOSPC:
                    raise
                self._output_disabled = True
                self.logger.warning(
                    f"Performance JSON output disabled after disk-full error at "
                    f"{self.output_path}; log profiling will continue"
                )
        self.logger.message(self._render(record), flush=True)
        return record

    def close(self) -> None:
        """Collectively publish the optional PETSc event log."""
        if self._petsc_log_path is None:
            return
        from petsc4py import PETSc as _PETSc

        PETSc: Any = _PETSc

        if self.parallel.is_root:
            self._petsc_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.parallel.barrier()
        viewer = PETSc.Viewer().createASCII(
            str(self._petsc_log_path),
            mode="w",
            comm=PETSc.COMM_WORLD,
        )
        try:
            PETSc.Log.view(viewer)
            viewer.flush()
        finally:
            viewer.destroy()
        self._petsc_log_path = None
