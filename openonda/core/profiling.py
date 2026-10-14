"""Wall-clock timing of solver sections, with optional MPI and memory reports.

The solvers time named sections of a time step with :meth:`Profiler.section`.
GPU kernels run asynchronously, so a profiler given a ``synchronize``
function (for example ``taichi.sync``) waits for the device before and after
each timed section. With MPI, :meth:`Profiler.end_step` gathers the section
times and the resident memory of every rank to rank 0, which appends one
JSON line per step to ``output_path``. A disabled profiler still measures
the wall-clock time of each step, without device synchronization, MPI
communication or file output.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
import errno
import logging
import os
from pathlib import Path
import resource
import sys
import time as clock
from typing import Any, Protocol

from . import log_style
from .file_output import append_jsonl

__all__ = [
    "ParallelRanks",
    "Profiler",
    "peak_resident_memory_bytes",
    "resident_memory_bytes",
]

_LOGGER = logging.getLogger("openonda")
_DISABLED_SECTION = nullcontext()
_UNTIMED = "untimed"


class ParallelRanks(Protocol):
    """MPI rank information used by the profiler (``ParallelContext``)."""

    @property
    def rank(self) -> int:
        """Rank of this process."""
        ...

    @property
    def is_parallel(self) -> bool:
        """Whether more than one rank runs."""
        ...

    @property
    def is_root(self) -> bool:
        """Whether this process is rank 0."""
        ...

    @property
    def comm(self) -> Any:
        """MPI communicator with ``gather(value, root=0)``."""
        ...


def peak_resident_memory_bytes() -> int:
    """Return the largest resident memory of this process so far, in bytes."""
    try:
        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return 0
    # Linux reports KiB, macOS bytes.
    return maximum if sys.platform == "darwin" else maximum * 1024


def resident_memory_bytes() -> int:
    """Return the current resident memory of this process, in bytes.

    Without ``/proc`` (macOS) the peak resident memory is returned.
    """
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return peak_resident_memory_bytes()


def _statistics(values: Iterable[float]) -> dict[str, float]:
    """Return the minimum, mean and maximum over the ranks."""
    samples = [float(value) for value in values]
    if not samples:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {"min": min(samples), "mean": sum(samples) / len(samples), "max": max(samples)}


class Profiler:
    """Accumulate the wall-clock time of named solver sections.

    Parameters
    ----------
    enabled : bool
        Time sections. When false, :meth:`section` returns a no-op context
        and only the whole-step wall-clock time is measured.
    synchronize : callable or None, default=None
        Zero-argument function that waits for the compute device, called
        before and after every timed section and step when enabled.
    parallel : ParallelRanks or None, default=None
        MPI ranks. When given and parallel, each step's section times and
        memory are gathered to rank 0.
    output_path : str or pathlib.Path or None, default=None
        JSON-lines file to which rank 0 appends one record per step when
        enabled, for example ``solution/fvm_performance.jsonl``.

    Notes
    -----
    Timed sections contain no MPI communication; the only collective call is
    the gather in :meth:`end_step`, which every rank must call when enabled.
    Sections may be nested; nested times are counted in both sections.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        synchronize: Callable[[], None] | None = None,
        parallel: ParallelRanks | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._synchronize = synchronize
        self._parallel = parallel
        self.output_path = None if output_path is None else Path(output_path)
        self._output_stopped = False
        self._step_metadata: dict[str, float | int] = {}
        self._step_start: float | None = None
        self._step_sections: dict[str, float] = {}
        self._step_calls: dict[str, int] = {}
        self._memory_start = 0
        self._memory_max = 0
        self._total_sections: dict[str, float] = {}
        self._total_calls: dict[str, int] = {}
        self.step_seconds = 0.0
        self.total_seconds = 0.0
        self.n_steps = 0

    @property
    def step_sections(self) -> dict[str, float]:
        """Section times of the current or last step, in s."""
        return dict(self._step_sections)

    def _wait_for_device(self) -> None:
        if self._synchronize is not None:
            self._synchronize()

    @contextmanager
    def _timed_section(self, name: str) -> Iterator[None]:
        self._wait_for_device()
        start = clock.perf_counter()
        try:
            yield
        finally:
            self._wait_for_device()
            self.record(name, clock.perf_counter() - start)

    def section(self, name: str) -> AbstractContextManager[None]:
        """Time a named section of the current step.

        Parameters
        ----------
        name : str
            Section name, for example ``"pressure solve"``.

        Returns
        -------
        contextlib.AbstractContextManager
            Context that records the elapsed wall-clock time on exit; a
            shared no-op context when the profiler is disabled.
        """
        if not self.enabled:
            return _DISABLED_SECTION
        return self._timed_section(name)

    def record(self, name: str, seconds: float) -> None:
        """Add an externally measured section time in s to the current step."""
        if not self.enabled:
            return
        self._step_sections[name] = self._step_sections.get(name, 0.0) + float(seconds)
        self._step_calls[name] = self._step_calls.get(name, 0) + 1
        self._total_sections[name] = self._total_sections.get(name, 0.0) + float(seconds)
        self._total_calls[name] = self._total_calls.get(name, 0) + 1
        if self._step_start is not None:
            self._memory_max = max(self._memory_max, resident_memory_bytes())

    def begin_step(self, step: int, time: float, time_step_size: float) -> None:
        """Start timing a time step.

        Parameters
        ----------
        step : int
            Index of the step being computed.
        time : float
            Solution time at the start of the step, in s.
        time_step_size : float
            Size of the step, in s.
        """
        self._step_metadata = {
            "step": int(step),
            "time": float(time),
            "time_step_size": float(time_step_size),
        }
        self._step_sections.clear()
        self._step_calls.clear()
        if self.enabled:
            self._wait_for_device()
            self._memory_start = resident_memory_bytes()
            self._memory_max = self._memory_start
        self._step_start = clock.perf_counter()

    def end_step(self, *, details: Mapping[str, float] | None = None) -> dict[str, Any] | None:
        """Finish timing the step and write its performance record.

        When enabled and parallel, every rank must call this method.

        Parameters
        ----------
        details : mapping of str to float or None, default=None
            Additional per-rank numbers of the step, for example linear
            solver iterations, reported as minimum, mean and maximum over the
            ranks.

        Returns
        -------
        dict or None
            On rank 0 of an enabled profiler, the step record: step, time,
            time-step size, number of ranks and the minimum, mean and maximum
            over the ranks of the step time, of each section time, of the
            details and of the resident memory. ``None`` otherwise.

        Raises
        ------
        RuntimeError
            If no step was started with :meth:`begin_step`.
        """
        if self._step_start is None:
            raise RuntimeError("end_step called without begin_step")
        if self.enabled:
            self._wait_for_device()
        self.step_seconds = clock.perf_counter() - self._step_start
        self._step_start = None
        self.total_seconds += self.step_seconds
        self.n_steps += 1
        if not self.enabled:
            return None

        memory_end = resident_memory_bytes()
        local = {
            "step_seconds": self.step_seconds,
            "sections": dict(self._step_sections),
            "calls": dict(self._step_calls),
            "details": {name: float(value) for name, value in (details or {}).items()},
            "memory": {
                "resident_start_bytes": self._memory_start,
                "resident_end_bytes": memory_end,
                "resident_max_bytes": max(self._memory_max, memory_end),
                "peak_resident_bytes": peak_resident_memory_bytes(),
            },
        }
        parallel = self._parallel
        if parallel is not None and parallel.is_parallel:
            ranks = parallel.comm.gather(local, root=0)
            if not parallel.is_root:
                return None
        else:
            ranks = [local]

        step = _statistics(rank["step_seconds"] for rank in ranks)
        names = sorted(
            {name for rank in ranks for name in rank["sections"]},
            key=lambda name: (-max(rank["sections"].get(name, 0.0) for rank in ranks), name),
        )
        sections = {}
        for name in names:
            seconds = _statistics(rank["sections"].get(name, 0.0) for rank in ranks)
            sections[name] = {
                "seconds": seconds,
                "calls": _statistics(rank["calls"].get(name, 0) for rank in ranks),
                "fraction_of_step": seconds["max"] / step["max"] if step["max"] > 0.0 else 0.0,
            }
        sections[_UNTIMED] = {
            "seconds": _statistics(
                max(0.0, rank["step_seconds"] - sum(rank["sections"].values())) for rank in ranks
            )
        }
        detail_names = sorted({name for rank in ranks for name in rank["details"]})
        memory = {
            name: _statistics(rank["memory"][name] for rank in ranks)
            for name in sorted(local["memory"])
        }
        memory["total_resident_end_bytes"] = float(
            sum(rank["memory"]["resident_end_bytes"] for rank in ranks)
        )
        record = {
            **self._step_metadata,
            "n_ranks": len(ranks),
            "step_seconds": step,
            "sections": sections,
            "details": {
                name: _statistics(rank["details"].get(name, 0.0) for rank in ranks)
                for name in detail_names
            },
            "memory": memory,
        }
        if self.output_path is not None and not self._output_stopped:
            try:
                append_jsonl(self.output_path, record)
            except OSError as error:
                if error.errno != errno.ENOSPC:
                    raise
                self._output_stopped = True
                _LOGGER.warning(f"Performance output stopped, the disk is full: {self.output_path}")
        return record

    def report(self, title: str = "runtime profile") -> log_style.FormattedText:
        """Return the cumulative timing report of all completed steps.

        Parameters
        ----------
        title : str, default="runtime profile"
            Report title, for example ``"VPM runtime profile"``.

        Returns
        -------
        log_style.FormattedText
            Report with the step count, total and mean step time, and per
            section the calls, total and mean time and share of step time.
        """
        rows: list[log_style.Row] = [
            ("steps", self.n_steps),
            ("step time, total", self.total_seconds, "s"),
        ]
        if self.n_steps:
            rows.append(("step time, mean", self.total_seconds / self.n_steps, "s"))
        sections: list[tuple[str, list[log_style.Row]]] = [("timing", rows)]
        for name, total in sorted(self._total_sections.items(), key=lambda item: -item[1]):
            calls = self._total_calls[name]
            share = 100.0 * total / self.total_seconds if self.total_seconds > 0.0 else 0.0
            sections.append(
                (
                    name,
                    [
                        ("calls", calls),
                        ("total", total, "s"),
                        ("mean", total / calls, "s"),
                        ("share of step time", share, "%"),
                    ],
                )
            )
        return log_style.block_report(title, sections)
