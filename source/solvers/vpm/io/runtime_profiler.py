"""
Runtime wall-clock profiler for the VPM solver.
=================================================
Measures, accumulates, and reports the wall-clock time spent in the main solver
stages and optional sub-solvers, without affecting numerical behaviour.

The profiler times named sections with a context manager, accumulates statistics
across time steps, and emits per-step and cumulative reports through the central
:class:`Logging` sink (so output style matches the rest of the solver).

Taichi kernels execute asynchronously, so every timed region synchronises the
backend before and after measurement via the ``sync`` callable (default
``taichi.sync``).  This keeps the measurement honest and is backend-safe for
CPU / CUDA / Vulkan / Metal.  When the profiler is disabled, :meth:`section`
returns a shared no-op context with no sync, timing, or bookkeeping overhead.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from collections.abc import Callable
from contextlib import nullcontext
import time

from .logging import Logging

# Shared no-op context for the disabled fast-path. ``nullcontext`` is stateless
# and reentrant, so a single instance can back every disabled ``section`` call.
_NULLCTX = nullcontext()


class _Section:
    """Internal context manager timing one named section (sync-wrapped)."""

    __slots__ = ("_profiler", "_name", "_t0")

    def __init__(self, profiler: "RuntimeProfiler", name: str) -> None:
        self._profiler = profiler
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> "_Section":
        self._profiler._synchronize()  # drain prior async GPU work
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self._profiler._synchronize()  # wait for this section's kernels
        self._profiler._record(self._name, time.perf_counter() - self._t0)
        return False


class _Step:
    """Internal context manager timing one full solver step."""

    __slots__ = ("_profiler", "_t0")

    def __init__(self, profiler: "RuntimeProfiler") -> None:
        self._profiler = profiler
        self._t0 = 0.0

    def __enter__(self) -> "_Step":
        self._profiler._last.clear()  # per-step breakdown is for this step only
        # In production the whole-step number is deliberately a cheap host-side
        # wall-clock metric.  Do not serialize asynchronous GPU work unless the
        # user explicitly requested a physically accurate detailed profile.
        if self._profiler.enabled:
            self._profiler._synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        profiler = self._profiler
        if profiler.enabled:
            profiler._synchronize()
        profiler.step_time = time.perf_counter() - self._t0
        profiler.wall_time += profiler.step_time
        profiler.n_steps += 1
        return False


class RuntimeProfiler:
    """Accumulating wall-clock profiler for the VPM solver.

    Parameters
    ----------
    enabled : bool
        When ``False``, :meth:`section` is a zero-overhead no-op.  The
        whole-step timer remains as a cheap host-side measurement and performs
        no device synchronization.
    sync : Callable[[], None] | None
        Backend synchronisation hook called before and after each timed region.
        Pass ``taichi.sync`` for correct GPU timing; ``None`` disables syncing
        (e.g. pure-CPU unit tests).

    Notes
    -----
    Section labels are recorded in first-seen order; the cumulative report sorts
    them by total time.  All statistics live in plain dicts — the profiler holds
    no GPU state and never transfers particle data.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        detailed: bool = False,
        sync: Callable[[], None] | None = None,
    ) -> None:
        """Create an accumulating host-side timing profiler.

        Parameters
        ----------
        enabled : bool, default=True
            Enable named section measurements. Disabled sections return a
            shared no-op context and do not synchronize the compute backend.
        detailed : bool, default=False
            Retain per-section detail in reports when true; the whole-step
            counters are maintained in either mode.
        sync : callable or None, optional
            Zero-argument backend synchronization hook, such as
            ``taichi.sync``. It is called around timed sections when supplied;
            ``None`` is appropriate for CPU-only tests.

        Notes
        -----
        All durations are wall-clock seconds. The profiler owns only Python
        dictionaries and counters; it neither copies particle arrays nor
        changes solver state.
        """
        self.enabled = enabled
        self.detailed = detailed
        self._sync = sync
        # name -> cumulative seconds / call count / most-recent (this-step) seconds
        self._cumulative: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._last: dict[str, float] = {}
        self.wall_time = 0.0  # cumulative full-step wall time [s]
        self.step_time = 0.0  # most-recent full-step wall time [s]
        self.n_steps = 0
        self.particle_count: int | None = None

    # -- synchronisation --------------------------------------------------------
    def _synchronize(self) -> None:
        if self._sync is not None:
            self._sync()

    # -- measurement ------------------------------------------------------------
    def section(self, name: str):
        """Time a named section. Returns a no-op context when disabled.

        >>> with profiler.section("velocity"):
                ...     solver.stage_rhs.evaluate(...)
        """
        if not self.enabled:
            return _NULLCTX
        return _Section(self, name)

    def step(self):
        """Time one full solver step (use around the body of ``advance``)."""
        return _Step(self)

    def _record(self, name: str, time_step_size: float) -> None:
        self._cumulative[name] = self._cumulative.get(name, 0.0) + time_step_size
        self._calls[name] = self._calls.get(name, 0) + 1
        self._last[name] = time_step_size

    # -- lifecycle ---------------------------------------------------------------
    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._cumulative.clear()
        self._calls.clear()
        self._last.clear()
        self.wall_time = 0.0
        self.step_time = 0.0
        self.n_steps = 0
        self.particle_count = None

    def set_particle_count(self, count: int | None) -> None:
        """Store the particle count to include in cumulative timing reports."""
        self.particle_count = None if count is None else int(count)

    # -- reporting ----------------------------------------------------------------
    def report_step(self) -> None:
        """Print the just-completed step's time and optional detailed breakdown.

        Reuses :meth:`Logging.step_timing` so the per-step console style is
        unchanged. The optional breakdown is controlled by ``detailed``.
        """
        detailed = self._last if self.detailed else None
        Logging.step_timing(self.step_time, detailed)

    def format_report(self) -> list[str]:
        """Render cumulative host timers in the common quantity/unit layout."""
        from source import log_style

        rows = [("steps", self.n_steps), ("step time, total", self.wall_time, "s")]
        if self.n_steps:
            rows.append(("step time, mean", self.wall_time / self.n_steps, "s"))
        if self.particle_count is not None:
            rows.append(("active particles", self.particle_count))
        if self._cumulative:
            measured = sum(self._cumulative.values())
            rows.extend(
                [
                    ("profiled section time", measured, "s"),
                    ("unprofiled time", self.wall_time - measured, "s"),
                ]
            )
        sections = [("timing", rows)]
        for name, total in sorted(self._cumulative.items(), key=lambda item: item[1], reverse=True):
            calls = self._calls[name]
            sections.append(
                (
                    name,
                    [
                        ("calls", calls),
                        ("total", total, "s"),
                        ("mean", total / calls if calls else 0.0, "s"),
                        (
                            "share of step time",
                            100.0 * total / self.wall_time if self.wall_time else 0.0,
                            "%",
                        ),
                    ],
                )
            )
        return log_style.block_report("VPM runtime profile", sections).splitlines()

    def report(self) -> None:
        """Publish the full profile with one write to the owning logger."""
        from source import log_style

        if Logging._routine_messages_enabled:
            Logging.message(log_style.FormattedText("\n".join(self.format_report())), flush=True)
