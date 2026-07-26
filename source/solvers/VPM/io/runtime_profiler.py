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
import os
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
        self._profiler._synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self._profiler._synchronize()
        p = self._profiler
        p.step_time = time.perf_counter() - self._t0
        p.wall_time += p.step_time
        p.n_steps += 1
        return False


class RuntimeProfiler:
    """Accumulating wall-clock profiler for the VPM solver.

    Parameters
    ----------
    enabled : bool
        When ``False``, :meth:`section` is a zero-overhead no-op (full-step
        timing via :meth:`step` still runs, as the per-step line is always shown).
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

    def __init__(self, *, enabled: bool = True, sync: Callable[[], None] | None = None) -> None:
        self.enabled = enabled
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

        >>> with profiler.section("Velocity"):
        ...     solver._update_velocities()
        """
        if not self.enabled:
            return _NULLCTX
        return _Section(self, name)

    def step(self):
        """Time one full solver step (use around the body of ``update_state``)."""
        return _Step(self)

    def _record(self, name: str, dt: float) -> None:
        self._cumulative[name] = self._cumulative.get(name, 0.0) + dt
        self._calls[name] = self._calls.get(name, 0) + 1
        self._last[name] = dt

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
        unchanged.  The breakdown is shown only when ``VPM_DETAILED_TIMING=1``
        is set.
        """
        detailed = self._last if os.environ.get("VPM_DETAILED_TIMING", "0") == "1" else None
        Logging.step_timing(self.step_time, self.wall_time, detailed)

    def format_report(self) -> list[str]:
        """Return the cumulative timing report as a list of text lines.

        Columns: section label, number of calls, cumulative seconds, average
        milliseconds per call, and percent of total wall time.  A footer adds the
        measured / unprofiled split and the per-step average.
        """
        title = f" VPM RUNTIME PROFILE  ({self.n_steps} steps, {self.wall_time:.3f} s wall)"
        particle_line = (
            f"  Number of particles     : {self.particle_count:d}"
            if self.particle_count is not None
            else None
        )
        if not self._cumulative:
            bar = "=" * max(len(title), 40)
            lines = ["", bar, title, bar]
            if particle_line is not None:
                lines.append(particle_line)
            lines.extend(["  (no sections recorded)", bar])
            return lines

        label_w = max(len("Section"), *(len(k) for k in self._cumulative))
        header = (
            f"  {'Section':<{label_w}}  {'calls':>6}  {'total[s]':>9}  {'avg[ms]':>9}  {'%wall':>6}"
        )
        bar = "=" * len(header)
        sep = "-" * len(header)
        lines = ["", bar, title]
        if particle_line is not None:
            lines.append(particle_line)
        lines.extend([sep, header, sep])

        measured = 0.0
        for name, total in sorted(self._cumulative.items(), key=lambda kv: kv[1], reverse=True):
            calls = self._calls[name]
            measured += total
            avg_ms = 1.0e3 * total / calls if calls else 0.0
            pct = 100.0 * total / self.wall_time if self.wall_time > 0 else 0.0
            lines.append(
                f"  {name:<{label_w}}  {calls:>6d}  {total:>9.3f}  {avg_ms:>9.2f}  {pct:>5.1f}%"
            )

        lines.append(sep)
        other = self.wall_time - measured
        meas_pct = 100.0 * measured / self.wall_time if self.wall_time > 0 else 0.0
        other_pct = 100.0 * other / self.wall_time if self.wall_time > 0 else 0.0
        avg_step_ms = 1.0e3 * self.wall_time / self.n_steps if self.n_steps else 0.0
        blank = " " * 9
        lines.append(
            f"  {'Measured':<{label_w}}  {blank[:6]}  {measured:>9.3f}  {blank:>9}  {meas_pct:>5.1f}%"
        )
        lines.append(
            f"  {'Unprofiled':<{label_w}}  {blank[:6]}  {other:>9.3f}  {blank:>9}  {other_pct:>5.1f}%"
        )
        lines.append(
            f"  {'Step total':<{label_w}}  {self.n_steps:>6d}  {self.wall_time:>9.3f}  "
            f"{avg_step_ms:>9.2f}  {100.0:>5.1f}%"
        )
        lines.append(bar)
        return lines

    def report(self) -> None:
        """Emit the cumulative timing report through the :class:`Logging` sink."""
        for line in self.format_report():
            Logging.message(line)
