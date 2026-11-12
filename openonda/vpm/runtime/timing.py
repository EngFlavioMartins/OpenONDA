"""Wall-time phases of the particle solver.

The solver times its step with the :class:`~openonda.core.profiling.Profiler`
that :attr:`LoggingConfig.detailed_timing` enables. A phase that contains
other timed phases is timed with :func:`remainder_section`, so the phase times
of a step do not overlap and their sum is at most the step time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter

import taichi as ti

from openonda.core.profiling import Profiler

__all__ = ["remainder_section"]


@contextmanager
def remainder_section(
    profiler: Profiler, name: str, *, synchronize: Callable[[], None] = ti.sync
) -> Iterator[None]:
    """Time a block and record the part that no phase inside it recorded.

    Parameters
    ----------
    profiler : Profiler
        Step profiler of the solver. When it is disabled the block runs
        untimed and the compute device is not synchronized.
    name : str
        Phase name, for example ``"Runge-Kutta update"``.
    synchronize : callable, default=taichi.sync
        Zero-argument function that waits for the compute device.

    Notes
    -----
    The compute device is synchronized before and after the block, because
    its kernels run asynchronously. The recorded time is the wall time of the
    block minus the time the profiler recorded while the block ran.
    """
    if not profiler.enabled:
        yield
        return
    synchronize()
    recorded_before = sum(profiler.step_sections.values())
    started = perf_counter()
    try:
        yield
    finally:
        synchronize()
        elapsed = perf_counter() - started
        recorded_inside = sum(profiler.step_sections.values()) - recorded_before
        profiler.record(name, max(0.0, elapsed - recorded_inside))
