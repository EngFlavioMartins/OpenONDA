"""Common sampler abstraction shared by every FVM sampler.

A sampler is a small object that owns its output file name, its geometry and
its own deterministic cadence (:class:`SamplingSchedule`).  Sampling is driven
by the :class:`~source.solvers.FVM.sampling.executor.FVMSamplerExecutor`,
which runs after every accepted solver step and lets each sampler decide
whether it is due.  The same samplers drive live runs and offline
post-processing (:class:`~source.solvers.FVM.sampling.postprocess.PostProcess`),
so a schedule's decision must be reproducible from ``time_step`` / ``flow_time``
alone — never from a mutable call counter.

All sampler output lands in ``<case_root>/samples/`` (see :func:`samples_dir`);
the path is deliberately not a per-sampler option.

Examples
--------
>>> schedule = SamplingSchedule(every_n_steps=10)   # due at steps 0, 10, 20, ...
>>> schedule.is_due(0, 0.0, 0.01)
True
>>> schedule.is_due(5, 0.05, 0.01)
False
>>> SamplingSchedule(every_time=0.1).is_due(15, 0.15, 0.01)
True
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from typing import Any

# Canonical CSV column order for FVM field samplers.  The leading columns match
# the VPM sampler's so one reader serves both solvers; ``p`` is an FVM-only
# extra.  This list is the single source of truth for headers and rows.
SAMPLER_CSV_COLUMNS = [
    "x",
    "y",
    "z",
    "Ux",
    "Uy",
    "Uz",
    "omega_x",
    "omega_y",
    "omega_z",
    "p",
]


def samples_dir(case_dir: str) -> str:
    """Return the one canonical sampler output directory for a case root."""
    return os.path.join(case_dir, "samples")


@dataclass(frozen=True)
class SamplingSchedule:
    """Deterministic sampling cadence owned by one sampler.

    Provide exactly one of ``every_n_steps`` (sample when
    ``time_step % every_n_steps == 0``) or ``every_time`` (sample on the first
    accepted step whose ``flow_time`` lies within half a time step of an
    integer multiple of ``every_time``).  Because the decision depends only on
    ``time_step``/``flow_time``, a live simulation and an offline
    ``PostProcess`` over archived snapshots produce identical events.

    Examples
    --------
    >>> SamplingSchedule(every_n_steps=10)
    SamplingSchedule(every_n_steps=10, every_time=None)
    >>> SamplingSchedule(every_time=0.1)
    SamplingSchedule(every_n_steps=None, every_time=0.1)
    """

    every_n_steps: int | None = None
    every_time: float | None = None

    def __post_init__(self) -> None:
        if (self.every_n_steps is None) == (self.every_time is None):
            raise ValueError("Provide exactly one of every_n_steps or every_time")
        if self.every_n_steps is not None and int(self.every_n_steps) < 1:
            raise ValueError("every_n_steps must be a positive integer")
        if self.every_time is not None and float(self.every_time) <= 0.0:
            raise ValueError("every_time must be positive")

    def is_due(self, time_step: int, flow_time: float, dt: float | None = None) -> bool:
        """Whether a sampling event at the given state should run."""
        if self.every_n_steps is not None:
            return int(time_step) % int(self.every_n_steps) == 0
        if dt is None or dt <= 0.0 or self.every_time is None:
            return False
        interval = float(self.every_time)
        target = round(float(flow_time) / interval) * interval
        return abs(float(flow_time) - target) <= 0.5 * float(dt) + 1e-12


class Sampler:
    """Base class for FVM samplers.

    Subclasses implement :meth:`sample`, which returns a plain dict of
    canonical columns; the write methods are provided by the subclass.
    ``file_name`` defaults to the lower-cased class name without the
    ``Sampler`` suffix.

    Examples
    --------
    >>> class MySampler(Sampler):
    ...     def sample(self, context):
    ...         return {"x": [0.0]}
    >>> MySampler(file_name="probe").name
    'probe'
    """

    def __init__(
        self,
        file_name: str | None = None,
        schedule: SamplingSchedule | None = None,
    ) -> None:
        self.file_name = file_name
        self.schedule = schedule if schedule is not None else SamplingSchedule(every_n_steps=1)

    @property
    def name(self) -> str:
        """Output stem: ``file_name`` or the class name without the suffix."""
        return self.file_name or self.__class__.__name__.lower().replace("sampler", "")

    def is_due(self, time_step: int, flow_time: float, dt: float | None = None) -> bool:
        """Whether this sampler runs at the given time/step."""
        return self.schedule.is_due(time_step, flow_time, dt)

    def sample(self, context) -> dict[str, Any] | None:
        raise NotImplementedError


def append_csv_rows(
    filepath: str,
    header: list[str],
    rows: list[list[Any]],
) -> None:
    """Append ``rows`` under ``header`` to a CSV file, writing the header once."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
    with open(filepath, "a", newline="") as stream:
        writer = csv.writer(stream)
        if write_header:
            writer.writerow(header)
        writer.writerows(rows)


def write_pvd(samples_dir: str, name: str, entries: list[tuple[float, str]]) -> None:
    """Write a ParaView Data Collection index for time-series playback."""
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    lines += [
        f'    <DataSet timestep="{time_val}" file="{filename}"/>' for time_val, filename in entries
    ]
    lines += ["  </Collection>", "</VTKFile>"]
    with open(os.path.join(samples_dir, f"{name}.pvd"), "w") as stream:
        stream.write("\n".join(lines))
