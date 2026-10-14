"""Visualization frames, samplers, restart backups and logging of an FVM case."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from openonda.core.schedules import EverySteps, Schedule
from openonda.core.write_precision import (
    DEFAULT_WRITE_PRECISION,
    WritePrecision,
    validate_write_precision,
)

from ._validation import check_schedule, lowercase_choice


@dataclass(frozen=True, slots=True)
class Output:
    """Cell-centred VTK XML visualization frames.

    Frames are written to ``solution/fvm/fvm_{step:06d}.vtu`` (``.pvtu`` for a
    partitioned run) and listed in ``solution/fvm.pvd``. The initial and final
    frames follow ``RunPlan.initial_output`` and ``RunPlan.final_output``.

    Parameters
    ----------
    schedule : EverySteps, EveryTime, FinalOnly or None, default=None
        Accepted states with a frame; ``None`` writes only the initial and
        final frames.
    precision : {'f32', 'f64'}, default='f32'
        Precision of the written cell data, independent of the compute
        precision.
    compression : {'zlib', 'lz4', 'none'}, default='zlib'
        Compression of the appended VTK data.
    asynchronous : bool, default=True
        Write frames on a background thread; writer errors surface at
        ``flush_output`` or ``close``.
    ghost_layers : {0, 1}, default=1
        Boundary ghost layers included in a partitioned frame.
    point_interpolation : {'none', 'boundary_weighted'}, default='none'
        Optional derived vertex data beside the cell data.
    """

    schedule: Schedule | None = None
    precision: WritePrecision = DEFAULT_WRITE_PRECISION
    compression: Literal["zlib", "lz4", "none"] = "zlib"
    asynchronous: bool = True
    ghost_layers: Literal[0, 1] = 1
    point_interpolation: Literal["none", "boundary_weighted"] = "none"

    def __post_init__(self) -> None:
        check_schedule("Output.schedule", self.schedule, optional=True)
        validate_write_precision(self.precision, field_name="Output.precision")
        lowercase_choice("Output.compression", self.compression, {"lz4", "none", "zlib"})
        if not isinstance(self.asynchronous, bool):
            raise TypeError("Output.asynchronous must be a boolean")
        if isinstance(self.ghost_layers, bool) or self.ghost_layers not in {0, 1}:
            raise ValueError("Output.ghost_layers must be 0 or 1")
        lowercase_choice(
            "Output.point_interpolation", self.point_interpolation, {"none", "boundary_weighted"}
        )


@dataclass(frozen=True, slots=True)
class Backup:
    """Restart backups ``solution/fvm/fvm_{step:06d}.npz``.

    A partitioned run writes the directory ``solution/fvm/fvm_{step:06d}``
    with one archive per rank. Backups keep the compute precision and the
    time history of the fields.

    Parameters
    ----------
    schedule : EverySteps, EveryTime, FinalOnly or None, default=None
        Accepted states with a backup; ``None`` writes no periodic backups.
    write_at_end : bool, default=True
        Write a backup of the last accepted state when the run ends.
    """

    schedule: Schedule | None = None
    write_at_end: bool = True

    def __post_init__(self) -> None:
        check_schedule("Backup.schedule", self.schedule, optional=True)
        if not isinstance(self.write_at_end, bool):
            raise TypeError("Backup.write_at_end must be a boolean")


@dataclass(frozen=True, slots=True)
class Samplers:
    """Scientific samplers written below ``samples/``.

    Parameters
    ----------
    samples : tuple of samplers
        :class:`~openonda.fvm.LineSampler`, :class:`~openonda.fvm.SurfaceSampler`,
        :class:`~openonda.fvm.ForceSampler` or :class:`~openonda.fvm.YPlusSampler`
        objects.
    """

    samples: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        for sampler in samples:
            if not callable(getattr(sampler, "sample", None)):
                raise TypeError(f"FVM sampler {type(sampler).__name__} must implement sample()")
            check_schedule(f"{type(sampler).__name__}.schedule", getattr(sampler, "schedule", None))
        object.__setattr__(self, "samples", samples)


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Log records of the accepted steps, written to ``solution/fvm.log``.

    Parameters
    ----------
    mode : {'simple', 'debug'}, default='simple'
        Compact step records or the full numerical diagnostics.
    schedule : EverySteps, EveryTime or FinalOnly, default=EverySteps(1)
        Accepted steps with a log record. Warnings are always logged.
    console : bool, default=True
        Mirror the log to the terminal.
    detailed_timing : bool, default=False
        Time the solver sections of every step and write
        ``solution/performance.jsonl``.
    """

    mode: Literal["simple", "debug"] = "simple"
    schedule: Schedule = field(default_factory=lambda: EverySteps(1))
    console: bool = True
    detailed_timing: bool = False

    def __post_init__(self) -> None:
        lowercase_choice("LoggingConfig.mode", self.mode, {"simple", "debug"})
        check_schedule("LoggingConfig.schedule", self.schedule)
        for name in ("console", "detailed_timing"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"LoggingConfig.{name} must be a boolean")
