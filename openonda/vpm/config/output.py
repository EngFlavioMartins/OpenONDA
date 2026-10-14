"""Visualization frames, scientific samples and restart backups of a VPM case."""

from __future__ import annotations

from dataclasses import dataclass

from openonda.core.schedules import EverySteps, EveryTime, FinalOnly, Schedule
from openonda.core.vtk_output import DEFAULT_COMPRESSION
from openonda.core.write_precision import (
    DEFAULT_WRITE_PRECISION,
    WritePrecision,
    validate_write_precision,
)

__all__ = ["Backup", "Output", "Samplers"]


def _check_schedule(owner: str, schedule: object) -> None:
    if schedule is not None and not isinstance(schedule, EverySteps | EveryTime | FinalOnly):
        raise TypeError(f"{owner}.schedule must be EverySteps, EveryTime, FinalOnly or None")


@dataclass(frozen=True, slots=True)
class Output:
    """Particle visualization frames.

    Frames are written to ``solution/vpm/vpm_{step:06d}.vtu`` and listed in
    ``solution/vpm.pvd``; an attached vortex lattice writes its surface to
    ``solution/vlm/vlm_{step:06d}.vtp`` and ``solution/vlm.pvd`` at the same
    states. The initial and final frames follow ``RunPlan.initial_output``
    and ``RunPlan.final_output``.

    Parameters
    ----------
    schedule : EverySteps, EveryTime, FinalOnly or None, default=None
        Accepted states with a frame; ``None`` writes only the initial and
        final frames.
    precision : {'f32', 'f64'}, default='f32'
        Precision of the written point data, independent of the compute
        precision.
    compression : {'zlib', 'lz4', 'lzma', 'none'}, default='zlib'
        Compression of the appended VTK data.
    """

    schedule: Schedule | None = None
    precision: WritePrecision = DEFAULT_WRITE_PRECISION
    compression: str = DEFAULT_COMPRESSION

    def __post_init__(self) -> None:
        _check_schedule("Output", self.schedule)
        validate_write_precision(self.precision, field_name="Output.precision")
        if self.compression not in {"lz4", "lzma", "none", "zlib"}:
            raise ValueError("Output.compression must be 'lz4', 'lzma', 'none' or 'zlib'")


@dataclass(frozen=True, slots=True)
class Backup:
    """Restart backups ``solution/vpm/vpm_{step:06d}.h5``.

    Backups keep the compute precision and contain everything needed to
    continue the run; they write no visualization frames.

    Parameters
    ----------
    schedule : EverySteps, EveryTime, FinalOnly or None, default=None
        Accepted states with a backup; ``None`` writes no periodic backups.
    write_at_end : bool, default=True
        Write a backup of the last accepted state when the run ends with a
        valid state that has no backup yet.
    """

    schedule: Schedule | None = None
    write_at_end: bool = True

    def __post_init__(self) -> None:
        _check_schedule("Backup", self.schedule)
        if not isinstance(self.write_at_end, bool):
            raise TypeError("Backup.write_at_end must be a bool")


@dataclass(frozen=True, slots=True)
class Samplers:
    """Scientific samplers written below ``samples/``.

    Parameters
    ----------
    samples : tuple of samplers
        Objects implementing ``write(context)``, ``sample(solver)``,
        ``write_csv(...)`` or ``write_vtk(...)``.
    """

    samples: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        for sample in samples:
            if not any(
                hasattr(sample, method) for method in ("sample", "write_csv", "write_vtk", "write")
            ):
                raise TypeError(
                    "each sampler must provide write(), sample(), write_csv() or write_vtk()"
                )
        object.__setattr__(self, "samples", samples)
