"""Canonical VPM sampling, backup, and log destinations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _directory(value: str | Path, name: str) -> str:
    path = Path(value)
    if not str(path).strip():
        raise ValueError(f"{name} must be a non-empty path")
    return str(path)


@dataclass(frozen=True, slots=True)
class Backup:
    """Numerical restart cadence and destinations owned by a VPM case.

    ``interval_steps=0`` disables periodic saves. ``directory`` and
    ``log_directory`` are relative to the case directory unless absolute.
    Backups keep compute precision and do not trigger samples.
    """

    interval_steps: int = 0
    directory: str = "solution"
    log_directory: str = "solution"

    def __post_init__(self) -> None:
        if isinstance(self.interval_steps, bool) or not isinstance(self.interval_steps, int):
            raise TypeError("backup interval_steps must be an integer")
        if self.interval_steps < 0:
            raise ValueError("backup interval_steps must be non-negative")
        object.__setattr__(self, "directory", _directory(self.directory, "backup directory"))
        object.__setattr__(self, "log_directory", _directory(self.log_directory, "log directory"))


@dataclass(frozen=True, slots=True)
class Samplers:
    """Scientific samples and their optional path below ``samples/``.

    Missing scientific output is fatal. Individual sample implementations may
    elect to handle their own recoverable errors before returning from
    ``write``.
    """

    samples: tuple[object, ...] = ()
    directory: str | None = None

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        for sample in samples:
            if not any(
                hasattr(sample, method) for method in ("sample", "save_csv", "save_vtp", "write")
            ):
                raise TypeError(
                    "each sampler must provide write(), sample(), save_csv(), or save_vtp()"
                )
        if self.directory is not None:
            raw_directory = str(self.directory)
            path = Path(raw_directory)
            if (
                not raw_directory.strip()
                or raw_directory.strip() in {".", ".."}
                or path.is_absolute()
                or any(part in {".", ".."} for part in path.parts)
            ):
                raise ValueError(
                    "sampler directory must be a non-empty relative path below samples/"
                )
            object.__setattr__(self, "directory", str(path))
        object.__setattr__(self, "samples", samples)
