"""Canonical VPM sampling, backup, and log-output configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _normalise_output_directory(value: str | Path, field_name: str) -> str:
    path = Path(value)
    if not str(path).strip():
        raise ValueError(f"{field_name} must be a non-empty path")
    return str(path)


@dataclass(frozen=True)
class Backup:
    """Scheduled numerical backups and the solver log destination.

    ``interval_steps=0`` disables scheduled backups. Both directories are
    resolved relative to the solver case directory when they are not absolute.
    """

    interval_steps: int = 0
    directory: str = "solution"
    log_directory: str = "solution"

    def __post_init__(self) -> None:
        if isinstance(self.interval_steps, bool) or not isinstance(self.interval_steps, int):
            raise TypeError("backup interval_steps must be an integer")
        if self.interval_steps < 0:
            raise ValueError("backup interval_steps must be non-negative")
        object.__setattr__(
            self,
            "directory",
            _normalise_output_directory(self.directory, "backup directory"),
        )
        object.__setattr__(
            self,
            "log_directory",
            _normalise_output_directory(self.log_directory, "backup log_directory"),
        )


@dataclass(frozen=True, init=False)
class Samplers:
    """The requested sampler objects and their optional output subdirectory.

    Samples are passed positionally. ``directory`` is a relative directory
    below the case's canonical ``samples/`` root.
    """

    samples: tuple[Any, ...]
    directory: str | None

    def __init__(self, *samples: Any, directory: str | Path | None = None) -> None:
        if directory is not None:
            raw_directory = str(directory)
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
            directory = str(path)

        for sample in samples:
            if not any(hasattr(sample, method) for method in ("sample", "save_csv", "save_vtp")):
                raise TypeError(
                    "each Samplers entry must provide sample(), save_csv(), or save_vtp()"
                )

        object.__setattr__(self, "samples", tuple(samples))
        object.__setattr__(self, "directory", directory)
