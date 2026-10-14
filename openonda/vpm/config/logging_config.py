"""Log verbosity, cadence and optional diagnostic work of a VPM run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from openonda.core.schedules import EverySteps, EveryTime, Schedule

__all__ = ["LoggingConfig"]


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Log records written to ``solution/vpm.log`` and the console.

    Parameters
    ----------
    mode : {'simple', 'debug'}, default='simple'
        ``debug`` adds module detail records and runs Taichi in its debug
        mode, which checks array bounds at a large cost in speed.
    schedule : EverySteps or EveryTime, default=EverySteps(1)
        Accepted steps with a step report. The last step of a run is always
        reported.
    console : bool, default=True
        Also write the records to the terminal.
    detailed_timing : bool, default=False
        Synchronize the compute device around each solver phase and report
        the time of every phase. Synchronization slows accelerator runs.
    validate_stages : bool, default=False
        Check particle core radii and volumes after every evolution stage.
        The check copies fields from the compute device.
    """

    mode: Literal["simple", "debug"] = "simple"
    schedule: Schedule = field(default_factory=lambda: EverySteps(1))
    console: bool = True
    detailed_timing: bool = False
    validate_stages: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"simple", "debug"}:
            raise ValueError("LoggingConfig.mode must be 'simple' or 'debug'")
        if not isinstance(self.schedule, EverySteps | EveryTime):
            raise TypeError("LoggingConfig.schedule must be EverySteps or EveryTime")
        for name in ("console", "detailed_timing", "validate_stages"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"LoggingConfig.{name} must be a bool")
