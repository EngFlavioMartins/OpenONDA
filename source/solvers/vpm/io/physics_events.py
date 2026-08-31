"""I/O adapter for optional numerical-physics diagnostics."""

from __future__ import annotations

from typing import Any

from .logging import Logging


class LoggingPhysicsEventObserver:
    """Present physics events through the VPM logging policy."""

    def warning(self, message: str) -> None:
        Logging.warning(message)

    def record(self, message: str, *details: Any) -> None:
        Logging.record(message, *details)
