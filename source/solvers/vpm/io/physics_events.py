"""I/O adapter for optional numerical-physics diagnostics."""

from __future__ import annotations

from .logging import Logging


class LoggingPhysicsEventObserver:
    """Present physics events through the VPM logging policy."""

    def warning(self, message: str) -> None:
        """Forward a physics warning to the configured VPM logger."""
        Logging.warning(message)

    def record(self, message: str, *details: object) -> None:
        """Forward a physics diagnostic and optional detail values."""
        Logging.record(message, *details)
