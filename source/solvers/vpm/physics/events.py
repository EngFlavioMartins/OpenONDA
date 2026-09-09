"""Dependency-neutral event interface for VPM numerical kernels."""

from __future__ import annotations

from typing import Protocol


class PhysicsEventObserver(Protocol):
    """Optional observer for numerical-kernel diagnostics.

    Implementations receive human-readable messages only; the observer is
    deliberately independent of logging, files, and solver state.
    """

    def warning(self, message: str) -> None:
        """Report a non-fatal numerical warning."""

        ...

    def record(self, message: str, *details: object) -> None:
        """Record a diagnostic message and optional scalar/detail values."""

        ...


class NullPhysicsEventObserver:
    """Default no-op observer keeping physics independent of I/O."""

    def warning(self, message: str) -> None:
        """Ignore a warning."""
        return None

    def record(self, message: str, *details: object) -> None:
        """Ignore a diagnostic record."""
        return None
