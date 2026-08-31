"""Dependency-neutral event interface for VPM numerical kernels."""

from __future__ import annotations

from typing import Any, Protocol


class PhysicsEventObserver(Protocol):
    """Optional observer for numerical-kernel diagnostics."""

    def warning(self, message: str) -> None: ...
    def record(self, message: str, *details: Any) -> None: ...


class NullPhysicsEventObserver:
    """Default observer: physics remains silent and independent of I/O."""

    def warning(self, message: str) -> None:
        return None

    def record(self, message: str, *details: Any) -> None:
        return None
