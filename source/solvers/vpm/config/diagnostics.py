"""Explicit controls for optional VPM diagnostic work."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Controls optional solver diagnostics without process-wide environment state.

    Parameters
    ----------
    detailed_timing
        Synchronize the backend around named solver phases and report their
        per-step timing breakdown. Disabled by default because synchronization
        can materially slow accelerator runs.
    validate_stages
        Validate particle core radii and volumes at evolution-stage boundaries.
        Disabled by default because it copies fields from the backend.
    """

    detailed_timing: bool = False
    validate_stages: bool = False
