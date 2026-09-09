"""Explicit controls for optional VPM diagnostic work."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Controls optional solver diagnostics without process-wide environment state.

    Parameters
    ----------
    detailed_timing : bool, default=False
        Synchronize the backend around named solver phases and report their
        per-step timing breakdown. Disabled by default because synchronization
        can materially slow accelerator runs.
    validate_stages : bool, default=False
        Validate particle core radii and volumes at evolution-stage boundaries.
        Disabled by default because it copies fields from the backend.

    Notes
    -----
    These flags affect observation cost only; they do not select a physical
    model or alter accepted particle-state units.
    """

    detailed_timing: bool = False
    validate_stages: bool = False
