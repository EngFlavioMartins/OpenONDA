"""Output events that trigger samplers, visualization frames and backups."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["OutputEvent"]


class OutputEvent(StrEnum):
    """Solver state at which scheduled output is written.

    Attributes
    ----------
    initial
        The initial state, before the first time step, written when
        ``RunPlan.initial_output`` (or a sampler's ``initial`` switch) is set.
    accepted_step
        A state after an accepted time step, written when a schedule is due.
    final
        The last state of a completed run, written once.
    failed
        The last valid state of a run that stopped on an error.
    """

    initial = "initial"
    accepted_step = "accepted_step"
    final = "final"
    failed = "failed"
