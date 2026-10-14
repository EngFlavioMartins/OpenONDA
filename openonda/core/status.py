"""Run status values written by every solver to its log and metadata."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["RunStatus"]


class RunStatus(StrEnum):
    """State of a solver run, as recorded in ``solution/<solver>_metadata.json``.

    Members compare equal to their lowercase string values, so a status read
    back from JSON compares directly with a member.

    Attributes
    ----------
    not_started
        The solver exists but no run or step has been requested.
    created
        The solver is constructed and its initial state is set.
    running
        A run or interactive stepping is in progress.
    completed
        The run reached ``RunPlan.end_time``.
    not_converged
        A steady or iterative solve stopped before its residual tolerance.
    partial
        The run stopped early on request and its last state is valid.
    failed
        The run stopped on an error.
    unstable
        The solution diverged or left its physical bounds.
    resolution_lost
        The discretization no longer resolves the flow, for example the
        particle overlap or core size limits were violated.
    wall_time_limit
        The run stopped at its wall-clock limit with a valid last state.
    """

    not_started = "not_started"
    created = "created"
    running = "running"
    completed = "completed"
    not_converged = "not_converged"
    partial = "partial"
    failed = "failed"
    unstable = "unstable"
    resolution_lost = "resolution_lost"
    wall_time_limit = "wall_time_limit"
