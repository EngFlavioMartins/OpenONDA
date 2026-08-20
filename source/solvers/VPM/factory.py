"""Construction helper for the VPM solver."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .config.setup import VPMSetup

if TYPE_CHECKING:
    from .core.solver import VPMSolver

_RANK_VARIABLES = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "MV2_COMM_WORLD_RANK",
    "SLURM_PROCID",
)


def _is_root_process() -> bool:
    """Return whether the current process owns the VPM particle state."""
    for name in _RANK_VARIABLES:
        value = os.environ.get(name)
        if value is not None:
            return int(value) == 0
    return True


def create_vpm_solver(setup: VPMSetup) -> VPMSolver | None:
    """Construct the VPM solver on the process that owns particle state.

    Serial runs construct one solver. In distributed FVM-VPM coupling, only the
    root process owns the VPM/GPU state and other ranks return ``None``.
    """
    if not _is_root_process():
        return None

    from .core.solver import VPMSolver

    return VPMSolver(setup=setup)


__all__ = ["create_vpm_solver"]
