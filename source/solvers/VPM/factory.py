"""Physics-first construction helper for the VPM solver."""

from __future__ import annotations

import os

from .config.types import VPMSetup

_RANK_VARIABLES = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "MV2_COMM_WORLD_RANK",
    "SLURM_PROCID",
)


def _is_root_process() -> bool:
    for name in _RANK_VARIABLES:
        value = os.environ.get(name)
        if value is not None:
            return int(value) == 0
    return True


def create_vpm_solver(setup: VPMSetup):
    """Construct the VPM solver on the process that owns particle state.

    Serial cases naturally build one solver. In a coupled MPI case only the
    root process owns the VPM/GPU state; other ranks return ``None`` for the
    coupler's distributed FVM path.
    """
    if not _is_root_process():
        return None
    from .core.solver import VPMSolver

    return VPMSolver(setup=setup)


__all__ = ["create_vpm_solver"]
