"""Construction helpers for the VPM solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .config.setup import VPMSetup

if TYPE_CHECKING:
    pass

_RANK_VARIABLES = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "MV2_COMM_WORLD_RANK",
    "SLURM_PROCID",
)
_SIZE_VARIABLES = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_NTASKS",
)


def _rank_and_size() -> tuple[int, int]:
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank()), int(MPI.COMM_WORLD.Get_size())
    except ImportError:
        pass
    rank, size = 0, 1
    for name in _RANK_VARIABLES:
        if os.environ.get(name) is not None:
            rank = int(os.environ[name])
            break
    for name in _SIZE_VARIABLES:
        if os.environ.get(name) is not None:
            size = int(os.environ[name])
            break
    return rank, size


@dataclass(frozen=True)
class _InactiveVPMSolver:
    setup: VPMSetup
    case_dir: Path
    _openonda_inactive_rank: bool = True


def _runtime_setup(setup: VPMSetup, case_dir: Path) -> VPMSetup:
    def resolved(path: str) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = case_dir / candidate
        return str(candidate.resolve())

    backup = replace(
        setup.backup,
        directory=resolved(setup.backup.directory),
        log_directory=resolved(setup.backup.log_directory),
    )
    return replace(setup, backup=backup)


def create_vpm_solver(setup: VPMSetup, *, case_dir: str | Path | None = None):
    """Construct VPM state without exposing MPI ownership to user code."""
    resolved_case = Path("." if case_dir is None else case_dir).resolve()
    runtime_setup = _runtime_setup(setup, resolved_case)
    rank, size = _rank_and_size()
    if size > 1 and rank != 0:
        return _InactiveVPMSolver(runtime_setup, resolved_case)
    from .core.solver import VPMSolver

    return VPMSolver(setup=runtime_setup, case_dir=resolved_case)


__all__ = ["create_vpm_solver"]
