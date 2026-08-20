"""Vortex-particle and vortex-lattice solvers for OpenONDA."""

import logging
import os
import sys

from .boundary_elements import vlm
from .boundary_elements.panels.solver.panel_solver import PanelSolver
from .boundary_elements.vlm.config import (
    ForceConfig,
    VLMMeshSetup,
    VLMSetup,
    VLMSurfaceSetup,
)
from .config import (
    AdvectionConfig,
    DivergenceRelaxationConfig,
    FilamentRefinementConfig,
    StabilizationConfig,
    StretchingConfig,
    TurbulenceConfig,
    VelocityConfig,
    ViscousConfig,
    VPMSetup,
)
from .core.solver import VPMSolver
from .diagnostics import RingDiagnosticsSampler
from .factory import create_vpm_solver
from .io.checkpoint import CheckpointManager
from .particles.distribution import ParticleDistributor
from .stabilization import (
    DivergenceRelaxationError,
    FilamentRefinementError,
    StabilizationError,
    StabilizationManager,
)

__all__ = [
    "AdvectionConfig",
    "CheckpointManager",
    "DivergenceRelaxationConfig",
    "DivergenceRelaxationError",
    "FilamentRefinementConfig",
    "FilamentRefinementError",
    "ForceConfig",
    "PanelSolver",
    "ParticleDistributor",
    "RingDiagnosticsSampler",
    "StabilizationConfig",
    "StabilizationError",
    "StabilizationManager",
    "StretchingConfig",
    "TurbulenceConfig",
    "VLMMeshSetup",
    "VLMSetup",
    "VLMSurfaceSetup",
    "VPMSetup",
    "VPMSolver",
    "VelocityConfig",
    "ViscousConfig",
    "create_vpm_solver",
    "vlm",
]

_log_file = os.environ.get("VPM_LOG", "")

if _log_file:
    logger = logging.getLogger("vpm")
    if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        file_handler = logging.FileHandler(_log_file, mode="a")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    if not getattr(sys, "_vpm_stdout_redirected", False):
        log_stream = open(  # noqa: SIM115
            _log_file,
            "a",
            buffering=1,
        )
        sys._vpm_original_stdout = sys.stdout
        sys._vpm_original_stderr = sys.stderr
        sys.stdout = log_stream
        sys.stderr = log_stream
        sys._vpm_stdout_redirected = True

        def _vpm_restore_stdout() -> None:
            """Restore stdout and stderr to their original streams."""
            if getattr(sys, "_vpm_stdout_redirected", False):
                sys.stdout.flush()
                sys.stdout = sys._vpm_original_stdout
                sys.stderr = sys._vpm_original_stderr
                sys._vpm_stdout_redirected = False

        sys._vpm_restore_stdout = _vpm_restore_stdout
