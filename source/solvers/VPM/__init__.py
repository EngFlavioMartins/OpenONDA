"""Vortex-particle and vortex-lattice solvers for OpenONDA.

The package exposes the main VPM solver and configuration objects together with
the boundary-element solvers used to represent lifting and solid surfaces.
Numerical kernels run through Taichi on the configured CPU or GPU backend.
"""

import logging
import os
import sys

from .boundary_elements import vlm
from .boundary_elements.panels.solver.panel_solver import PanelSolver
from .boundary_elements.vlm.config import ForceConfig, VLMMeshSetup, VLMSetup, VLMSurfaceSetup
from .config.types import (
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
from .core.solver import Solver
from .diagnostics import RingDiagnosticsSampler
from .factory import setup_vpm_solver
from .particles.distribution import ParticleDistributor
from .stabilization import (
    DivergenceRelaxationError,
    FilamentRefinementError,
    StabilizationError,
    StabilizationManager,
)

__all__ = [
    "Solver",
    "VPMSetup",
    "setup_vpm_solver",
    "AdvectionConfig",
    "StretchingConfig",
    "TurbulenceConfig",
    "ViscousConfig",
    "DivergenceRelaxationConfig",
    "DivergenceRelaxationError",
    "FilamentRefinementConfig",
    "FilamentRefinementError",
    "StabilizationConfig",
    "StabilizationError",
    "StabilizationManager",
    "VelocityConfig",
    "ForceConfig",
    "VLMSetup",
    "VLMSurfaceSetup",
    "VLMMeshSetup",
    "ParticleDistributor",
    "RingDiagnosticsSampler",
    "PanelSolver",
    "vlm",
]

# VPM package logging setup
#
# When workers set the VPM_LOG environment variable (as allrun_*.py scripts do)
# all simulation output — both print() calls and logger records — is captured
# into that file.  Without VPM_LOG the package is silent: no log file is
# created at import time and stdout/stderr are left untouched so that
# interactive / coupled runs stay in control of their own output.
_log_file = os.environ.get("VPM_LOG", "")

if _log_file:
    # Caller explicitly opted in — redirect all output to the specified file.
    _logger = logging.getLogger("vpm")
    if not any(isinstance(h, logging.FileHandler) for h in _logger.handlers):
        _fh = logging.FileHandler(_log_file, mode="a")
        _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        _logger.addHandler(_fh)
    _logger.setLevel(logging.INFO)

    if not getattr(sys, "_vpm_stdout_redirected", False):
        _log_stream = open(_log_file, "a", buffering=1)  # noqa: SIM115
        sys._vpm_original_stdout = sys.stdout
        sys._vpm_original_stderr = sys.stderr
        sys.stdout = _log_stream
        sys.stderr = _log_stream
        sys._vpm_stdout_redirected = True

        def _vpm_restore_stdout() -> None:
            """Restore stdout/stderr to their original streams."""
            if getattr(sys, "_vpm_stdout_redirected", False):
                sys.stdout.flush()
                sys.stdout = sys._vpm_original_stdout
                sys.stderr = sys._vpm_original_stderr
                sys._vpm_stdout_redirected = False

        sys._vpm_restore_stdout = _vpm_restore_stdout
