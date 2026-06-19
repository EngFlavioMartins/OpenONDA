"""
Initialization module for VPM.
==================
Initialization module for VPM. module.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

# OpenONDA/solvers/VPM/__init__.py
"""
Vortex Particle Method (VPM) solver module for OpenONDA.

This module provides a complete VPM implementation with:
- DNS, LES, and inviscid flow models
- GPU acceleration via Taichi
- Robust backup/restore functionality
- Advanced turbulence modeling
- Comprehensive diagnostics

The module is organized into several sub-packages:
- config: Constants, types, and configuration
- core: Main solver implementation
- particles: Particle data structures and physics
- turbulence: LES turbulence models
- spatial: Spatial algorithms (neighbor search, etc.)
- io: Input/output and backup systems
- utils: Utility functions and logging
"""

from .boundary_elements.panels.solver.panel_solver import PanelSolver
from .config.types import ForceConfig, SolverConfig, StabilizationConfig, VelocityConfig
from .core.solver import Solver
from .particles.distribution import ParticleDistributor

# VLM module (optional import)
try:
    from .boundary_elements import vlm

    _has_vlm = True
except ImportError:
    _has_vlm = False
    vlm = None

__all__ = [
    "Solver",
    "SolverConfig",
    "StabilizationConfig",
    "VelocityConfig",
    "ForceConfig",
    "ParticleDistributor",
    "PanelSolver",
]

if _has_vlm:
    __all__.append("vlm")

# -----------------------------------------------------------------------------
# VPM package logging setup
#
# When workers set the VPM_LOG environment variable (as allrun_*.py scripts do)
# all simulation output — both print() calls and logger records — is captured
# into that file.  Without VPM_LOG the package is silent: no log file is
# created at import time and stdout/stderr are left untouched so that
# interactive / coupled runs stay in control of their own output.
# -----------------------------------------------------------------------------
import logging
import os
import sys

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
