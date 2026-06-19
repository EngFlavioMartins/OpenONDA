"""
OpenONDA 3D Panel Solver Package.
==================
GPU-accelerated potential flow solver for arbitrary 3D triangular meshes.
Features Taichi-lang kernels, BiCGSTAB linear solvers, and decoupled wake shedding.

Mirrors ``vlm/`` package conventions: same module layout, naming patterns,
and API design.  See ``solver/``, ``coupling/``, ``kernels/``, ``geometry/``.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from .coupling.kinematics import (
    CompositePanel,
    HeavingPanel,
    ManeuverPanel,
    PanelKinematics,
    PitchingPanel,
    Plunging,
    RampedRotation,
    RotatingPanel,
    Static,
    StaticPanel,
    TranslatingPanel,
)
from .coupling.separation import SeparationModel
from .coupling.shedding import CouplingConfig, VortexShedder
from .geometry.stl_io import load_stl, save_stl
from .geometry.surface_io import (
    load_scene,
    load_surface_metadata,
    save_scene,
    save_surface_metadata,
)
from .solver.diagnostics import PanelDiagnostics
from .solver.forces import PanelForceEvaluator
from .solver.lattice import PanelLattice
from .solver.loading_distribution import PanelLoadingDistribution
from .solver.panel_solver import ForceConfig, PanelSolver
from .solver.vtk_export import panel_mesh_to_vtp

__all__ = [
    "PanelSolver",
    "ForceConfig",
    "PanelLattice",
    "CouplingConfig",
    "PanelKinematics",
    "StaticPanel",
    "TranslatingPanel",
    "RotatingPanel",
    "PitchingPanel",
    "HeavingPanel",
    "ManeuverPanel",
    "CompositePanel",
    "Static",
    "Plunging",
    "RampedRotation",
    "load_stl",
    "save_stl",
    "save_surface_metadata",
    "load_surface_metadata",
    "save_scene",
    "load_scene",
    "VortexShedder",
    "SeparationModel",
    "PanelDiagnostics",
    "PanelForceEvaluator",
    "PanelLoadingDistribution",
    "panel_mesh_to_vtp",
]
