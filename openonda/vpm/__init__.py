"""Vortex-particle and vortex-lattice solvers for OpenONDA."""

import os

# OpenONDA reports its backend after solver construction is established. Taichi's
# import-time banner would otherwise repeat in the launcher and every MPI rank.
os.environ.setdefault("ENABLE_TAICHI_HEADER_PRINT", "False")

from .boundary_elements import vlm
from .boundary_elements.vlm.config import (
    ForceConfig,
    VLMMeshSetup,
    VLMSetup,
    VLMSurfaceSetup,
)
from .boundary_elements.vlm.coupling.kinematics import (
    ManeuverVLM,
    PitchingVLM,
    RotatingVLM,
    SmoothRampVLM,
    StaticVLM,
    TranslatingVLM,
)
from .config import (
    Backup,
    DiagnosticsConfig,
    DivergenceLimit,
    DivergenceRelaxationConfig,
    FilamentRefinementConfig,
    FiniteStateCheck,
    GrowthLimit,
    LagrangianCFLLimit,
    MisalignmentLimit,
    Numerics,
    ParticleStateError,
    ParticleStateLimits,
    ParticleStrengthLimit,
    RestartState,
    RunPlan,
    Samplers,
    StabilizationConfig,
    TurbulenceConfig,
    ViscousConfig,
    VPMCase,
)
from .core.solver import VPMSolver
from .diagnostics import FlowIntegralsSampler
from .initialization import (
    InitialCondition,
    ParticleDistribution,
    VortexParticleSet,
)
from .io.sampling import EverySteps, EveryTime, FinalOnly, LineSampler, SurfaceSampler, VLMSampler
from .numerics.rk_tableaux import RK2, RK4, SSPRK3
from .physics.induction.direct import DirectInduction
from .physics.induction.fmm import FMMInduction
from .physics.induction.gaussian_mesh.parameters import GaussianMeshParameters
from .physics.induction.gaussian_mesh.session import GaussianSlabSettings
from .physics.induction.slip_slab import SlipSlabInduction
from .physics.induction.treecode import TreecodeInduction
from .stabilization import (
    DivergenceRelaxationError,
    FilamentRefinementError,
    StabilizationError,
    StabilizationManager,
)

__all__ = [
    "LineSampler",
    "ManeuverVLM",
    "PitchingVLM",
    "RotatingVLM",
    "SmoothRampVLM",
    "StaticVLM",
    "SurfaceSampler",
    "TranslatingVLM",
    "DirectInduction",
    "SlipSlabInduction",
    "FMMInduction",
    "GaussianMeshParameters",
    "GaussianSlabSettings",
    "DivergenceRelaxationConfig",
    "DivergenceRelaxationError",
    "Backup",
    "DivergenceLimit",
    "DiagnosticsConfig",
    "FinalOnly",
    "FilamentRefinementConfig",
    "FilamentRefinementError",
    "FiniteStateCheck",
    "InitialCondition",
    "ForceConfig",
    "FlowIntegralsSampler",
    "EverySteps",
    "EveryTime",
    "GrowthLimit",
    "ParticleStateError",
    "ParticleStateLimits",
    "LagrangianCFLLimit",
    "ParticleStrengthLimit",
    "MisalignmentLimit",
    "Numerics",
    "RestartState",
    "ParticleDistribution",
    "Samplers",
    "StabilizationConfig",
    "StabilizationError",
    "StabilizationManager",
    "RK2",
    "RK4",
    "SSPRK3",
    "TurbulenceConfig",
    "VLMSampler",
    "VLMMeshSetup",
    "VLMSetup",
    "VLMSurfaceSetup",
    "RunPlan",
    "VPMCase",
    "VPMSolver",
    "TreecodeInduction",
    "ViscousConfig",
    "VortexParticleSet",
    "vlm",
]
