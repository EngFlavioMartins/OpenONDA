"""Vortex-particle and vortex-lattice solvers for OpenONDA."""

from openonda.core.schedules import EverySteps, EveryTime, FinalOnly

from .boundary_elements import vlm
from .boundary_elements.vlm.config import ForceConfig, VLMConfig, VLMMeshConfig, VLMSurface
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
    DivergenceLimit,
    DivergenceRelaxationConfig,
    FilamentRefinementConfig,
    FiniteStateCheck,
    GrowthLimit,
    LagrangianCFLLimit,
    LoggingConfig,
    MisalignmentLimit,
    Numerics,
    Output,
    ParticleStateError,
    ParticleStateLimits,
    ParticleStrengthLimit,
    RunPlan,
    Samplers,
    StabilizationConfig,
    TurbulenceConfig,
    ViscousConfig,
    VPMCase,
)
from .core.solver import VPMSolver
from .initialization import (
    InitialCondition,
    ParticleDistribution,
    VortexParticleSet,
)
from .numerics.rk_tableaux import RK2, RK4, SSPRK3
from .physics.induction.direct import DirectInduction
from .physics.induction.fmm import FMMInduction
from .physics.induction.gaussian_mesh.parameters import GaussianMeshParameters
from .physics.induction.gaussian_mesh.session import GaussianSlabSettings
from .physics.induction.slip_slab import SlipSlabInduction
from .physics.induction.treecode import TreecodeInduction
from .sampling import FlowIntegralsSampler, LineSampler, SurfaceSampler, VLMSampler
from .stabilization import (
    DivergenceRelaxationError,
    FilamentRefinementError,
    StabilizationError,
    StabilizationManager,
)

__all__ = [
    "RK2",
    "RK4",
    "SSPRK3",
    "Backup",
    "DirectInduction",
    "DivergenceLimit",
    "DivergenceRelaxationConfig",
    "DivergenceRelaxationError",
    "EverySteps",
    "EveryTime",
    "FMMInduction",
    "FilamentRefinementConfig",
    "FilamentRefinementError",
    "FinalOnly",
    "FiniteStateCheck",
    "FlowIntegralsSampler",
    "ForceConfig",
    "GaussianMeshParameters",
    "GaussianSlabSettings",
    "GrowthLimit",
    "InitialCondition",
    "LagrangianCFLLimit",
    "LineSampler",
    "LoggingConfig",
    "ManeuverVLM",
    "MisalignmentLimit",
    "Numerics",
    "Output",
    "ParticleDistribution",
    "ParticleStateError",
    "ParticleStateLimits",
    "ParticleStrengthLimit",
    "PitchingVLM",
    "RotatingVLM",
    "RunPlan",
    "Samplers",
    "SlipSlabInduction",
    "SmoothRampVLM",
    "StabilizationConfig",
    "StabilizationError",
    "StabilizationManager",
    "StaticVLM",
    "SurfaceSampler",
    "TranslatingVLM",
    "TreecodeInduction",
    "TurbulenceConfig",
    "VLMConfig",
    "VLMMeshConfig",
    "VLMSampler",
    "VLMSurface",
    "VPMCase",
    "VPMSolver",
    "ViscousConfig",
    "VortexParticleSet",
    "vlm",
]
