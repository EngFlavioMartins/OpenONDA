"""Vortex-particle and vortex-lattice solvers for OpenONDA."""

from openonda.core.kinematics import (
    CompositeMotion,
    HeavingMotion,
    LinearPeriodicMotion,
    ManeuverMotion,
    PitchingMotion,
    RigidMotion,
    RigidPose,
    RotatingMotion,
    SmoothRampMotion,
    StaticMotion,
    TranslatingMotion,
)
from openonda.core.schedules import EverySteps, EveryTime, FinalOnly

from .boundary_elements import vlm
from .boundary_elements.vlm.config import ForceConfig, VLMConfig, VLMMeshConfig, VLMSurface
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
    "CompositeMotion",
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
    "HeavingMotion",
    "InitialCondition",
    "LagrangianCFLLimit",
    "LineSampler",
    "LinearPeriodicMotion",
    "LoggingConfig",
    "ManeuverMotion",
    "MisalignmentLimit",
    "Numerics",
    "Output",
    "ParticleDistribution",
    "ParticleStateError",
    "ParticleStateLimits",
    "ParticleStrengthLimit",
    "PitchingMotion",
    "RigidMotion",
    "RigidPose",
    "RotatingMotion",
    "RunPlan",
    "Samplers",
    "SlipSlabInduction",
    "SmoothRampMotion",
    "StabilizationConfig",
    "StabilizationError",
    "StabilizationManager",
    "StaticMotion",
    "SurfaceSampler",
    "TranslatingMotion",
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
