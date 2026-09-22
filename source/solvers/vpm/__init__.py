"""Vortex-particle and vortex-lattice solvers for OpenONDA."""

import os

# OpenONDA reports its backend after solver ownership is established. Taichi's
# import-time banner would otherwise repeat in the launcher and every MPI rank.
os.environ.setdefault("ENABLE_TAICHI_HEADER_PRINT", "False")

from .boundary_elements import vlm
from .boundary_elements.vlm.config import (
    ForceConfig,
    VLMMeshSetup,
    VLMSetup,
    VLMSurfaceSetup,
)
from .config import (
    Backup,
    DiagnosticsConfig,
    DivergenceLimit,
    DivergenceRelaxationConfig,
    FilamentRefinementConfig,
    FiniteStateCheck,
    GrowthLimit,
    HealthError,
    HealthLimits,
    LagrangianCFLLimit,
    MisalignmentLimit,
    Numerics,
    ParticleStrengthLimit,
    ResourceLimitError,
    ResourceLimits,
    RestartState,
    RunPlan,
    Samplers,
    StabilizationConfig,
    TurbulenceConfig,
    ViscousConfig,
    VPMCase,
)
from .core.solver import VPMSolver
from .diagnostics import FlowIntegralsSampler, RingDiagnosticsSampler
from .initialization import (
    CylindricalDistribution,
    FilamentDisturbance,
    InitialCondition,
    InitialVelocity,
    IsotropicTurbulence,
    NoisyRectangularDistribution,
    ParticleCoreCompensation,
    ParticleDistribution,
    RectangularDistribution,
    TaylorGreenVortex,
    ToroidalDistribution,
    TriangularPrismDistribution,
    VortexDoublet,
    VortexFilament,
    VortexParticleSet,
    VortexRing,
    WidnallDisturbance,
)
from .io.sampling import EverySteps, EveryTime, FinalOnly, VLMSampler
from .numerics.rk_tableaux import RK2, RK4, SSPRK3
from .physics.induction.direct import DirectInduction
from .physics.induction.fmm import FMMInduction
from .physics.induction.planar import PlanarInduction
from .physics.induction.slip_slab import SlipSlabInduction
from .physics.induction.treecode import TreecodeInduction
from .stabilization import (
    DivergenceRelaxationError,
    FilamentRefinementError,
    StabilizationError,
    StabilizationManager,
)

__all__ = [
    "DirectInduction",
    "PlanarInduction",
    "SlipSlabInduction",
    "FMMInduction",
    "DivergenceRelaxationConfig",
    "DivergenceRelaxationError",
    "Backup",
    "DivergenceLimit",
    "DiagnosticsConfig",
    "FinalOnly",
    "FilamentRefinementConfig",
    "FilamentRefinementError",
    "FiniteStateCheck",
    "FilamentDisturbance",
    "InitialVelocity",
    "InitialCondition",
    "ForceConfig",
    "FlowIntegralsSampler",
    "IsotropicTurbulence",
    "EverySteps",
    "EveryTime",
    "GrowthLimit",
    "HealthError",
    "HealthLimits",
    "LagrangianCFLLimit",
    "ParticleStrengthLimit",
    "ResourceLimitError",
    "ResourceLimits",
    "MisalignmentLimit",
    "Numerics",
    "RestartState",
    "ParticleDistribution",
    "ParticleCoreCompensation",
    "Samplers",
    "RingDiagnosticsSampler",
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
    "WidnallDisturbance",
    "CylindricalDistribution",
    "NoisyRectangularDistribution",
    "RectangularDistribution",
    "ToroidalDistribution",
    "TriangularPrismDistribution",
    "TaylorGreenVortex",
    "VortexDoublet",
    "VortexFilament",
    "VortexRing",
    "vlm",
]
