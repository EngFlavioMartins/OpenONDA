"""Public finite-volume solver API.

Meshing construction lives in :mod:`openonda.fvm.mesher` so that solver
configuration and mesh-generation intent remain separate namespaces::

    import openonda.fvm as fvm
    import openonda.fvm.mesher as msh

    mesh = msh.CartesianMesher(
        domain=msh.BoxDomain(...),
        surfaces=(msh.STLSurface(...),),
    )
"""

from source.solvers.fvm import (
    BackupConfig,
    BoundaryConfig,
    ComputeConfig,
    DiscretizationConfig,
    FieldState,
    ForceSampler,
    FVMSetup,
    IBMForceSampler,
    LinearSolverConfig,
    LinearSolveResult,
    LineSampler,
    LoggingConfig,
    MaximumCourantTimeStep,
    MeshMotionConfig,
    MeshQualityConfig,
    OutputConfig,
    PimpleControl,
    RunAcceptanceLimits,
    RunSchedule,
    StepDiagnostics,
    SurfaceSampler,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
    YPlusSampler,
    analyse_grid_study,
    create_fvm_solver,
    update_grid_study,
)
from source.solvers.fvm.fields.diagnostics import (
    compute_continuity_error,
    compute_enstrophy,
    compute_kinetic_energy,
)
from source.solvers.fvm.immersed_boundary import ImmersedBody

from . import mesher

__all__ = [
    "BackupConfig",
    "BoundaryConfig",
    "ComputeConfig",
    "DiscretizationConfig",
    "FieldState",
    "ForceSampler",
    "FVMSetup",
    "IBMForceSampler",
    "ImmersedBody",
    "LinearSolveResult",
    "LinearSolverConfig",
    "LineSampler",
    "LoggingConfig",
    "MaximumCourantTimeStep",
    "MeshMotionConfig",
    "MeshQualityConfig",
    "OutputConfig",
    "PimpleControl",
    "RunAcceptanceLimits",
    "RunSchedule",
    "StepDiagnostics",
    "SurfaceSampler",
    "TimeConfig",
    "TransportConfig",
    "TurbulenceConfig",
    "YPlusSampler",
    "compute_continuity_error",
    "compute_enstrophy",
    "compute_kinetic_energy",
    "analyse_grid_study",
    "create_fvm_solver",
    "mesher",
    "update_grid_study",
]
