"""Resolved settings consumed by the FVM numerical kernels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from openonda.core.simulation.forcing import VelocityBoundary

from ..motion.mesh_motion import MeshMotion, mesh_motion_to_dict
from .boundaries import BoundaryConfig
from .case import FVMCase, InitialFields, RunPlan
from .mesh_quality import MeshQualityConfig
from .numerics import Numerics
from .output import Backup, LoggingConfig, Output


@dataclass(frozen=True, slots=True)
class FVMConfiguration:
    """Settings of one FVM run after the execution backends are resolved.

    A multi-rank case selects PETSc, synchronous output and an iterative
    replacement of a serial direct solve. :meth:`from_case` applies these
    rules once; the solver and its kernels read only this object.
    """

    case_name: str
    cores: int
    numerics: Numerics
    run: RunPlan
    output: Output
    logging: LoggingConfig
    backup: Backup
    mesh_quality: MeshQualityConfig
    boundaries: tuple[BoundaryConfig, ...]
    velocity_boundaries: tuple[VelocityBoundary, ...]
    samplers: tuple[object, ...]
    initial_conditions: InitialFields
    mesh_motion: MeshMotion | None

    @classmethod
    def from_case(cls, case: FVMCase) -> FVMConfiguration:
        """Return the resolved settings of ``case``."""
        numerics = case.numerics
        output = case.output
        if case.cores > 1:
            parallel_mode = (
                "petsc_replicated"
                if numerics.execution.parallel_mode == "petsc_replicated"
                or any(boundary.velocity_type == "cyclic" for boundary in case.boundaries)
                else "petsc_partitioned"
            )
            execution = replace(
                numerics.execution,
                linear_backend="petsc",
                parallel_mode=parallel_mode,
            )
            # A direct SciPy solve is serial: use a tightly converged
            # iterative solve instead (BiCGSTAB for momentum, AMG for pressure)
            # and never loosen an authored tolerance.
            linear_options: dict[str, object] = {}
            for equation, solver_field, replacement in (
                ("momentum", "linear_solver", "bicgstab"),
                ("pressure", "pressure_solver", "amg"),
            ):
                method = getattr(numerics.linear, solver_field) or numerics.linear.linear_solver
                if method == "spsolve":
                    linear_options.update(
                        {
                            solver_field: replacement,
                            f"{equation}_tolerance": min(
                                getattr(numerics.linear, f"{equation}_tolerance"), 1e-10
                            ),
                            f"{equation}_relative_tolerance": 0.0,
                        }
                    )
            numerics = replace(
                numerics,
                execution=execution,
                linear=replace(numerics.linear, **linear_options),
            )
            output = replace(output, asynchronous=False)
        return cls(
            case_name=case.name,
            cores=case.cores,
            numerics=numerics,
            run=case.run,
            output=output,
            logging=case.logging,
            backup=case.backup,
            mesh_quality=case.mesh_quality,
            boundaries=case.boundaries,
            velocity_boundaries=case.velocity_boundaries,
            samplers=case.samplers.samples,
            initial_conditions=case.initial_conditions,
            mesh_motion=case.mesh_motion,
        )

    def algorithm_parameters(self) -> dict[str, Any]:
        """Return the flat scheme, linear-solver and coupling settings of the kernels."""
        parameters: dict[str, Any] = {}
        for group in (self.numerics.schemes, self.numerics.linear, self.numerics.pimple):
            parameters.update(asdict(group))
        return parameters

    def restart_configuration(self) -> dict[str, Any]:
        """Return the settings a restart backup must match.

        Only settings that change the computed solution are included. The
        execution backend, the linear-solver options, the output, logging,
        backup and sampler settings, the step-acceptance and mesh-quality
        limits, the core count and the end time may change when a run is
        continued. The initial fields are replaced by the saved state. A
        boundary velocity that is a Python function is identified by its
        qualified name.
        """
        numerics = self.numerics
        return {
            "time_step_size": numerics.time_step_size,
            "time_step_control": (
                None if numerics.time_step_control is None else asdict(numerics.time_step_control)
            ),
            "transport": asdict(numerics.transport),
            "schemes": asdict(numerics.schemes),
            "pimple": asdict(numerics.pimple),
            "turbulence": asdict(numerics.turbulence),
            "start_time": self.run.start_time,
            "boundaries": [asdict(boundary) for boundary in self.boundaries],
            "velocity_boundaries": [asdict(item) for item in self.velocity_boundaries],
            "mesh_motion": (
                None if self.mesh_motion is None else mesh_motion_to_dict(self.mesh_motion)
            ),
        }
