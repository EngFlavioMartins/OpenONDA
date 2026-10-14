"""Resolved settings consumed by the FVM numerical kernels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from openonda.core.simulation.forcing import VelocityBoundary

from ..motion.kinematics import MeshMotion, mesh_motion_to_dict
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
                output_mode="synchronous",
            )
            # A direct SciPy solve is serial: use a tightly converged GMRES
            # solve instead and never loosen an authored tolerance.
            linear_options: dict[str, object] = {}
            for equation in ("momentum", "pressure"):
                method = (
                    getattr(numerics.linear, f"{equation}_solver") or numerics.linear.linear_solver
                )
                if method == "spsolve":
                    linear_options.update(
                        {
                            f"{equation}_solver": "gmres",
                            f"{equation}_tolerance": min(
                                getattr(numerics.linear, f"{equation}_tolerance"), 1e-10
                            ),
                            f"{equation}_relative_tolerance": 0.0,
                            f"{equation}_final_relative_tolerance": 0.0,
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

        Output, logging, backup and sampler settings and the end time are
        excluded: they may change when a run is extended.
        """
        return {
            "case_name": self.case_name,
            "cores": self.cores,
            "numerics": asdict(self.numerics),
            "start_time": self.run.start_time,
            "mesh_quality": asdict(self.mesh_quality),
            "boundaries": [asdict(boundary) for boundary in self.boundaries],
            "velocity_boundaries": [asdict(item) for item in self.velocity_boundaries],
            "initial_conditions": asdict(self.initial_conditions),
            "mesh_motion": (
                None if self.mesh_motion is None else mesh_motion_to_dict(self.mesh_motion)
            ),
        }
