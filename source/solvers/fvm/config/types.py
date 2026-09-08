"""Configuration models for the incompressible FVM solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Real
from typing import Any, Literal

from source.write_precision import (
    DEFAULT_WRITE_PRECISION,
    WritePrecision,
    validate_write_precision,
)

from .scheduling import RunSchedule


def _finite_real(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    """Validate and normalize one scalar configuration value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = normalized <= minimum if strict_minimum else normalized < minimum
        if invalid:
            relation = ">" if strict_minimum else ">="
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return normalized


def _validate_vector(name: str, value: Any, *, allow_per_face: bool = True) -> None:
    """Validate a vector or per-face vector without retaining an input alias."""
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    valid = array.shape == (3,)
    if allow_per_face:
        valid = valid or (array.ndim == 2 and array.shape[1:] == (3,))
    if not valid:
        suffix = " or (n_faces, 3)" if allow_per_face else ""
        raise ValueError(f"{name} must have shape (3,){suffix}; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")


def _strict_int(name: str, value: Any, *, minimum: int = 0) -> int:
    """Validate an integer count while rejecting bools and fractional values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


@dataclass
class BoundaryConfig:
    """Boundary-condition specification for one mesh patch."""

    name: str
    velocity_type: str = "fixedValue"
    velocity_value: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pressure_type: str = "zeroGradient"
    kinematic_pressure_value: float = 0.0
    flux_type: str = "zeroGradient"
    flux_value: float = 0.0
    eddy_viscosity_type: str = "calculated"
    eddy_viscosity_value: float = 0.0
    neighbour_patch: str | None = None
    mesh_type: Literal["patch", "wall", "empty", "cyclic"] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("BoundaryConfig.name must be a non-empty string")
        for name in (
            "velocity_type",
            "pressure_type",
            "flux_type",
            "eddy_viscosity_type",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"BoundaryConfig.{name} must be a non-empty string")
        _validate_vector("BoundaryConfig.velocity_value", self.velocity_value)
        for name in (
            "kinematic_pressure_value",
            "flux_value",
            "eddy_viscosity_value",
        ):
            setattr(self, name, _finite_real(f"BoundaryConfig.{name}", getattr(self, name)))
        if self.neighbour_patch is not None and (
            not isinstance(self.neighbour_patch, str) or not self.neighbour_patch.strip()
        ):
            raise ValueError("BoundaryConfig.neighbour_patch must be a non-empty string or None")
        if self.mesh_type is not None and self.mesh_type not in {
            "patch",
            "wall",
            "empty",
            "cyclic",
        }:
            raise ValueError("BoundaryConfig.mesh_type must be patch, wall, empty, cyclic, or None")

    @staticmethod
    def inlet(
        name: str,
        velocity: list[float],
    ) -> BoundaryConfig:
        """Return a fixed-velocity inlet."""
        return BoundaryConfig(
            name=name,
            velocity_type="fixedValue",
            velocity_value=velocity,
            pressure_type="zeroGradient",
        )

    @staticmethod
    def outlet(
        name: str,
        kinematic_pressure: float = 0.0,
    ) -> BoundaryConfig:
        """Return an outlet with fixed kinematic pressure."""
        return BoundaryConfig(
            name=name,
            velocity_type="inletOutlet",
            pressure_type="fixedValue",
            kinematic_pressure_value=kinematic_pressure,
        )

    @staticmethod
    def freestream(
        name: str,
        velocity: list[float],
        kinematic_pressure: float = 0.0,
    ) -> BoundaryConfig:
        """Return an incompressible external-flow boundary."""
        return BoundaryConfig(
            name=name,
            velocity_type="freestream",
            velocity_value=velocity,
            pressure_type="freestream",
            kinematic_pressure_value=kinematic_pressure,
        )

    @staticmethod
    def cyclic(
        name: str,
        neighbour_patch: str,
    ) -> BoundaryConfig:
        """Return one side of a periodic patch pair."""
        return BoundaryConfig(
            name=name,
            velocity_type="cyclic",
            pressure_type="cyclic",
            neighbour_patch=neighbour_patch,
            mesh_type="cyclic",
        )

    @staticmethod
    def wall(name: str) -> BoundaryConfig:
        """Return a no-slip wall."""
        return BoundaryConfig(
            name=name,
            velocity_type="fixedValue",
            velocity_value=[0.0, 0.0, 0.0],
            pressure_type="zeroGradient",
            eddy_viscosity_type="calculated",
            mesh_type="wall",
        )

    @staticmethod
    def slip(name: str) -> BoundaryConfig:
        """Return an impermeable zero-shear boundary."""
        return BoundaryConfig(
            name=name,
            velocity_type="slip",
            pressure_type="zeroGradient",
            eddy_viscosity_type="zeroGradient",
            mesh_type="patch",
        )

    @staticmethod
    def empty(name: str) -> BoundaryConfig:
        """Return an empty patch for an extruded two-dimensional mesh."""
        return BoundaryConfig(
            name=name,
            velocity_type="empty",
            velocity_value=[0.0, 0.0, 0.0],
            pressure_type="empty",
            eddy_viscosity_type="zeroGradient",
            mesh_type="empty",
        )


@dataclass
class MeshQualityConfig:
    """Mesh-quality limits applied during solver construction."""

    max_non_orthogonality_deg: float | None = None
    max_skewness: float | None = None
    max_aspect_ratio: float | None = None
    max_lsq_condition: float | None = None

    def __post_init__(self) -> None:
        for name, minimum, strict in (
            ("max_non_orthogonality_deg", 0.0, False),
            ("max_skewness", 0.0, True),
            ("max_aspect_ratio", 0.0, True),
            ("max_lsq_condition", 0.0, True),
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(
                    self,
                    name,
                    _finite_real(
                        f"MeshQualityConfig.{name}",
                        value,
                        minimum=minimum,
                        strict_minimum=strict,
                    ),
                )


@dataclass(frozen=True, slots=True)
class MaximumCourantTimeStep:
    """Automatic FVM time-step control based on the maximum Courant number.

    ``maximum`` is the target cell Courant number.  Before every transient
    solve, the solver measures the Courant number of the current face-flux
    field and selects the next time step with OpenFOAM's damped adjustment:
    reductions may take effect immediately, while increases are limited to
    twenty percent per accepted step and are additionally damped near the
    target.  ``maximum_time_step_size`` provides the optional ``maxDeltaT``
    equivalent in seconds.

    This object and :class:`TimeConfig` are immutable.  Time-step policy is a
    numerical construction choice: configure it before creating the solver;
    the solver alone owns the evolving runtime time-step size afterward.
    """

    maximum: float = 0.9
    maximum_time_step_size: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.maximum, bool) or not isinstance(self.maximum, Real):
            raise TypeError("MaximumCourantTimeStep.maximum must be a real number")
        if not math.isfinite(self.maximum) or self.maximum <= 0.0:
            raise ValueError("MaximumCourantTimeStep.maximum must be finite and positive")
        object.__setattr__(self, "maximum", float(self.maximum))

        maximum_time_step_size = self.maximum_time_step_size
        if maximum_time_step_size is None:
            return
        if isinstance(maximum_time_step_size, bool) or not isinstance(maximum_time_step_size, Real):
            raise TypeError(
                "MaximumCourantTimeStep.maximum_time_step_size must be a real number or None"
            )
        if not math.isfinite(maximum_time_step_size) or maximum_time_step_size <= 0.0:
            raise ValueError(
                "MaximumCourantTimeStep.maximum_time_step_size must be finite and positive"
            )
        object.__setattr__(self, "maximum_time_step_size", float(maximum_time_step_size))


@dataclass(frozen=True, slots=True)
class TimeConfig:
    """Immutable time integration, output cadence, and step-control policy.

    ``time_step_size`` is the initial step size.  Fixed stepping is used when
    ``adjustment`` is ``None``; pass :class:`MaximumCourantTimeStep` to make
    step selection a solver-owned maximum-Courant policy.
    """

    time_step_size: float = 0.01
    start_time: float = 0.0
    end_time: float = 1.0
    output_schedule: RunSchedule = field(default_factory=lambda: RunSchedule(every_n_steps=10))
    adjustment: MaximumCourantTimeStep | None = None

    def __post_init__(self) -> None:
        for name in ("time_step_size", "start_time", "end_time"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"TimeConfig.{name} must be a real number")
            if not math.isfinite(value):
                raise ValueError(f"TimeConfig.{name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.time_step_size <= 0.0:
            raise ValueError("TimeConfig.time_step_size must be positive")
        if self.end_time <= self.start_time:
            raise ValueError("TimeConfig.end_time must be greater than start_time")
        if self.adjustment is not None and not isinstance(self.adjustment, MaximumCourantTimeStep):
            raise TypeError(
                "TimeConfig.adjustment must be a MaximumCourantTimeStep instance or None"
            )
        if not isinstance(self.output_schedule, RunSchedule):
            raise TypeError("TimeConfig.output_schedule must be a RunSchedule")


@dataclass
class DiscretizationConfig:
    """Spatial and temporal discretisation settings."""

    convection_scheme: Literal[
        "upwind",
        "central",
        "limitedLinear",
        "LUST",
        "linearUpwind",
        "vanLeer",
        "MUSCL",
        "minmod",
        "superbee",
    ] = "limitedLinear"
    gradient_scheme: Literal["gauss", "lsq"] = "lsq"
    time_scheme: Literal["euler_implicit", "backward"] = "euler_implicit"

    def __post_init__(self) -> None:
        convection = str(self.convection_scheme).lower()
        if convection not in {
            "upwind",
            "central",
            "linear",
            "deferred",
            "lust",
            "linearupwind",
            "vanleer",
            "minmod",
            "muscl",
            "superbee",
            "limitedlinear",
        }:
            raise ValueError(f"Unsupported convection_scheme={self.convection_scheme!r}")
        gradient = str(self.gradient_scheme).lower()
        if gradient not in {"gauss", "lsq"}:
            raise ValueError(f"Unsupported gradient_scheme={self.gradient_scheme!r}")
        time_scheme = str(self.time_scheme).lower()
        if time_scheme not in {
            "euler",
            "euler_implicit",
            "backward_euler",
            "backward",
            "bdf2",
        }:
            raise ValueError(f"Unsupported time_scheme={self.time_scheme!r}")
        self.convection_scheme = convection
        self.gradient_scheme = gradient
        self.time_scheme = time_scheme


@dataclass
class LinearSolverConfig:
    """Momentum and pressure linear-solver settings."""

    linear_solver: Literal[
        "bicgstab",
        "gmres",
        "cg",
        "amg",
        "spsolve",
    ] = "bicgstab"
    momentum_solver: (
        Literal[
            "bicgstab",
            "gmres",
            "cg",
            "spsolve",
        ]
        | None
    ) = None
    pressure_solver: (
        Literal[
            "amg",
            "bicgstab",
            "gmres",
            "cg",
            "spsolve",
        ]
        | None
    ) = None
    pressure_nullspace_method: Literal[
        "auto",
        "reference",
        "petsc",
    ] = "auto"
    linear_failure_action: Literal[
        "raise",
        "direct_fallback",
    ] = "raise"
    reuse_ilu: bool = True

    momentum_tolerance: float = 1e-4
    momentum_relative_tolerance: float = 0.0
    momentum_final_relative_tolerance: float | None = 0.0
    momentum_max_iterations: int = 1000

    pressure_tolerance: float = 1e-8
    pressure_relative_tolerance: float = 0.0
    pressure_final_relative_tolerance: float | None = 0.0
    pressure_max_iterations: int = 500

    amg_tolerance: float | None = None
    amg_max_iterations: int | None = None
    amg_reuse_tolerance: float = 0.05

    ilu_drop_tolerance: float = 1e-4
    ilu_fill_factor: float = 10.0
    ilu_reuse_tolerance: float | None = None

    def __post_init__(self) -> None:
        valid = {"bicgstab", "gmres", "cg", "amg", "spsolve"}
        for name in ("linear_solver", "momentum_solver", "pressure_solver"):
            value = getattr(self, name)
            if value is None and name != "linear_solver":
                continue
            choice = str(value).lower()
            if choice not in valid:
                raise ValueError(f"Unsupported LinearSolverConfig.{name}={value!r}")
            setattr(self, name, choice)
        if self.momentum_solver == "amg" or (
            self.momentum_solver is None and self.linear_solver == "amg"
        ):
            raise ValueError("LinearSolverConfig effective momentum solver cannot be 'amg'")
        nullspace = str(self.pressure_nullspace_method).lower()
        if nullspace not in {"auto", "reference", "petsc"}:
            raise ValueError(
                "LinearSolverConfig.pressure_nullspace_method must be auto, reference, or petsc"
            )
        self.pressure_nullspace_method = nullspace
        failure_action = str(self.linear_failure_action).lower()
        if failure_action not in {"raise", "direct_fallback"}:
            raise ValueError(
                "LinearSolverConfig.linear_failure_action must be raise or direct_fallback"
            )
        self.linear_failure_action = failure_action
        if not isinstance(self.reuse_ilu, bool):
            raise TypeError("LinearSolverConfig.reuse_ilu must be a boolean")
        for name in (
            "momentum_tolerance",
            "pressure_tolerance",
            "ilu_drop_tolerance",
            "ilu_fill_factor",
            "amg_reuse_tolerance",
        ):
            setattr(
                self,
                name,
                _finite_real(
                    f"LinearSolverConfig.{name}",
                    getattr(self, name),
                    minimum=0.0,
                    strict_minimum=True,
                ),
            )
        for name in ("momentum_relative_tolerance", "pressure_relative_tolerance"):
            setattr(
                self,
                name,
                _finite_real(f"LinearSolverConfig.{name}", getattr(self, name), minimum=0.0),
            )
        for name in (
            "momentum_final_relative_tolerance",
            "pressure_final_relative_tolerance",
            "ilu_reuse_tolerance",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(
                    self,
                    name,
                    _finite_real(f"LinearSolverConfig.{name}", value, minimum=0.0),
                )
        if self.amg_tolerance is not None:
            self.amg_tolerance = _finite_real(
                "LinearSolverConfig.amg_tolerance",
                self.amg_tolerance,
                minimum=0.0,
                strict_minimum=True,
            )
        for name in ("momentum_max_iterations", "pressure_max_iterations"):
            setattr(
                self,
                name,
                _strict_int(f"LinearSolverConfig.{name}", getattr(self, name), minimum=1),
            )
        if self.amg_max_iterations is not None:
            self.amg_max_iterations = _strict_int(
                "LinearSolverConfig.amg_max_iterations",
                self.amg_max_iterations,
                minimum=1,
            )


@dataclass
class PimpleControl:
    """PIMPLE, PISO, or SIMPLE pressure-velocity coupling controls."""

    algorithm: Literal["SIMPLE", "PIMPLE", "PISO"] = "PIMPLE"
    n_correctors: int = 2
    n_outer_correctors: int = 1
    n_orthogonal_correctors: int = 0
    n_nonorthogonal_correctors: int | None = None
    min_outer_correctors: int = 1
    outer_residual_tolerance: float | None = None
    outer_continuity_tolerance: float | None = None
    max_iterations: int = 20
    tolerance: float = 1e-6
    velocity_relaxation: float = 1.0
    pressure_relaxation: float = 1.0
    ddt_corr: bool = True
    ibm_forcing_loops: int = 2
    ibm_second_solve: bool = True

    def __post_init__(self) -> None:
        algorithm = str(self.algorithm).upper()
        if algorithm not in {"SIMPLE", "PIMPLE", "PISO"}:
            raise ValueError(f"Unsupported PimpleControl.algorithm={self.algorithm!r}")
        self.algorithm = algorithm
        old_value = _strict_int(
            "PimpleControl.n_orthogonal_correctors",
            self.n_orthogonal_correctors,
            minimum=0,
        )
        if self.n_nonorthogonal_correctors is not None:
            new_value = _strict_int(
                "PimpleControl.n_nonorthogonal_correctors",
                self.n_nonorthogonal_correctors,
                minimum=0,
            )
            if old_value != 0 and old_value != new_value:
                raise ValueError(
                    "PimpleControl.n_orthogonal_correctors and n_nonorthogonal_correctors disagree"
                )
            old_value = new_value
        self.n_orthogonal_correctors = old_value
        self.n_nonorthogonal_correctors = old_value
        for name in (
            "n_correctors",
            "n_outer_correctors",
            "min_outer_correctors",
            "max_iterations",
            "ibm_forcing_loops",
        ):
            setattr(
                self,
                name,
                _strict_int(f"PimpleControl.{name}", getattr(self, name), minimum=1),
            )
        if self.min_outer_correctors > self.n_outer_correctors:
            raise ValueError("PimpleControl.min_outer_correctors cannot exceed n_outer_correctors")
        self.tolerance = _finite_real(
            "PimpleControl.tolerance", self.tolerance, minimum=0.0, strict_minimum=True
        )
        for name in ("velocity_relaxation", "pressure_relaxation"):
            setattr(
                self,
                name,
                _finite_real(
                    f"PimpleControl.{name}",
                    getattr(self, name),
                    minimum=0.0,
                    maximum=1.0,
                    strict_minimum=True,
                ),
            )
        for name in ("outer_residual_tolerance", "outer_continuity_tolerance"):
            value = getattr(self, name)
            if value is not None:
                setattr(
                    self,
                    name,
                    _finite_real(f"PimpleControl.{name}", value, minimum=0.0, strict_minimum=True),
                )
        if not isinstance(self.ddt_corr, bool):
            raise TypeError("PimpleControl.ddt_corr must be a boolean")
        if not isinstance(self.ibm_second_solve, bool):
            raise TypeError("PimpleControl.ibm_second_solve must be a boolean")


@dataclass
class TransportConfig:
    """Fluid density and molecular kinematic viscosity."""

    density: float = 1.225
    kinematic_viscosity: float = 1.5e-5

    def __post_init__(self) -> None:
        self.density = _finite_real(
            "TransportConfig.density", self.density, minimum=0.0, strict_minimum=True
        )
        self.kinematic_viscosity = _finite_real(
            "TransportConfig.kinematic_viscosity",
            self.kinematic_viscosity,
            minimum=0.0,
            strict_minimum=True,
        )

    @staticmethod
    def air() -> TransportConfig:
        """Return standard sea-level air properties."""
        return TransportConfig(
            density=1.225,
            kinematic_viscosity=1.5e-5,
        )

    @staticmethod
    def water() -> TransportConfig:
        """Return fresh-water properties near 20 degrees Celsius."""
        return TransportConfig(
            density=1000.0,
            kinematic_viscosity=1.0e-6,
        )


@dataclass
class MeshMotionConfig:
    """Rigid-body mesh motion or a static mesh."""

    method: Literal["static", "rigidMotion"] = "static"
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_speed: float = 0.0
    axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    origin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self) -> None:
        method = str(self.method)
        if method not in {"static", "rigidMotion"}:
            raise ValueError("MeshMotionConfig.method must be 'static' or 'rigidMotion'")
        self.method = method
        _validate_vector("MeshMotionConfig.velocity", self.velocity, allow_per_face=False)
        _validate_vector("MeshMotionConfig.axis", self.axis, allow_per_face=False)
        _validate_vector("MeshMotionConfig.origin", self.origin, allow_per_face=False)
        self.angular_speed = _finite_real(
            "MeshMotionConfig.angular_speed", self.angular_speed, minimum=0.0
        )
        import numpy as np

        if method == "rigidMotion" and np.linalg.norm(np.asarray(self.axis, dtype=float)) == 0.0:
            raise ValueError("MeshMotionConfig.axis must be non-zero for rigidMotion")

    @staticmethod
    def static() -> MeshMotionConfig:
        """Return static mesh motion."""
        return MeshMotionConfig(method="static")

    @staticmethod
    def rigid(
        velocity: list[float] | None = None,
        angular_speed: float = 0.0,
        axis: list[float] | None = None,
        origin: list[float] | None = None,
    ) -> MeshMotionConfig:
        """Return rigid translation/rotation mesh motion."""
        return MeshMotionConfig(
            method="rigidMotion",
            velocity=[0.0, 0.0, 0.0] if velocity is None else velocity,
            angular_speed=angular_speed,
            axis=[0.0, 0.0, 1.0] if axis is None else axis,
            origin=[0.0, 0.0, 0.0] if origin is None else origin,
        )


@dataclass
class TurbulenceConfig:
    """LES/SGS model configuration with model-specific coefficients."""

    model: str = "None"
    smagorinsky_coefficient: float = 0.17
    subgrid_kinetic_energy_coefficient: float = 0.094
    subgrid_dissipation_coefficient: float = 1.048
    wale_coefficient: float = 0.325
    sigma_coefficient: float = 1.35
    dynamic: bool = False

    def __post_init__(self) -> None:
        valid_models = {
            "none",
            "iles",
            "dns",
            "smagorinsky",
            "equilibriumsmagorinsky",
            "equilibrium_smagorinsky",
            "wale",
            "sigma",
            "dynamicsmagorinsky",
            "dynamic_smagorinsky",
        }
        model = str(self.model).lower()
        if model not in valid_models:
            raise ValueError(f"Unsupported TurbulenceConfig.model={self.model!r}")
        self.model = model
        for name, minimum, strict in (
            ("smagorinsky_coefficient", 0.0, False),
            ("subgrid_kinetic_energy_coefficient", 0.0, False),
            ("subgrid_dissipation_coefficient", 0.0, True),
            ("wale_coefficient", 0.0, True),
            ("sigma_coefficient", 0.0, True),
        ):
            setattr(
                self,
                name,
                _finite_real(
                    f"TurbulenceConfig.{name}",
                    getattr(self, name),
                    minimum=minimum,
                    strict_minimum=strict,
                ),
            )
        if not isinstance(self.dynamic, bool):
            raise TypeError("TurbulenceConfig.dynamic must be a boolean")

    @staticmethod
    def smagorinsky(
        smagorinsky_coefficient: float = 0.17,
        dynamic: bool = False,
    ) -> TurbulenceConfig:
        """Return classical or dynamic Smagorinsky configuration."""
        return TurbulenceConfig(
            model="Smagorinsky",
            smagorinsky_coefficient=smagorinsky_coefficient,
            dynamic=dynamic,
        )

    @staticmethod
    def equilibrium_smagorinsky(
        subgrid_kinetic_energy_coefficient: float = 0.094,
        subgrid_dissipation_coefficient: float = 1.048,
    ) -> TurbulenceConfig:
        """Return algebraic-equilibrium Smagorinsky configuration."""
        equivalent_smagorinsky_coefficient = (
            subgrid_kinetic_energy_coefficient**0.75 / subgrid_dissipation_coefficient**0.25
            if subgrid_kinetic_energy_coefficient >= 0.0 and subgrid_dissipation_coefficient > 0.0
            else float("nan")
        )
        return TurbulenceConfig(
            model="EquilibriumSmagorinsky",
            smagorinsky_coefficient=equivalent_smagorinsky_coefficient,
            subgrid_kinetic_energy_coefficient=subgrid_kinetic_energy_coefficient,
            subgrid_dissipation_coefficient=subgrid_dissipation_coefficient,
        )

    @staticmethod
    def wale(
        wale_coefficient: float = 0.325,
    ) -> TurbulenceConfig:
        """Return WALE configuration."""
        return TurbulenceConfig(
            model="WALE",
            wale_coefficient=wale_coefficient,
        )

    @staticmethod
    def sigma(
        sigma_coefficient: float = 1.35,
    ) -> TurbulenceConfig:
        """Return sigma-model configuration."""
        return TurbulenceConfig(
            model="sigma",
            sigma_coefficient=sigma_coefficient,
        )

    @staticmethod
    def dynamic_smagorinsky() -> TurbulenceConfig:
        """Return Germano-Lilly dynamic Smagorinsky configuration."""
        return TurbulenceConfig(
            model="dynamicSmagorinsky",
            dynamic=True,
        )

    @staticmethod
    def none() -> TurbulenceConfig:
        """Return configuration without an explicit SGS model."""
        return TurbulenceConfig(model="None")


@dataclass
class ComputeConfig:
    """Sparse assembly, linear algebra, parallelism, and output execution."""

    operator_backend: Literal[
        "numpy",
        "numba",
        "taichi",
    ] = "numpy"
    linear_backend: Literal[
        "scipy",
        "petsc",
    ] = "scipy"
    parallel_mode: Literal[
        "serial",
        "petsc_replicated",
        "petsc_partitioned",
    ] = "serial"
    output_mode: Literal[
        "synchronous",
        "threaded",
    ] = "synchronous"

    def __post_init__(self) -> None:
        choices = {
            "operator_backend": {"numpy", "numba", "taichi"},
            "linear_backend": {"scipy", "petsc"},
            "parallel_mode": {"serial", "petsc_replicated", "petsc_partitioned"},
            "output_mode": {"synchronous", "threaded"},
        }
        for name, valid in choices.items():
            value = str(getattr(self, name)).lower()
            if value not in valid:
                raise ValueError(f"Unsupported ComputeConfig.{name}={getattr(self, name)!r}")
            setattr(self, name, value)
        if self.parallel_mode == "serial" and self.linear_backend != "scipy":
            raise ValueError("ComputeConfig serial mode requires linear_backend='scipy'")
        if self.parallel_mode != "serial" and self.linear_backend != "petsc":
            raise ValueError(
                f"ComputeConfig.parallel_mode={self.parallel_mode!r} requires linear_backend='petsc'"
            )
        if self.parallel_mode == "petsc_partitioned" and self.output_mode == "threaded":
            raise ValueError(
                "ComputeConfig petsc_partitioned mode does not support threaded output"
            )

    @staticmethod
    def petsc_replicated() -> ComputeConfig:
        """Return replicated PETSc execution."""
        return ComputeConfig(
            linear_backend="petsc",
            parallel_mode="petsc_replicated",
        )

    @staticmethod
    def petsc_partitioned() -> ComputeConfig:
        """Return partitioned PETSc execution."""
        return ComputeConfig(
            linear_backend="petsc",
            parallel_mode="petsc_partitioned",
        )


@dataclass
class OutputConfig:
    """ParaView visualization-output policy."""

    format: Literal["vtk_xml"] = "vtk_xml"
    data_location: Literal["cell"] = "cell"
    encoding: Literal["appended"] = "appended"
    compression: Literal[
        "lz4",
        "none",
        "zlib",
    ] = "zlib"
    precision: WritePrecision = DEFAULT_WRITE_PRECISION
    asynchronous: bool = True
    ghost_layers: Literal[0, 1] = 1
    point_interpolation: Literal[
        "none",
        "boundary_weighted",
    ] = "none"

    def __post_init__(self) -> None:
        if self.format != "vtk_xml":
            raise ValueError("Only format='vtk_xml' is currently supported")
        if self.data_location != "cell":
            raise ValueError("FVM visualization output must remain cell-centred")
        if self.encoding != "appended":
            raise ValueError("Only appended-binary VTK encoding is supported")
        if self.compression not in {"lz4", "none", "zlib"}:
            raise ValueError("compression must be 'lz4', 'none', or 'zlib'")
        validate_write_precision(self.precision, field_name="output precision")
        if not isinstance(self.asynchronous, bool):
            raise TypeError("asynchronous must be a boolean")
        if isinstance(self.ghost_layers, bool) or not isinstance(self.ghost_layers, int):
            raise TypeError("ghost_layers must be an integer")
        if self.ghost_layers not in {0, 1}:
            raise ValueError("ghost_layers must be zero or one")
        if self.point_interpolation not in {
            "none",
            "boundary_weighted",
        }:
            raise ValueError("point_interpolation must be 'none' or 'boundary_weighted'")


@dataclass
class RunAcceptanceLimits:
    """Warning and abort thresholds for structured step diagnostics."""

    sustained_steps: int = 1
    max_continuity_error_warning: float | None = None
    max_continuity_error_abort: float | None = None
    max_equation_residual_warning: float | None = None
    max_equation_residual_abort: float | None = None
    max_courant_number_warning: float | None = None
    max_courant_number_abort: float | None = None
    max_velocity_magnitude_warning: float | None = None
    max_velocity_magnitude_abort: float | None = None

    def __post_init__(self) -> None:
        self.sustained_steps = _strict_int(
            "RunAcceptanceLimits.sustained_steps", self.sustained_steps, minimum=1
        )
        for metric in (
            "max_continuity_error",
            "max_equation_residual",
            "max_courant_number",
            "max_velocity_magnitude",
        ):
            warning_name = f"{metric}_warning"
            abort_name = f"{metric}_abort"
            warning = getattr(self, warning_name)
            abort = getattr(self, abort_name)
            if warning is not None:
                warning = _finite_real(
                    f"RunAcceptanceLimits.{warning_name}", warning, minimum=0.0, strict_minimum=True
                )
            if abort is not None:
                abort = _finite_real(
                    f"RunAcceptanceLimits.{abort_name}", abort, minimum=0.0, strict_minimum=True
                )
            if warning is not None and abort is not None and warning > abort:
                raise ValueError(f"RunAcceptanceLimits.{warning_name} cannot exceed {abort_name}")
            setattr(self, warning_name, warning)
            setattr(self, abort_name, abort)


@dataclass
class LoggingConfig:
    """Console and log-file verbosity with step- or time-based reporting."""

    mode: Literal["simple", "debug"] = "simple"
    schedule: RunSchedule = field(default_factory=lambda: RunSchedule(every_n_steps=1))
    console: bool = True
    filename: str = "fvm.log"

    def __post_init__(self) -> None:
        if self.mode not in {"simple", "debug"}:
            raise ValueError("log mode must be 'simple' or 'debug'")
        if not isinstance(self.schedule, RunSchedule):
            raise TypeError("LoggingConfig.schedule must be a RunSchedule")
        if not isinstance(self.console, bool):
            raise TypeError("console must be a boolean")
        if not self.filename:
            raise ValueError("log filename must not be empty")


@dataclass(frozen=True, slots=True)
class BackupConfig:
    """Automatic restart-backup policy.

    ``schedule=None`` disables periodic backups.  A relative ``path`` is
    resolved beneath the solver's solution directory.  ``write_at_end`` adds
    one final restart when the configured horizon is not itself scheduled.
    """

    schedule: RunSchedule | None = None
    path: str = "backup"
    write_at_end: bool = False

    def __post_init__(self) -> None:
        if self.schedule is not None and not isinstance(self.schedule, RunSchedule):
            raise TypeError("BackupConfig.schedule must be a RunSchedule or None")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("BackupConfig.path must be a non-empty string")
        if not isinstance(self.write_at_end, bool):
            raise TypeError("BackupConfig.write_at_end must be a boolean")


@dataclass
class FVMSetup:
    """Top-level setup for an incompressible finite-volume simulation."""

    case_name: str
    cores: int = 1

    mesh: MeshQualityConfig = field(default_factory=MeshQualityConfig)
    execution: ComputeConfig = field(default_factory=ComputeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    acceptance: RunAcceptanceLimits = field(default_factory=RunAcceptanceLimits)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    schemes: DiscretizationConfig = field(default_factory=DiscretizationConfig)
    linear: LinearSolverConfig = field(default_factory=LinearSolverConfig)
    pimple: PimpleControl = field(default_factory=PimpleControl)
    transport: TransportConfig = field(default_factory=TransportConfig)
    dynamic_mesh: MeshMotionConfig = field(default_factory=MeshMotionConfig.static)

    boundaries: list[BoundaryConfig] = field(default_factory=list)
    samplers: tuple = ()
    turbulence: TurbulenceConfig | None = None

    initial_velocity: list[float] | None = field(default_factory=lambda: [0.0, 0.0, 0.0])
    initial_kinematic_pressure: float | None = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.cores, bool) or not isinstance(self.cores, int):
            raise TypeError("cores must be an integer")
        if self.cores < 1:
            raise ValueError("cores must be at least one")
        nested_types = {
            "mesh": MeshQualityConfig,
            "execution": ComputeConfig,
            "output": OutputConfig,
            "acceptance": RunAcceptanceLimits,
            "logging": LoggingConfig,
            "backup": BackupConfig,
            "time": TimeConfig,
            "schemes": DiscretizationConfig,
            "linear": LinearSolverConfig,
            "pimple": PimpleControl,
            "transport": TransportConfig,
            "dynamic_mesh": MeshMotionConfig,
        }
        for name, expected in nested_types.items():
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"FVMSetup.{name} must be a {expected.__name__} instance")
        if not isinstance(self.case_name, str) or not self.case_name.strip():
            raise ValueError("FVMSetup.case_name must be a non-empty string")
        if not isinstance(self.boundaries, list | tuple):
            raise TypeError("FVMSetup.boundaries must be a list or tuple of BoundaryConfig objects")
        for index, boundary in enumerate(self.boundaries):
            if not isinstance(boundary, BoundaryConfig):
                raise TypeError(f"FVMSetup.boundaries[{index}] must be a BoundaryConfig")
        boundary_names = [boundary.name for boundary in self.boundaries]
        if len(boundary_names) != len(set(boundary_names)):
            raise ValueError("FVMSetup.boundaries must not contain duplicate patch names")
        if self.initial_velocity is not None:
            _validate_vector(
                "FVMSetup.initial_velocity", self.initial_velocity, allow_per_face=True
            )
        if self.initial_kinematic_pressure is not None:
            if isinstance(self.initial_kinematic_pressure, bool) or not isinstance(
                self.initial_kinematic_pressure, Real
            ):
                raise TypeError("FVMSetup.initial_kinematic_pressure must be a real scalar")
            if not math.isfinite(float(self.initial_kinematic_pressure)):
                raise ValueError("FVMSetup.initial_kinematic_pressure must be finite")
            self.initial_kinematic_pressure = float(self.initial_kinematic_pressure)
        if not isinstance(self.samplers, list | tuple):
            raise TypeError("FVMSetup.samplers must be a list or tuple")
        for index, sampler in enumerate(self.samplers):
            if not callable(getattr(sampler, "sample", None)):
                raise TypeError(f"FVMSetup.samplers[{index}] must implement sample()")
            if not isinstance(getattr(sampler, "schedule", None), RunSchedule):
                raise TypeError(f"FVMSetup.samplers[{index}] must provide a RunSchedule")
        self.samplers = tuple(self.samplers or ())

    def algorithm_params(self) -> dict[str, Any]:
        """Return the flat controls consumed by the algorithm layer."""
        merged: dict[str, Any] = {}
        for group in (
            self.schemes,
            self.linear,
            self.pimple,
        ):
            merged.update(vars(group))
        # ``n_nonorthogonal_correctors`` is the canonical public spelling;
        # retain the older key in the low-level dictionary for existing
        # algorithm callers during the migration.
        merged["n_nonorthogonal_correctors"] = self.pimple.n_nonorthogonal_correctors
        return merged

    def save(self, filepath: str) -> None:
        """Serialize this setup using canonical field names."""
        from source.solvers.fvm.sampling.base import (
            sampler_to_dict,
        )

        data = asdict(self)
        if self.samplers:
            data["samplers"] = [sampler_to_dict(sampler) for sampler in self.samplers]
        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(data, stream, indent=4)

    @classmethod
    def load(cls, filepath: str) -> FVMSetup:
        """Load a canonical FVM setup JSON file."""
        with open(filepath, encoding="utf-8") as stream:
            data = json.load(stream)

        data = dict(data)
        known_top_level = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known_top_level)
        if unknown:
            raise ValueError("Unknown top-level FVMSetup field(s): " + ", ".join(unknown))

        time_data = dict(data.get("time") or {})
        output_schedule_data = time_data.get("output_schedule")
        if output_schedule_data is not None:
            time_data["output_schedule"] = RunSchedule.from_dict(output_schedule_data)
        adjustment_data = time_data.get("adjustment")
        if adjustment_data is not None:
            if not isinstance(adjustment_data, dict):
                raise TypeError("Serialized TimeConfig.adjustment must be an object or null")
            time_data["adjustment"] = MaximumCourantTimeStep(**adjustment_data)
        linear_data = dict(data.get("linear") or {})
        pimple_data = dict(data.get("pimple") or {})
        transport_data = dict(data.get("transport") or {})
        logging_data = dict(data.get("logging") or {})
        logging_schedule_data = logging_data.get("schedule")
        if logging_schedule_data is not None:
            logging_data["schedule"] = RunSchedule.from_dict(logging_schedule_data)
        backup_data = dict(data.get("backup") or {})
        backup_schedule_data = backup_data.get("schedule")
        if backup_schedule_data is not None:
            backup_data["schedule"] = RunSchedule.from_dict(backup_schedule_data)

        boundaries = [BoundaryConfig(**boundary) for boundary in data.get("boundaries", [])]

        turbulence_data = data.get("turbulence")
        turbulence = None
        if turbulence_data:
            turbulence = TurbulenceConfig(**dict(turbulence_data))

        from source.solvers.fvm.sampling.base import (
            sampler_from_dict,
        )

        raw_samplers = data.get("samplers", [])
        if not isinstance(raw_samplers, list):
            raise TypeError("Serialized FVMSetup.samplers must be a list")
        samplers = []
        for index, item in enumerate(raw_samplers):
            if not isinstance(item, dict):
                raise TypeError(f"Serialized sampler at index {index} must be an object")
            samplers.append(sampler_from_dict(item))

        return cls(
            case_name=data["case_name"],
            cores=data.get("cores", 1),
            mesh=MeshQualityConfig(**data.get("mesh", {})),
            execution=ComputeConfig(**data.get("execution", {})),
            output=OutputConfig(**data.get("output", {})),
            acceptance=RunAcceptanceLimits(**data.get("acceptance", {})),
            logging=LoggingConfig(**logging_data),
            backup=BackupConfig(**backup_data),
            time=TimeConfig(**time_data),
            schemes=DiscretizationConfig(**data.get("schemes", {})),
            linear=LinearSolverConfig(**linear_data),
            pimple=PimpleControl(**pimple_data),
            samplers=tuple(samplers),
            transport=TransportConfig(**transport_data),
            dynamic_mesh=MeshMotionConfig(
                **data.get(
                    "dynamic_mesh",
                    {"method": "static"},
                )
            ),
            boundaries=boundaries,
            turbulence=turbulence,
            initial_velocity=data.get(
                "initial_velocity",
                [0.0, 0.0, 0.0],
            ),
            initial_kinematic_pressure=data.get(
                "initial_kinematic_pressure",
                0.0,
            ),
        )


def validate_fvm_setup(setup: FVMSetup) -> None:
    """Revalidate a setup at the solver admission boundary.

    The low-level configuration objects remain mutable for compatibility with
    existing callers.  A caller can therefore mutate a nested object after its
    dataclass constructor has run.  Re-running the small, side-effect-free
    validators here prevents such a mutation from reaching mesh allocation,
    backend initialization, or output creation.
    """
    if not isinstance(setup, FVMSetup):
        raise TypeError("setup must be an FVMSetup instance")
    nested_types = {
        "mesh": MeshQualityConfig,
        "execution": ComputeConfig,
        "output": OutputConfig,
        "acceptance": RunAcceptanceLimits,
        "logging": LoggingConfig,
        "backup": BackupConfig,
        "time": TimeConfig,
        "schemes": DiscretizationConfig,
        "linear": LinearSolverConfig,
        "pimple": PimpleControl,
        "transport": TransportConfig,
        "dynamic_mesh": MeshMotionConfig,
    }
    for name, expected in nested_types.items():
        value = getattr(setup, name)
        if not isinstance(value, expected):
            raise TypeError(f"FVMSetup.{name} must be a {expected.__name__} instance")
        value.__post_init__()
    if not isinstance(setup.boundaries, list | tuple):
        raise TypeError("FVMSetup.boundaries must be a list or tuple of BoundaryConfig objects")
    for index, boundary in enumerate(setup.boundaries):
        if not isinstance(boundary, BoundaryConfig):
            raise TypeError(f"FVMSetup.boundaries[{index}] must be a BoundaryConfig")
        boundary.__post_init__()
    setup.__post_init__()
    if setup.pimple.algorithm == "SIMPLE":
        if setup.schemes.time_scheme not in {"euler", "euler_implicit"}:
            raise ValueError(
                "SIMPLE is a steady algorithm; use an Euler-labelled spatial scheme "
                "or select PISO/PIMPLE for backward/BDF2 time integration"
            )
        if setup.time.adjustment is not None:
            raise ValueError("SIMPLE is steady and cannot use maximum-Courant time-step adjustment")
        if setup.time.output_schedule.every_time is not None:
            raise ValueError(
                "SIMPLE is steady; use an accepted-step output schedule instead of every_time"
            )
