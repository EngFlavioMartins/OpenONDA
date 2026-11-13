"""Numerical settings of an FVM case: every choice that changes the solution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ._validation import finite_real, lowercase_choice, strict_int
from .turbulence import TurbulenceConfig

CONVECTION_SCHEMES = frozenset(
    {
        "central",
        "limited_linear",
        "linear_upwind",
        "lust",
        "upwind",
    }
)


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """Constant fluid properties of the incompressible equations.

    Parameters
    ----------
    density : float, default=1.225
        Positive density in kg/m³. It dimensionalizes kinematic pressure and
        surface forces and cancels from the velocity equations.
    kinematic_viscosity : float, default=1.5e-5
        Positive molecular kinematic viscosity ``nu`` in m²/s.
    """

    density: float = 1.225
    kinematic_viscosity: float = 1.5e-5

    def __post_init__(self) -> None:
        for name in ("density", "kinematic_viscosity"):
            object.__setattr__(
                self,
                name,
                finite_real(
                    f"TransportConfig.{name}", getattr(self, name), minimum=0.0, strict_minimum=True
                ),
            )


@dataclass(frozen=True, slots=True)
class DiscretizationConfig:
    """Spatial reconstruction and time discretization.

    Parameters
    ----------
    convection_scheme : str, default='limited_linear'
        Face interpolation of the convected velocity: ``upwind`` (first
        order, bounded), ``central``, ``linear_upwind``, ``lust`` (0.75
        central + 0.25 linear upwind), or the TVD-limited ``limited_linear``.
    gradient_scheme : {'gauss', 'lsq'}, default='lsq'
        Cell-centred gradient: Gauss divergence theorem or least squares.
    time_scheme : {'euler_implicit', 'backward'}, default='euler_implicit'
        Time derivative. ``backward`` is the second-order BDF2 formula once
        two accepted time levels exist.
    """

    convection_scheme: Literal[
        "upwind",
        "central",
        "linear_upwind",
        "lust",
        "limited_linear",
    ] = "limited_linear"
    gradient_scheme: Literal["gauss", "lsq"] = "lsq"
    time_scheme: Literal["euler_implicit", "backward"] = "euler_implicit"

    def __post_init__(self) -> None:
        lowercase_choice(
            "DiscretizationConfig.convection_scheme", self.convection_scheme, CONVECTION_SCHEMES
        )
        lowercase_choice(
            "DiscretizationConfig.gradient_scheme", self.gradient_scheme, {"gauss", "lsq"}
        )
        lowercase_choice(
            "DiscretizationConfig.time_scheme", self.time_scheme, {"euler_implicit", "backward"}
        )


@dataclass(frozen=True, slots=True)
class LinearSolverConfig:
    """Momentum/pressure linear-algebra settings for one FVM case.

    Parameters
    ----------
    linear_solver : {'bicgstab', 'amg', 'spsolve'}
        Default method. Momentum cannot use ``amg`` directly.
    pressure_solver : str or None
        Pressure method; the default is AMG and ``None`` inherits
        ``linear_solver``. Momentum always uses ``linear_solver``.
    pressure_nullspace_method : {'auto', 'reference', 'petsc'}
        Constant-pressure nullspace treatment for singular incompressible
        systems. ``reference`` pins a datum; ``petsc`` attaches a nullspace.
    momentum_tolerance, pressure_tolerance : float
        Positive absolute residual tolerances in assembled equation units.
        The default momentum tolerance of ``1e-9`` converges the predictor
        well below the discretisation error; DILU keeps it inexpensive.
    momentum_relative_tolerance, pressure_relative_tolerance : float
        Non-negative residual reductions relative to each solve's initial norm;
        the final corrector of a time step solves to the absolute tolerance only.
    momentum_max_iterations, pressure_max_iterations : int
        Positive iteration caps.

    Momentum and pressure may select different methods; pressure also has an
    explicit constant-nullspace settings. Absolute tolerances are in assembled
    equation residual units and relative tolerances are normalized against the
    initial residual. A solve that does not converge raises. The momentum
    solve is BiCGSTAB with a diagonal incomplete LU preconditioner, rebuilt
    from the current matrix in one sweep (``(D* + L) D*^-1 (D* + U)``); the
    PETSc backend uses its own block-Jacobi preconditioner.

    Raises
    ------
    TypeError
        If Boolean/integer settings fields have invalid types.
    ValueError
        If a method is unsupported or a tolerance/count violates its range.
    """

    linear_solver: Literal["bicgstab", "amg", "spsolve"] = "bicgstab"
    pressure_solver: Literal["amg", "bicgstab", "spsolve"] | None = "amg"
    pressure_nullspace_method: Literal[
        "auto",
        "reference",
        "petsc",
    ] = "auto"

    momentum_tolerance: float = 1e-9
    momentum_relative_tolerance: float = 0.0
    momentum_max_iterations: int = 1000

    pressure_tolerance: float = 1e-8
    pressure_relative_tolerance: float = 0.0
    pressure_max_iterations: int = 500

    def __post_init__(self) -> None:
        valid = {"bicgstab", "amg", "spsolve"}
        for name in ("linear_solver", "pressure_solver"):
            value = getattr(self, name)
            if value is None and name != "linear_solver":
                continue
            if value not in valid:
                raise ValueError(f"Unsupported LinearSolverConfig.{name}={value!r}")
        if self.linear_solver == "amg":
            raise ValueError("LinearSolverConfig.linear_solver drives momentum and cannot be 'amg'")
        nullspace = self.pressure_nullspace_method
        if nullspace not in {"auto", "reference", "petsc"}:
            raise ValueError(
                "LinearSolverConfig.pressure_nullspace_method must be auto, reference, or petsc"
            )
        object.__setattr__(self, "pressure_nullspace_method", nullspace)
        for name in (
            "momentum_tolerance",
            "pressure_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                finite_real(
                    f"LinearSolverConfig.{name}",
                    getattr(self, name),
                    minimum=0.0,
                    strict_minimum=True,
                ),
            )
        for name in ("momentum_relative_tolerance", "pressure_relative_tolerance"):
            object.__setattr__(
                self,
                name,
                finite_real(f"LinearSolverConfig.{name}", getattr(self, name), minimum=0.0),
            )
        for name in ("momentum_max_iterations", "pressure_max_iterations"):
            object.__setattr__(
                self,
                name,
                strict_int(f"LinearSolverConfig.{name}", getattr(self, name), minimum=1),
            )


@dataclass(frozen=True, slots=True)
class PimpleConfig:
    """Transient pressure--velocity coupling with PIMPLE.

    Parameters
    ----------
    n_correctors : int, default=2
        Pressure-correction solves per outer pass.
    n_outer_correctors : int, default=1
        Nonlinear outer passes; PISO uses one.
    n_nonorthogonal_correctors : int, default=0
        Extra pressure solves for the non-orthogonal correction.
    velocity_relaxation, pressure_relaxation : float, default=1.0
        Under-relaxation factors in ``(0, 1]``.
    """

    n_correctors: int = 2
    n_outer_correctors: int = 1
    n_nonorthogonal_correctors: int = 0
    velocity_relaxation: float = 1.0
    pressure_relaxation: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_nonorthogonal_correctors",
            strict_int(
                "PimpleConfig.n_nonorthogonal_correctors",
                self.n_nonorthogonal_correctors,
                minimum=0,
            ),
        )
        for name in ("n_correctors", "n_outer_correctors"):
            object.__setattr__(
                self, name, strict_int(f"PimpleConfig.{name}", getattr(self, name), minimum=1)
            )
        for name in ("velocity_relaxation", "pressure_relaxation"):
            object.__setattr__(
                self,
                name,
                finite_real(
                    f"PimpleConfig.{name}",
                    getattr(self, name),
                    minimum=0.0,
                    maximum=1.0,
                    strict_minimum=True,
                ),
            )


@dataclass(frozen=True, slots=True)
class ComputeConfig:
    """Execution backends of the operators, linear algebra, MPI and output.

    Parameters
    ----------
    operator_backend : {'numba'}, default='numba'
        Implementation of the local discrete operators. Numba is the only
        implementation; the field remains because coupled cases set it.
    linear_backend : {'scipy', 'petsc'}, default='scipy'
        Sparse linear-algebra library. Serial runs use SciPy.
    parallel_mode : {'serial', 'petsc_replicated', 'petsc_partitioned'}
        Mesh distribution: every rank holds the global mesh (replicated) or
        its owned cells plus a halo layer (partitioned).
    """

    operator_backend: Literal["numba"] = "numba"
    linear_backend: Literal["scipy", "petsc"] = "scipy"
    parallel_mode: Literal["serial", "petsc_replicated", "petsc_partitioned"] = "serial"

    def __post_init__(self) -> None:
        choices = {
            "operator_backend": {"numba"},
            "linear_backend": {"scipy", "petsc"},
            "parallel_mode": {"serial", "petsc_replicated", "petsc_partitioned"},
        }
        for name, valid in choices.items():
            lowercase_choice(f"ComputeConfig.{name}", getattr(self, name), valid)
        if self.parallel_mode == "serial" and self.linear_backend != "scipy":
            raise ValueError("ComputeConfig serial mode requires linear_backend='scipy'")
        if self.parallel_mode != "serial" and self.linear_backend != "petsc":
            raise ValueError(
                f"ComputeConfig.parallel_mode={self.parallel_mode!r} requires linear_backend='petsc'"
            )


@dataclass(frozen=True, slots=True)
class MaximumCourantTimeStep:
    """Adaptive time-step size from the maximum cell Courant number.

    Before every transient step the solver measures the Courant number of the
    current face fluxes and selects the next step with OpenFOAM's damped
    adjustment: reductions act immediately, increases are limited to twenty
    percent per accepted step. Physical-time output events shorten the step
    so they are reached exactly.

    Parameters
    ----------
    maximum : float, default=0.9
        Target maximum cell Courant number.
    maximum_time_step_size : float or None
        Optional upper bound of the time-step size in s.
    """

    maximum: float = 0.9
    maximum_time_step_size: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum",
            finite_real(
                "MaximumCourantTimeStep.maximum", self.maximum, minimum=0.0, strict_minimum=True
            ),
        )
        if self.maximum_time_step_size is not None:
            object.__setattr__(
                self,
                "maximum_time_step_size",
                finite_real(
                    "MaximumCourantTimeStep.maximum_time_step_size",
                    self.maximum_time_step_size,
                    minimum=0.0,
                    strict_minimum=True,
                ),
            )


@dataclass(frozen=True, slots=True)
class RunAcceptanceLimits:
    """Warning and rejection thresholds of candidate FVM steps.

    Parameters
    ----------
    sustained_steps : int, default=1
        Consecutive exceeding steps before a step is rejected.
    max_continuity_error_warning, max_continuity_error_abort : float or None
        Maximum cell divergence in 1/s.
    max_equation_residual_warning, max_equation_residual_abort : float or None
        Maximum momentum or pressure equation residual.
    max_courant_number_warning, max_courant_number_abort : float or None
        Maximum cell Courant number.
    max_velocity_magnitude_warning, max_velocity_magnitude_abort : float or None
        Maximum cell velocity magnitude in m/s.

    ``None`` disables a threshold. A warning is logged; an abort rejects the
    candidate step with :class:`~openonda.fvm.AcceptanceLimitError`.
    """

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
        object.__setattr__(
            self,
            "sustained_steps",
            strict_int("RunAcceptanceLimits.sustained_steps", self.sustained_steps, minimum=1),
        )
        for metric in (
            "max_continuity_error",
            "max_equation_residual",
            "max_courant_number",
            "max_velocity_magnitude",
        ):
            values = {}
            for level in ("warning", "abort"):
                name = f"{metric}_{level}"
                value = getattr(self, name)
                if value is not None:
                    value = finite_real(
                        f"RunAcceptanceLimits.{name}", value, minimum=0.0, strict_minimum=True
                    )
                values[level] = value
                object.__setattr__(self, name, value)
            if (
                values["warning"] is not None
                and values["abort"] is not None
                and values["warning"] > values["abort"]
            ):
                raise ValueError(
                    f"RunAcceptanceLimits.{metric}_warning cannot exceed {metric}_abort"
                )


@dataclass(frozen=True, slots=True)
class Numerics:
    """Every FVM setting that changes the computed solution.

    Parameters
    ----------
    time_step_size : float, default=0.01
        Fixed, or initial adaptive, time-step size in s.
    time_step_control : MaximumCourantTimeStep or None
        Adaptive time-step size; ``None`` keeps ``time_step_size`` fixed.
    transport : TransportConfig
        Density and molecular kinematic viscosity.
    schemes : DiscretizationConfig
        Convection, gradient and time discretization.
    linear : LinearSolverConfig
        Momentum and pressure linear solvers.
    pimple : PimpleConfig
        Pressure--velocity coupling.
    execution : ComputeConfig
        Operator, linear-algebra, MPI and output backends.
    turbulence : TurbulenceConfig
        Subgrid-scale closure; :meth:`TurbulenceConfig.dns` by default.
    acceptance : RunAcceptanceLimits
        Warning and rejection thresholds of candidate steps.
    """

    time_step_size: float = 0.01
    time_step_control: MaximumCourantTimeStep | None = None
    transport: TransportConfig = field(default_factory=TransportConfig)
    schemes: DiscretizationConfig = field(default_factory=DiscretizationConfig)
    linear: LinearSolverConfig = field(default_factory=LinearSolverConfig)
    pimple: PimpleConfig = field(default_factory=PimpleConfig)
    execution: ComputeConfig = field(default_factory=ComputeConfig)
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig.dns)
    acceptance: RunAcceptanceLimits = field(default_factory=RunAcceptanceLimits)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "time_step_size",
            finite_real(
                "Numerics.time_step_size", self.time_step_size, minimum=0.0, strict_minimum=True
            ),
        )
        if self.time_step_control is not None and not isinstance(
            self.time_step_control, MaximumCourantTimeStep
        ):
            raise TypeError("Numerics.time_step_control must be a MaximumCourantTimeStep or None")
        expected = {
            "transport": TransportConfig,
            "schemes": DiscretizationConfig,
            "linear": LinearSolverConfig,
            "pimple": PimpleConfig,
            "execution": ComputeConfig,
            "turbulence": TurbulenceConfig,
            "acceptance": RunAcceptanceLimits,
        }
        for name, cls in expected.items():
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"Numerics.{name} must be a {cls.__name__} instance")
