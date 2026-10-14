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
        "minmod",
        "muscl",
        "superbee",
        "upwind",
        "van_leer",
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
        central + 0.25 linear upwind), or the TVD-limited ``limited_linear``,
        ``van_leer``, ``muscl``, ``minmod`` and ``superbee``.
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
        "van_leer",
        "muscl",
        "minmod",
        "superbee",
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
    linear_solver : {'bicgstab', 'gmres', 'cg', 'amg', 'spsolve'}
        Default method. Momentum cannot use ``amg`` directly.
    momentum_solver, pressure_solver : str or None
        Equation-specific methods. Momentum inherits ``linear_solver`` by
        default; pressure defaults to AMG. Explicit ``None`` inherits
        ``linear_solver`` for either equation.
    pressure_nullspace_method : {'auto', 'reference', 'petsc'}
        Constant-pressure nullspace treatment for singular incompressible
        systems. ``reference`` pins a datum; ``petsc`` attaches a nullspace.
    linear_failure_action : {'raise', 'direct_fallback'}
        Action when an iterative solve does not converge. The default
        ``raise`` never silently replaces the configured numerical method.
    momentum_preconditioner : {'dilu', 'jacobi', 'ilu'}
        Preconditioner of the SciPy Krylov momentum solve. ``dilu`` is the
        diagonal incomplete LU (``(D* + L) D*^-1 (D* + U)``, off-diagonal
        coefficients unchanged), rebuilt from the current matrix in one sweep;
        ``jacobi`` divides by the diagonal; ``ilu`` is SciPy's thresholded
        incomplete LU controlled by the ``ilu_*`` settings. The PETSc backend
        uses its own block-Jacobi preconditioner.
    reuse_ilu : bool
        Reuse an ILU preconditioner while matrix-change checks permit it.
    momentum_tolerance, pressure_tolerance : float
        Positive absolute residual tolerances in assembled equation units.
        The default momentum tolerance of ``1e-9`` converges the predictor
        well below the discretisation error; DILU keeps it inexpensive.
    momentum_relative_tolerance, pressure_relative_tolerance : float
        Non-negative residual reductions relative to each solve's initial norm.
    momentum_final_relative_tolerance, pressure_final_relative_tolerance : float or None
        Optional relative tolerances for final correctors; ``None`` retains
        the ordinary equation setting.
    momentum_max_iterations, pressure_max_iterations : int
        Positive iteration caps.
    amg_tolerance, amg_max_iterations : float or int or None
        Optional AMG-specific positive tolerance and iteration cap; ``None``
        uses the pressure tolerance and ``pressure_max_iterations``.
    amg_reuse_tolerance : float
        Positive dimensionless matrix-change tolerance for AMG reuse.
    ilu_drop_tolerance, ilu_fill_factor : float
        Positive SciPy ILU sparsification and fill controls.
    ilu_reuse_tolerance : float or None
        Optional non-negative dimensionless matrix-change check for ILU reuse.

    Momentum and pressure may select different methods; pressure also has an
    explicit constant-nullspace settings. Absolute tolerances are in assembled
    equation residual units and relative tolerances are normalized against the
    initial residual. ``linear_failure_action='raise'`` is the safe default;
    ``'direct_fallback'`` explicitly permits a fallback solve.

    Raises
    ------
    TypeError
        If Boolean/integer settings fields have invalid types.
    ValueError
        If a method is unsupported or a tolerance/count violates its range.
    """

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
    ) = "amg"
    pressure_nullspace_method: Literal[
        "auto",
        "reference",
        "petsc",
    ] = "auto"
    linear_failure_action: Literal[
        "raise",
        "direct_fallback",
    ] = "raise"
    momentum_preconditioner: Literal[
        "dilu",
        "jacobi",
        "ilu",
    ] = "dilu"
    reuse_ilu: bool = True

    momentum_tolerance: float = 1e-9
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
            choice = value
            if choice not in valid:
                raise ValueError(f"Unsupported LinearSolverConfig.{name}={value!r}")
            object.__setattr__(self, name, choice)
        if self.momentum_solver == "amg" or (
            self.momentum_solver is None and self.linear_solver == "amg"
        ):
            raise ValueError("LinearSolverConfig effective momentum solver cannot be 'amg'")
        nullspace = self.pressure_nullspace_method
        if nullspace not in {"auto", "reference", "petsc"}:
            raise ValueError(
                "LinearSolverConfig.pressure_nullspace_method must be auto, reference, or petsc"
            )
        object.__setattr__(self, "pressure_nullspace_method", nullspace)
        failure_action = self.linear_failure_action
        if failure_action not in {"raise", "direct_fallback"}:
            raise ValueError(
                "LinearSolverConfig.linear_failure_action must be raise or direct_fallback"
            )
        object.__setattr__(self, "linear_failure_action", failure_action)
        preconditioner = self.momentum_preconditioner
        if preconditioner not in {"dilu", "jacobi", "ilu"}:
            raise ValueError(
                "LinearSolverConfig.momentum_preconditioner must be dilu, jacobi, or ilu"
            )
        object.__setattr__(self, "momentum_preconditioner", preconditioner)
        if not isinstance(self.reuse_ilu, bool):
            raise TypeError("LinearSolverConfig.reuse_ilu must be a boolean")
        for name in (
            "momentum_tolerance",
            "pressure_tolerance",
            "ilu_drop_tolerance",
            "ilu_fill_factor",
            "amg_reuse_tolerance",
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
        for name in (
            "momentum_final_relative_tolerance",
            "pressure_final_relative_tolerance",
            "ilu_reuse_tolerance",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    finite_real(f"LinearSolverConfig.{name}", value, minimum=0.0),
                )
        if self.amg_tolerance is not None:
            object.__setattr__(
                self,
                "amg_tolerance",
                finite_real(
                    "LinearSolverConfig.amg_tolerance",
                    self.amg_tolerance,
                    minimum=0.0,
                    strict_minimum=True,
                ),
            )
        for name in ("momentum_max_iterations", "pressure_max_iterations"):
            object.__setattr__(
                self,
                name,
                strict_int(f"LinearSolverConfig.{name}", getattr(self, name), minimum=1),
            )
        if self.amg_max_iterations is not None:
            object.__setattr__(
                self,
                "amg_max_iterations",
                strict_int(
                    "LinearSolverConfig.amg_max_iterations", self.amg_max_iterations, minimum=1
                ),
            )


@dataclass(frozen=True, slots=True)
class PimpleConfig:
    """Pressure--velocity coupling with SIMPLE, PISO or PIMPLE.

    Parameters
    ----------
    algorithm : {'pimple', 'piso', 'simple'}, default='pimple'
        Pressure--velocity coupling algorithm. ``simple`` is steady.
    n_correctors : int, default=2
        Pressure-correction solves per outer pass.
    n_outer_correctors : int, default=1
        Nonlinear outer passes; PISO uses one.
    n_nonorthogonal_correctors : int, default=0
        Extra pressure solves for the non-orthogonal correction.
    min_outer_correctors : int, default=1
        Outer passes before residual-based early termination.
    outer_residual_tolerance : float or None
        Optional maximum equation residual for early termination.
    outer_continuity_tolerance : float or None
        Optional maximum cell divergence in 1/s for early termination.
    max_iterations : int, default=20
        SIMPLE iteration limit.
    tolerance : float, default=1e-6
        SIMPLE convergence tolerance.
    velocity_relaxation, pressure_relaxation : float, default=1.0
        Under-relaxation factors in ``(0, 1]``.
    ddt_corr : bool, default=True
        Include the transient Rhie--Chow flux correction.
    """

    algorithm: Literal["pimple", "piso", "simple"] = "pimple"
    n_correctors: int = 2
    n_outer_correctors: int = 1
    n_nonorthogonal_correctors: int = 0
    min_outer_correctors: int = 1
    outer_residual_tolerance: float | None = None
    outer_continuity_tolerance: float | None = None
    max_iterations: int = 20
    tolerance: float = 1e-6
    velocity_relaxation: float = 1.0
    pressure_relaxation: float = 1.0
    ddt_corr: bool = True

    def __post_init__(self) -> None:
        lowercase_choice("PimpleConfig.algorithm", self.algorithm, {"pimple", "piso", "simple"})
        object.__setattr__(
            self,
            "n_nonorthogonal_correctors",
            strict_int(
                "PimpleConfig.n_nonorthogonal_correctors",
                self.n_nonorthogonal_correctors,
                minimum=0,
            ),
        )
        for name in (
            "n_correctors",
            "n_outer_correctors",
            "min_outer_correctors",
            "max_iterations",
        ):
            object.__setattr__(
                self, name, strict_int(f"PimpleConfig.{name}", getattr(self, name), minimum=1)
            )
        if self.min_outer_correctors > self.n_outer_correctors:
            raise ValueError("PimpleConfig.min_outer_correctors cannot exceed n_outer_correctors")
        if self.algorithm == "piso" and self.n_outer_correctors != 1:
            raise ValueError("PimpleConfig algorithm 'piso' requires n_outer_correctors == 1")
        object.__setattr__(
            self,
            "tolerance",
            finite_real("PimpleConfig.tolerance", self.tolerance, minimum=0.0, strict_minimum=True),
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
        for name in ("outer_residual_tolerance", "outer_continuity_tolerance"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    finite_real(f"PimpleConfig.{name}", value, minimum=0.0, strict_minimum=True),
                )
        if not isinstance(self.ddt_corr, bool):
            raise TypeError("PimpleConfig.ddt_corr must be a boolean")


@dataclass(frozen=True, slots=True)
class ComputeConfig:
    """Execution backends of the operators, linear algebra, MPI and output.

    Parameters
    ----------
    operator_backend : {'numpy', 'numba', 'taichi'}, default='numpy'
        Implementation of the local discrete operators.
    linear_backend : {'scipy', 'petsc'}, default='scipy'
        Sparse linear-algebra library. Serial runs use SciPy.
    parallel_mode : {'serial', 'petsc_replicated', 'petsc_partitioned'}
        Mesh distribution: every rank holds the global mesh (replicated) or
        its owned cells plus a halo layer (partitioned).
    output_mode : {'synchronous', 'threaded'}, default='synchronous'
        Write visualization frames inline or on a background thread.
    """

    operator_backend: Literal["numpy", "numba", "taichi"] = "numpy"
    linear_backend: Literal["scipy", "petsc"] = "scipy"
    parallel_mode: Literal["serial", "petsc_replicated", "petsc_partitioned"] = "serial"
    output_mode: Literal["synchronous", "threaded"] = "synchronous"

    def __post_init__(self) -> None:
        choices = {
            "operator_backend": {"numpy", "numba", "taichi"},
            "linear_backend": {"scipy", "petsc"},
            "parallel_mode": {"serial", "petsc_replicated", "petsc_partitioned"},
            "output_mode": {"synchronous", "threaded"},
        }
        for name, valid in choices.items():
            lowercase_choice(f"ComputeConfig.{name}", getattr(self, name), valid)
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
        if self.pimple.algorithm == "simple":
            if self.schemes.time_scheme != "euler_implicit":
                raise ValueError(
                    "The steady 'simple' algorithm requires time_scheme='euler_implicit'"
                )
            if self.time_step_control is not None:
                raise ValueError("The steady 'simple' algorithm has no adaptive time step")
