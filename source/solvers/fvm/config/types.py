"""Configuration models for the incompressible FVM solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal

from source.write_precision import (
    DEFAULT_WRITE_PRECISION,
    WritePrecision,
    validate_write_precision,
)


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


@dataclass
class TimeConfig:
    """Time integration and output cadence."""

    time_step_size: float = 0.01
    start_time: float = 0.0
    end_time: float = 1.0
    output_interval_steps: int = 10
    output_interval_time: float | None = None
    adjust_time_step: bool = False
    max_courant_number: float = 1.0
    max_time_step_size: float = 0.1
    min_time_step_size: float = 1e-4
    time_step_size_adjust_coeff: float = 1.2

    @staticmethod
    def steady(
        max_iterations: int = 1000,
        output_interval_steps: int = 100,
    ) -> TimeConfig:
        """Return a steady SIMPLE time configuration."""
        return TimeConfig(
            time_step_size=1.0,
            start_time=0.0,
            end_time=float(max_iterations),
            output_interval_steps=output_interval_steps,
        )

    @staticmethod
    def transient(
        time_step_size: float,
        duration: float,
        output_interval_steps: int = 10,
    ) -> TimeConfig:
        """Return a transient time configuration."""
        return TimeConfig(
            time_step_size=time_step_size,
            start_time=0.0,
            end_time=duration,
            output_interval_steps=output_interval_steps,
        )


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
    pressure_nullspace_policy: Literal[
        "auto",
        "reference",
        "petsc",
    ] = "auto"
    linear_failure_policy: Literal[
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


@dataclass
class PimpleControl:
    """PIMPLE, PISO, or SIMPLE pressure-velocity coupling controls."""

    algorithm: Literal["SIMPLE", "PIMPLE", "PISO"] = "PIMPLE"
    n_correctors: int = 2
    n_outer_correctors: int = 1
    n_orthogonal_correctors: int = 0
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


@dataclass
class TransportConfig:
    """Fluid density and molecular kinematic viscosity."""

    density: float = 1.225
    kinematic_viscosity: float = 1.5e-5

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
        if self.ghost_layers not in {0, 1}:
            raise ValueError("ghost_layers must be zero or one")
        if self.point_interpolation not in {
            "none",
            "boundary_weighted",
        }:
            raise ValueError("point_interpolation must be 'none' or 'boundary_weighted'")


@dataclass
class RunAcceptancePolicy:
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


@dataclass
class LoggingConfig:
    """Console and log-file verbosity."""

    mode: Literal["simple", "debug"] = "simple"
    interval_steps: int = 1
    console: bool = True
    filename: str = "fvm.log"

    def __post_init__(self) -> None:
        if self.mode not in {"simple", "debug"}:
            raise ValueError("log mode must be 'simple' or 'debug'")
        if isinstance(self.interval_steps, bool) or not isinstance(self.interval_steps, int):
            raise TypeError("logging interval must be an integer")
        if self.interval_steps < 1:
            raise ValueError("logging interval must be at least one")
        if not isinstance(self.console, bool):
            raise TypeError("console must be a boolean")
        if not self.filename:
            raise ValueError("log filename must not be empty")


@dataclass
class FVMSetup:
    """Top-level setup for an incompressible finite-volume simulation."""

    case_name: str
    cores: int = 1

    mesh: MeshQualityConfig = field(default_factory=MeshQualityConfig)
    execution: ComputeConfig = field(default_factory=ComputeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    acceptance: RunAcceptancePolicy = field(default_factory=RunAcceptancePolicy)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
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
        linear_data = dict(data.get("linear") or {})
        pimple_data = dict(data.get("pimple") or {})
        transport_data = dict(data.get("transport") or {})
        logging_data = dict(data.get("logging") or {})

        boundaries = [BoundaryConfig(**boundary) for boundary in data.get("boundaries", [])]

        turbulence_data = data.get("turbulence")
        turbulence = None
        if turbulence_data:
            turbulence = TurbulenceConfig(**dict(turbulence_data))

        from source.solvers.fvm.sampling.base import (
            sampler_from_dict,
        )

        samplers = tuple(
            sampler_from_dict(item) for item in data.get("samplers", []) if isinstance(item, dict)
        )

        return cls(
            case_name=data["case_name"],
            cores=data.get("cores", 1),
            mesh=MeshQualityConfig(**data.get("mesh", {})),
            execution=ComputeConfig(**data.get("execution", {})),
            output=OutputConfig(**data.get("output", {})),
            acceptance=RunAcceptancePolicy(**data.get("acceptance", {})),
            logging=LoggingConfig(**logging_data),
            time=TimeConfig(**time_data),
            schemes=DiscretizationConfig(**data.get("schemes", {})),
            linear=LinearSolverConfig(**linear_data),
            pimple=PimpleControl(**pimple_data),
            samplers=samplers,
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
