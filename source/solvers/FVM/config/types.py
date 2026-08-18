from dataclasses import asdict, dataclass, field
import json
from typing import Literal


@dataclass
class BoundaryConfig:
    """Boundary-condition specification for one mesh patch.

    Wraps the type and value for every field (velocity, pressure, scalar,
    turbulent viscosity) applied to a single patch.

    Use the factory methods (:meth:`inlet`, :meth:`outlet`, :meth:`wall`,
    etc.) for common boundary types; they set sensible defaults for all
    fields at once.

    Examples
    --------
    >>> BoundaryConfig.inlet("inlet", velocity=[1.0, 0.0, 0.0])
    >>> BoundaryConfig.wall("cube")
    """

    name: str
    """Patch name, matched to the mesh patch surface."""
    type_velocity: str = "fixedValue"
    """Velocity boundary-condition type (OpenFOAM-style)."""
    value_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Velocity value ``[u, v, w]`` applied by fixed-value velocity BCs."""
    type_p: str = "zeroGradient"
    """Pressure boundary-condition type (OpenFOAM-style)."""
    value_p: float = 0.0
    """Kinematic pressure ``p/ρ`` [m²/s²] applied by fixed-value pressure BCs."""
    type_phi: str = "zeroGradient"
    """Face-flux boundary-condition type (OpenFOAM-style)."""
    value_phi: float = 0.0
    """Face-flux value applied by fixed-value flux BCs."""
    type_nut: str = "calculated"
    """Turbulent-viscosity boundary-condition type (OpenFOAM-style)."""
    value_nut: float = 0.0
    """Turbulent-viscosity value applied by fixed-value BCs."""
    neighbour_patch: str | None = None
    """Name of the paired patch for cyclic boundaries."""
    mesh_type: Literal["patch", "wall", "empty", "cyclic"] | None = None
    """Mesh-topology hint: patch, wall, empty, or cyclic."""

    @staticmethod
    def inlet(name: str, velocity: list[float]) -> "BoundaryConfig":
        """Create a Dirichlet velocity / zero-gradient pressure inlet.

        Args:
            name:     Patch name (e.g. ``"inlet"``).
            velocity: Prescribed velocity vector ``[u, v, w]``.

        Returns:
            A new :class:`BoundaryConfig` suitable for inflow boundaries.
        """
        return BoundaryConfig(
            name=name, type_velocity="fixedValue", value_velocity=velocity, type_p="zeroGradient"
        )

    @staticmethod
    def outlet(name: str, p: float = 0.0) -> "BoundaryConfig":
        """Create an outlet with fixed pressure and ``inletOutlet`` velocity.

        Uses ``inletOutlet`` for velocity: zero-gradient on outflow, zero
        inflow on reverse flow.  This is bounded and does not rely on
        patch-name heuristics.

        Args:
            name: Patch name (e.g. ``"outlet"``).
            p: Prescribed kinematic pressure ``p/ρ`` [m²/s²]. Defaults
                to 0.0. Do not supply pressure in pascals.

        Returns:
            A new :class:`BoundaryConfig` suitable for outflow boundaries.
        """
        return BoundaryConfig(
            name=name, type_velocity="inletOutlet", type_p="fixedValue", value_p=p
        )

    @staticmethod
    def freestream(name: str, velocity: list[float], p: float = 0.0) -> "BoundaryConfig":
        """Create an incompressible far-field boundary.

        Velocity is prescribed on inflow and extrapolated on outflow. Pressure
        is extrapolated on inflow and fixed to *p* on outflow.

        Args:
            name:     Patch name.
            velocity: Freestream velocity vector ``[u, v, w]``.

        Returns:
            A new :class:`BoundaryConfig` for external-flow farfield boundaries.
        """
        return BoundaryConfig(
            name=name,
            type_velocity="freestream",
            value_velocity=velocity,
            type_p="freestream",
            value_p=p,
        )

    @staticmethod
    def cyclic(name: str, neighbour_patch: str) -> "BoundaryConfig":
        """Create one side of a translational periodic patch pair."""
        return BoundaryConfig(
            name=name,
            type_velocity="cyclic",
            type_p="cyclic",
            neighbour_patch=neighbour_patch,
            mesh_type="cyclic",
        )

    @staticmethod
    def wall(name: str) -> "BoundaryConfig":
        """Create a no-slip wall boundary condition.

        Sets velocity to zero (fixed value) and pressure to zero-gradient.
        Turbulent viscosity is computed by the selected native model.

        Args:
            name: Patch name (e.g. ``"bottomWall"``).

        Returns:
            A new :class:`BoundaryConfig` for solid walls.
        """
        return BoundaryConfig(
            name=name,
            type_velocity="fixedValue",
            value_velocity=[0.0, 0.0, 0.0],
            type_p="zeroGradient",
            type_nut="calculated",
            mesh_type="wall",
        )

    @staticmethod
    def slip(name: str) -> "BoundaryConfig":
        """Create an impermeable, zero-shear boundary."""
        return BoundaryConfig(
            name=name,
            type_velocity="slip",
            type_p="zeroGradient",
            type_nut="zeroGradient",
            mesh_type="patch",
        )

    @staticmethod
    def empty(name: str) -> "BoundaryConfig":
        """Empty boundary condition for 2D simulations (frontAndBack patches).

        Sets velocity to 'empty' (zero normal flux, no tangential constraint)
        and pressure to 'empty' (no pressure gradient in that direction).
        This is the correct BC for the out-of-plane faces when running a
        quasi-2D extruded mesh.
        """
        return BoundaryConfig(
            name=name,
            type_velocity="empty",
            value_velocity=[0.0, 0.0, 0.0],
            type_p="empty",
            type_nut="zeroGradient",
            mesh_type="empty",
        )


@dataclass
class MeshConfig:
    """Quality limits applied to a solver-native or Gmsh mesh.

    Meshes are supplied directly to :func:`setup_fvm_solver` as an in-memory
    mesh, a callable that builds one, or a Gmsh ``.msh`` path.  Set any limit
    below to reject a mesh that does not satisfy it.
    """

    max_non_orthogonality_deg: float | None = None
    """Reject meshes with face non-orthogonality above this angle (degrees); None disables."""
    max_skewness: float | None = None
    """Reject meshes with cell skewness above this value; None disables."""
    max_aspect_ratio: float | None = None
    """Reject meshes with cell aspect ratio above this value; None disables."""
    max_lsq_condition: float | None = None
    """Reject meshes whose least-squares gradient stencil condition exceeds this; None disables."""


@dataclass
class TimeConfig:
    """Time integration and output cadence.

    Times are in seconds. ``write_interval`` counts solver steps;
    ``write_interval_time`` is a physical-time interval. Set
    ``adjust_timestep=True`` to enforce ``max_cfl``.

    Examples
    --------
    >>> TimeConfig.transient(dt=0.01, duration=10.0)
    >>> TimeConfig.steady(max_iter=500)
    """

    delta_t: float = 0.01
    """Time-step size (seconds)."""
    start_time: float = 0.0
    """Simulation start time (seconds)."""
    end_time: float = 1.0
    """Simulation end time (seconds); ``end_time - start_time`` gives the step count."""
    write_interval: int = 10
    """Write output every *N*-th solver step."""
    write_interval_time: float | None = None
    """Physical-time output interval (seconds); overrides ``write_interval`` when set."""
    adjust_timestep: bool = False
    """Dynamically adapt ``delta_t`` to respect ``max_cfl``."""
    max_cfl: float = 1.0
    """Target maximum CFL number when ``adjust_timestep`` is enabled."""
    max_delta_t: float = 0.1
    """Upper bound on the adaptive time step (seconds)."""
    min_delta_t: float = 1e-4
    """Lower bound on the adaptive time step (seconds)."""
    dt_adjust_coeff: float = 1.2
    """Multiplicative factor used when adapting the time step."""

    @staticmethod
    def steady(max_iter: int = 1000, write_interval: int = 100) -> "TimeConfig":
        """Create a steady-state time configuration.

        Sets ``delta_t=1``, so ``n_steps = end_time - start_time = max_iter``,
        which is treated by the SIMPLE algorithm as outer iterations.

        Args:
            max_iter:       Equivalent number of SIMPLE iterations.
            write_interval: Save frequency in iterations.

        Returns:
            :class:`TimeConfig` suitable for steady SIMPLE.
        """
        return TimeConfig(delta_t=1, start_time=0, end_time=max_iter, write_interval=write_interval)

    @staticmethod
    def transient(dt: float, duration: float, write_interval: int = 10) -> "TimeConfig":
        """Create a transient time configuration.

        Args:
            dt:             Time-step size (seconds).
            duration:       Total simulation time (seconds).
            write_interval: Save every *write_interval*-th step.

        Returns:
            :class:`TimeConfig` suitable for PIMPLE / PISO.
        """
        return TimeConfig(
            delta_t=dt, start_time=0, end_time=duration, write_interval=write_interval
        )


@dataclass
class SchemesConfig:
    """Spatial and temporal discretisation settings.

    ``LUST`` is the production LES/DNS choice. ``limitedLinear`` or
    ``upwind`` is more dissipative and useful for difficult coarse meshes.

    Examples
    --------
    >>> # Low-dissipation setup for LES
    >>> SchemesConfig(convection_scheme="LUST", gradient_scheme="lsq")
    """

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
    """Convection discretization: upwind, central, limitedLinear, LUST, linearUpwind,
    vanLeer, MUSCL, minmod, or superbee."""
    gradient_scheme: Literal["gauss", "lsq"] = "lsq"
    """Gradient discretization: ``gauss`` or ``lsq``."""
    time_scheme: Literal["euler_implicit", "backward"] = "euler_implicit"
    """Temporal discretization: ``euler_implicit`` or ``backward``."""


@dataclass
class LinearSolverConfig:
    """Momentum and pressure linear-solver settings.

    ``momentum_tol`` and ``pressure_tol`` are the absolute normalized
    residual targets.  The corresponding ``*_rel_tol`` values allow an
    intermediate PIMPLE solve to stop after reducing its initial residual by
    that factor. ``*_final_rel_tol`` controls the final momentum/pressure
    stage; ``0`` therefore requests the absolute target. Iteration
    limits are per component and pressure correction. ``amg`` uses PyAMG in
    serial and PETSc GAMG in partitioned runs.

    Examples
    --------
    >>> LinearSolverConfig(pressure_tol=1e-10, momentum_maxiter=2000)
    """

    linear_solver: Literal["bicgstab", "gmres", "cg", "amg", "spsolve"] = "bicgstab"
    """Default linear solver for all components: bicgstab, gmres, cg, amg, or spsolve."""
    momentum_solver: Literal["bicgstab", "gmres", "cg", "spsolve"] | None = None
    """Solver override for the momentum equations; None uses ``linear_solver``."""
    pressure_solver: Literal["amg", "bicgstab", "gmres", "cg", "spsolve"] | None = None
    """Solver override for the pressure correction; None uses ``linear_solver``."""
    pressure_nullspace_policy: Literal["auto", "reference", "petsc"] = "auto"
    """How the singular pressure nullspace is treated: auto, reference, or petsc."""
    linear_failure_policy: Literal["raise", "direct_fallback"] = "raise"
    """Behavior when a linear solve fails to converge: raise or direct_fallback."""
    reuse_ilu: bool = True
    """Reuse the ILU preconditioner factorization across solves."""
    momentum_tol: float = 1e-4
    """Absolute normalized residual target for momentum solves."""
    momentum_rel_tol: float = 0.0
    """Stop momentum solves once the initial residual drops by this factor; 0 requests the absolute target."""
    momentum_final_rel_tol: float | None = 0.0
    """Relative residual target for the final momentum stage; 0 requests the absolute target."""
    momentum_maxiter: int = 1000
    """Iteration limit per momentum solve."""
    pressure_tol: float = 1e-8
    """Absolute normalized residual target for pressure-correction solves."""
    pressure_rel_tol: float = 0.0
    """Stop pressure solves once the initial residual drops by this factor; 0 requests the absolute target."""
    pressure_final_rel_tol: float | None = 0.0
    """Relative residual target for the final pressure stage; 0 requests the absolute target."""
    pressure_maxiter: int = 500
    """Iteration limit per pressure-correction solve."""
    amg_tol: float | None = None
    """AMG solver tolerance; None uses the component tolerance."""
    amg_maxiter: int | None = None
    """AMG iteration limit; None uses the component limit."""
    amg_reuse_tol: float = 0.05
    """Reuse the AMG hierarchy while the residual stays below this tolerance."""
    ilu_drop_tol: float = 1e-4
    """ILU drop tolerance for the preconditioner."""
    ilu_fill_factor: float = 10.0
    """ILU fill factor for the preconditioner."""
    ilu_reuse_tol: float | None = None
    """Reuse the ILU factorization while the residual stays below this tolerance."""


@dataclass
class PimpleControl:
    """PIMPLE, PISO, or SIMPLE pressure-velocity coupling.

    Corrector counts are dimensionless. Relaxation factors lie in ``(0, 1]``;
    transient PIMPLE normally uses 1.0.

    ``alpha_u`` / ``alpha_p`` are equation relaxation factors. Transient
    PIMPLE applies them only while the outer loop is still converging, and
    runs the **final** outer corrector unrelaxed. They therefore accelerate
    the outer loop without
    altering the committed time step, and are inert when
    ``n_outer_correctors == 1``.  Steady SIMPLE relaxes every sweep, since
    there the relaxation *is* the pseudo-time march.
    """

    algorithm: Literal["SIMPLE", "PIMPLE", "PISO"] = "PIMPLE"
    """Pressure-velocity coupling algorithm: SIMPLE, PIMPLE, or PISO."""
    n_correctors: int = 2
    """Number of pressure-correction loops per step (PISO/PIMPLE)."""
    n_outer_correctors: int = 1
    """Number of outer corrector loops (PIMPLE)."""
    n_orthogonal_correctors: int = 0
    """Number of non-orthogonality correctors per pressure solve."""
    min_outer_correctors: int = 1
    """Minimum number of outer correctors always run."""
    outer_residual_tolerance: float | None = None
    """Stop the outer loop once residuals fall below this; None disables."""
    outer_continuity_tolerance: float | None = None
    """Stop the outer loop once the continuity error falls below this; None disables."""
    max_iter: int = 20
    """Hard iteration limit for the pressure solve."""
    tolerance: float = 1e-6
    """Absolute normalized residual target for pressure solves."""
    alpha_u: float = 1.0
    """Velocity under-relaxation factor in (0, 1]."""
    alpha_p: float = 1.0
    """Pressure under-relaxation factor in (0, 1]."""
    ibm_forcing_loops: int = 2
    """Number of immersed-boundary forcing sweeps per step."""
    ibm_second_solve: bool = True
    """Run a second pressure solve after immersed-boundary forcing."""


@dataclass
class TransportConfig:
    """Fluid properties (density and viscosity).

    Provides factory methods for common fluids (:meth:`air`, :meth:`water`).

    The kinematic viscosity ``nu`` defines the Reynolds number together with
    the freestream velocity and a reference length:

        Re = U_ref * L_ref / nu

    The flow equations use kinematic pressure ``p/ρ`` and volumetric face
    flux, so a spatially constant density cancels from velocity/pressure
    evolution. ``density`` converts kinematic pressure and viscosity to
    dimensional surface forces.

    Examples
    --------
    >>> TransportConfig.air()
    >>> TransportConfig.water()
    """

    density: float = 1.225
    """Fluid density (kg/m³)."""
    nu: float = 1.5e-5
    """Kinematic viscosity (m²/s)."""

    @staticmethod
    def air() -> "TransportConfig":
        """Standard air properties at sea level.

        Returns:
            :class:`TransportConfig` with ``density=1.225`` kg/m³ and
            ``nu=1.5e-5`` m²/s.
        """
        return TransportConfig(density=1.225, nu=1.5e-5)

    @staticmethod
    def water() -> "TransportConfig":
        """Standard fresh-water properties at 20°C.

        Returns:
            :class:`TransportConfig` with ``density=1000.0`` kg/m³ and
            ``nu=1.0e-6`` m²/s.
        """
        return TransportConfig(density=1000.0, nu=1.0e-6)


@dataclass
class DynamicMeshConfig:
    """Mesh motion (rigid-body or static).

    Controls whether the mesh translates and/or rotates as a rigid body, or
    remains stationary.  Translational velocity and rotational speed about a
    user-defined axis through a specified origin are set independently.

    Examples
    --------
    >>> DynamicMeshConfig.static()
    >>> DynamicMeshConfig.rigid(velocity=[1.0, 0.0, 0.0], omega=0.5)
    """

    method: Literal["static", "rigidMotion"] = "static"
    """Mesh-motion mode: ``static`` or ``rigidMotion``."""
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Rigid-body translation velocity ``[vx, vy, vz]`` (m/s)."""
    omega: float = 0.0
    """Angular velocity about the axis (rad/s)."""
    axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    """Rotation axis direction ``[ax, ay, az]``."""
    origin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Rotation centre ``[ox, oy, oz]``."""

    @staticmethod
    def static() -> "DynamicMeshConfig":
        """Create a static (no motion) mesh configuration.

        Returns:
            :class:`DynamicMeshConfig` with ``method="static"``.
        """
        return DynamicMeshConfig(method="static")

    @staticmethod
    def rigid(
        velocity: list[float] | None = None,
        omega: float = 0,
        axis: list[float] | None = None,
        origin: list[float] | None = None,
    ) -> "DynamicMeshConfig":
        """Create a rigid-body motion configuration.

        The domain translates with *velocity* and rotates with *omega*
        around *axis* through *origin*.

        Args:
            velocity: Translation velocity vector ``[vx, vy, vz]`` (m/s).
            omega:    Angular velocity (rad/s).
            axis:     Rotation axis direction ``[ax, ay, az]``.
            origin:   Rotation centre ``[ox, oy, oz]``.

        Returns:
            :class:`DynamicMeshConfig` with ``method="rigidMotion"``.
        """
        if origin is None:
            origin = [0, 0, 0]
        if axis is None:
            axis = [0, 0, 1]
        if velocity is None:
            velocity = [0, 0, 0]
        return DynamicMeshConfig(
            method="rigidMotion", velocity=velocity, omega=omega, axis=axis, origin=origin
        )


@dataclass
class TurbulenceConfig:
    """Configuration for turbulence/LES models in the FVM solver.

    Use :meth:`equilibrium_smagorinsky` for the algebraic SGS-energy form with
    a ``cubeRootVol`` filter and explicit ``Ck``/``Ce`` coefficients.

    Examples
    --------
    >>> les = TurbulenceConfig.equilibrium_smagorinsky()
    >>> les.model, les.Ck, les.Ce
    ('EquilibriumSmagorinsky', 0.094, 1.048)
    >>> setup = FVMSetup(case_name="cube", turbulence=les)
    """

    # Model name (case-insensitive): "None"/"ILES", "Smagorinsky", "WALE",
    # "sigma", "dynamicSmagorinsky".  ``Cs`` carries the model coefficient
    # (Smagorinsky Cs, WALE Cw, sigma Cσ) — use the factories below to get the
    # right default for each model.
    model: str = "None"
    """SGS model name: "None"/"ILES", "Smagorinsky", "WALE", "sigma", or
    "dynamicSmagorinsky"."""
    Cs: float = 0.17  # model coefficient (meaning depends on model)
    """Model coefficient: Smagorinsky Cs, WALE Cw, or sigma Cσ."""
    dynamic: bool = False  # Smagorinsky only: use the Germano/Lilly dynamic procedure
    """Smagorinsky only: use the Germano-Lilly dynamic procedure."""
    Ck: float = 0.094  # algebraic-equilibrium SGS energy coefficient
    """Algebraic-equilibrium SGS kinetic-energy coefficient."""
    Ce: float = 1.048  # base LES dissipation coefficient
    """Base LES dissipation coefficient."""

    @staticmethod
    def smagorinsky(Cs: float = 0.17, dynamic: bool = False) -> "TurbulenceConfig":
        """Classical Smagorinsky LES model.

        Args:
            Cs:      Smagorinsky coefficient (default 0.17).
            dynamic: If ``True``, use the Germano–Lilly dynamic procedure.

        Returns:
            :class:`TurbulenceConfig` for Smagorinsky LES.
        """
        return TurbulenceConfig(model="Smagorinsky", Cs=Cs, dynamic=dynamic)

    @staticmethod
    def equilibrium_smagorinsky(Ck: float = 0.094, Ce: float = 1.048) -> "TurbulenceConfig":
        r"""Algebraic-equilibrium Smagorinsky LES.

        The model obtains SGS kinetic energy from

        ``a=Ce/Delta``, ``b=(2/3)tr(D)``,
        ``c=2*Ck*Delta*(dev(D):D)`` and
        ``k=((-b+sqrt(b^2+4ac))/(2a))^2``, then evaluates
        ``nu_t=Ck*Delta*sqrt(k)``. For divergence-free flow, the equivalent
        classical coefficient is ``Cs=Ck^(3/4)/Ce^(1/4)``.

        Parameters
        ----------
        Ck:
            SGS kinetic-energy coefficient; default ``0.094``.
        Ce:
            SGS dissipation coefficient; default ``1.048``.

        Returns
        -------
        TurbulenceConfig
            Configuration consumed by
            :class:`source.solvers.FVM.turbulence.EquilibriumSmagorinsky`.

        Examples
        --------
        >>> cfg = TurbulenceConfig.equilibrium_smagorinsky()
        >>> round(cfg.Ck**0.75 / cfg.Ce**0.25, 3)
        0.168
        """
        equivalent_cs = Ck**0.75 / Ce**0.25 if Ck >= 0.0 and Ce > 0.0 else float("nan")
        return TurbulenceConfig(
            model="EquilibriumSmagorinsky",
            Cs=equivalent_cs,
            Ck=Ck,
            Ce=Ce,
        )

    @staticmethod
    def wale(Cw: float = 0.325) -> "TurbulenceConfig":
        """Wall-adapting WALE model (Nicoud & Ducros 1999).

        Recommended for wall-bounded LES because ν_t → 0 with the
        correct y³ near-wall scaling.

        Args:
            Cw: WALE model coefficient (default 0.325).

        Returns:
            :class:`TurbulenceConfig` for WALE LES.
        """
        return TurbulenceConfig(model="WALE", Cs=Cw)

    @staticmethod
    def sigma(Csigma: float = 1.35) -> "TurbulenceConfig":
        """sigma model (Nicoud et al. 2011).

        Produces zero ν_t in 2D, pure-shear, and solid-rotation regions,
        making it suitable for transitional flows.

        Args:
            Csigma: sigma model coefficient (default 1.35).

        Returns:
            :class:`TurbulenceConfig` for sigma LES.
        """
        return TurbulenceConfig(model="sigma", Cs=Csigma)

    @staticmethod
    def dynamic_smagorinsky() -> "TurbulenceConfig":
        """Germano–Lilly dynamic Smagorinsky (globally averaged coefficient).

        Returns:
            :class:`TurbulenceConfig` for dynamic Smagorinsky LES.
        """
        return TurbulenceConfig(model="dynamicSmagorinsky", dynamic=True)

    @staticmethod
    def none() -> "TurbulenceConfig":
        """No subgrid model (ILES / DNS).

        Pair with a low-dissipation convection scheme.

        Returns:
            :class:`TurbulenceConfig` with ``model="None"``.
        """
        return TurbulenceConfig(model="None")


@dataclass
class ExecutionConfig:
    """Sparse assembly, linear algebra, and output execution.

    ``petsc_partitioned`` stores owned cells plus one halo layer per rank and
    assembles only owned PETSc rows. The FVM state is always float64 on CPU;
    ``OutputSetup.precision`` independently controls visualization storage.
    """

    operator_backend: Literal["numpy", "numba", "taichi"] = "numpy"
    """Sparse-assembly backend: numpy, numba, or taichi."""
    linear_backend: Literal["scipy", "petsc"] = "scipy"
    """Linear-algebra backend: scipy or petsc."""
    parallel_mode: Literal["serial", "petsc_replicated", "petsc_partitioned"] = "serial"
    """Parallel execution mode: serial, petsc_replicated, or petsc_partitioned."""
    output_mode: Literal["synchronous", "threaded"] = "synchronous"
    """Visualization-output execution: synchronous or threaded."""

    @staticmethod
    def petsc_replicated() -> "ExecutionConfig":
        """Use replicated NumPy assembly with collective PETSc solves."""
        return ExecutionConfig(linear_backend="petsc", parallel_mode="petsc_replicated")

    @staticmethod
    def petsc_partitioned() -> "ExecutionConfig":
        """Use owned-plus-halo fields and owned-row PETSc solves."""
        return ExecutionConfig(linear_backend="petsc", parallel_mode="petsc_partitioned")


@dataclass
class OutputSetup:
    """ParaView visualization output policy.

    Finite-volume fields remain cell-centred in the file. ParaView can derive
    smooth display values with its ``Cell Data to Point Data`` filter without
    changing the authoritative solver output.

    That filter averages interior cells only, so it cannot show the applied
    boundary condition at a wall.  Setting ``point_interpolation`` to
    ``'boundary_weighted'`` additionally writes an inverse-distance
    interpolation of each field as point data: weighted
    weighted from the surrounding cells, and taken from the boundary faces
    at boundary points.  Cell data remains authoritative and untouched.
    """

    format: Literal["vtk_xml"] = "vtk_xml"
    """Visualization file format; only ``vtk_xml`` is supported."""
    data_location: Literal["cell"] = "cell"
    """Where fields are stored; must remain cell-centred."""
    encoding: Literal["appended"] = "appended"
    """VTK binary encoding; only ``appended`` is supported."""
    compression: Literal["lz4", "none", "zlib"] = "lz4"
    """VTK compression: lz4, none, or zlib."""
    precision: Literal["float32", "float64"] = "float64"
    """Field storage precision: float32 or float64."""
    asynchronous: bool = True
    """Write output on a background thread."""
    ghost_layers: Literal[0, 1] = 1
    """Number of ghost-cell layers written (0 or 1)."""
    point_interpolation: Literal["none", "boundary_weighted"] = "none"
    """Add interpolated point data: ``none`` or ``boundary_weighted``."""

    def __post_init__(self) -> None:
        if self.format != "vtk_xml":
            raise ValueError("Only format='vtk_xml' is currently supported")
        if self.data_location != "cell":
            raise ValueError("FVM visualization output must remain cell-centred")
        if self.encoding != "appended":
            raise ValueError("Only safe appended-binary VTK encoding is supported")
        if self.compression not in {"lz4", "none", "zlib"}:
            raise ValueError("compression must be 'lz4', 'none', or 'zlib'")
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision must be 'float32' or 'float64'")
        if not isinstance(self.asynchronous, bool):
            raise TypeError("asynchronous must be a boolean")
        if self.ghost_layers not in {0, 1}:
            raise ValueError("ghost_layers must be zero or one")
        if self.point_interpolation not in {"none", "boundary_weighted"}:
            raise ValueError("point_interpolation must be 'none' or 'boundary_weighted'")


@dataclass
class RunAcceptancePolicy:
    """Warning and abort thresholds applied to structured step diagnostics.

    Abort thresholds are evaluated over ``sustained_steps`` consecutive
    solves. Non-finite fields and failed linear solves always abort
    immediately.
    """

    sustained_steps: int = 1
    """Consecutive steps an abort threshold must be exceeded before aborting."""
    continuity_warning: float | None = None
    """Warn when continuity error exceeds this; None disables."""
    continuity_abort: float | None = None
    """Abort when continuity error exceeds this; None disables."""
    residual_warning: float | None = None
    """Warn when the normalized residual exceeds this; None disables."""
    residual_abort: float | None = None
    """Abort when the normalized residual exceeds this; None disables."""
    cfl_warning: float | None = None
    """Warn when the CFL number exceeds this; None disables."""
    cfl_abort: float | None = None
    """Abort when the CFL number exceeds this; None disables."""
    velocity_warning: float | None = None
    """Warn when the maximum velocity exceeds this; None disables."""
    velocity_abort: float | None = None
    """Abort when the maximum velocity exceeds this; None disables."""


@dataclass
class LogConfig:
    """Console and log-file verbosity.

    ``simple`` prints one table row per reported step (convergence and wall
    time); ``debug`` prints the full per-step diagnostics block and the
    performance profile.  The environment variable ``FVM_LOG`` overrides
    ``mode`` for a single run.
    """

    mode: Literal["simple", "debug"] = "simple"
    """Output verbosity: ``simple`` or ``debug``."""
    interval: int = 1
    """Log every *N*-th step."""
    console: bool = True
    """Also print diagnostics to the console."""
    filename: str = "fvm.log"
    """Log-file name."""

    def __post_init__(self) -> None:
        if self.mode not in {"simple", "debug"}:
            raise ValueError("log mode must be 'simple' or 'debug'")
        if isinstance(self.interval, bool) or not isinstance(self.interval, int):
            raise TypeError("log interval must be an integer")
        if self.interval < 1:
            raise ValueError("log interval must be at least one")
        if not isinstance(self.console, bool):
            raise TypeError("console must be a boolean")
        if not self.filename:
            raise ValueError("log filename must not be empty")


@dataclass
class FVMSetup:
    """Top-level configuration object for a finite-volume simulation.

    Aggregates all sub-configurations (mesh, time, schemes, linear solvers,
    PIMPLE control, forces, transport, turbulence, boundaries, output, and
    execution backends) into a single dataclass.  Create a fully populated
    instance and pass it to :func:`source.solvers.FVM.setup_fvm_solver`.

    Most fields accept a sub-config object; leaving one at its default
    produces a sensible baseline.  The ``case_name`` and (for non-generated
    meshes) an explicit mesh path are required.

    Examples
    --------
    >>> FVMSetup(
    ...     case_name="my_run",
    ...     schemes=SchemesConfig(convection_scheme="LUST"),
    ...     time=TimeConfig.transient(dt=0.01, duration=10.0),
    ...     boundaries=[BoundaryConfig.inlet("in", [1, 0, 0])],
    ... )
    """

    case_name: str
    """Name of the case directory, written into the run folder."""
    cores: int = 1
    """Number of MPI ranks used for the run."""
    mesh: MeshConfig = field(default_factory=MeshConfig)
    """Mesh quality limits applied to the solver-native or Gmsh mesh."""
    execution: "ExecutionConfig" = field(default_factory=ExecutionConfig)
    """Sparse-assembly, linear algebra, and output execution backends."""
    output: "OutputSetup" = field(default_factory=OutputSetup)
    """ParaView visualization output policy."""
    acceptance: "RunAcceptancePolicy" = field(default_factory=RunAcceptancePolicy)
    """Warning and abort thresholds applied to step diagnostics."""
    logging: "LogConfig" = field(default_factory=LogConfig)
    """Console and log-file verbosity."""
    time: TimeConfig = field(default_factory=TimeConfig)
    """Time integration and output cadence."""
    schemes: SchemesConfig = field(default_factory=SchemesConfig)
    """Spatial and temporal discretisation schemes."""
    linear: LinearSolverConfig = field(default_factory=LinearSolverConfig)
    """Momentum and pressure linear-solver settings."""
    pimple: PimpleControl = field(default_factory=PimpleControl)
    """PIMPLE, PISO, or SIMPLE pressure-velocity coupling control."""
    transport: TransportConfig = field(default_factory=TransportConfig)
    """Fluid properties (density and viscosity)."""
    dynamic_mesh: DynamicMeshConfig = field(default_factory=DynamicMeshConfig.static)
    """Mesh motion (rigid-body or static)."""
    boundaries: list[BoundaryConfig] = field(default_factory=list)
    """Boundary-condition specifications for the mesh patches."""
    samplers: tuple = ()
    """Field samplers evaluated at the write interval."""
    turbulence: TurbulenceConfig | None = None
    """Turbulence/LES model configuration."""
    initial_velocity: list[float] | None = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Initial velocity field ``[u, v, w]`` (m/s)."""
    initial_p: float | None = 0.0
    """Initial kinematic pressure field ``p/ρ`` [m²/s²]."""

    def __post_init__(self) -> None:
        """Validate user-facing process settings.

        Parallel backend selection is deliberately deferred to
        :func:`source.solvers.FVM.setup_fvm_solver`; case files only choose a
        core count.
        """
        if isinstance(self.cores, bool) or not isinstance(self.cores, int):
            raise TypeError("cores must be an integer")
        if self.cores < 1:
            raise ValueError("cores must be at least one")
        self.samplers = tuple(self.samplers or ())

    def algorithm_params(self) -> dict:
        """Flat parameter dict consumed by the PIMPLE/SIMPLE algorithm layer."""
        merged: dict = {}
        for group in (self.schemes, self.linear, self.pimple):
            merged.update(vars(group))
        return merged

    def save(self, filepath: str):
        """Serialise this configuration to a JSON file.

        Samplers are stored through the sampler registry, so an explicit
        :class:`~source.solvers.FVM.sampling.base.Sampler` is captured as a
        JSON-safe ``{"type": ..., ...}`` dict and reconstructed on :meth:`load`.

        Args:
            filepath: Path for the output JSON file.
        """
        from source.solvers.FVM.sampling.base import sampler_to_dict

        data = asdict(self)
        if self.samplers:
            data["samplers"] = [sampler_to_dict(s) for s in self.samplers]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, filepath: str):
        """Load a configuration from a JSON file.

        Args:
            filepath: Path to a JSON file produced by :meth:`save`.

        Returns:
            A new :class:`FVMSetup` instance.
        """
        with open(filepath) as f:
            data = json.load(f)
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown top-level FVMSetup field(s): {', '.join(unknown)}")

        # Manual reconstruction of nested dataclasses
        mesh_data = data.get("mesh", {})
        execution_data = data.get("execution", {})
        output_data = data.get("output", {})
        acceptance_data = data.get("acceptance", {})
        logging_data = data.get("logging", {})
        time_data = data.get("time", {})
        transport_data = data.get("transport", {})
        dynamic_mesh_data = data.get("dynamic_mesh", {"method": "static"})
        turbulence_data = data.get("turbulence", None)

        mesh = MeshConfig(**mesh_data)
        execution = ExecutionConfig(**execution_data)
        output = OutputSetup(**output_data)
        acceptance = RunAcceptancePolicy(**acceptance_data)
        log = LogConfig(**logging_data)
        time = TimeConfig(**time_data)
        dynamic_mesh = DynamicMeshConfig(**dynamic_mesh_data)

        transport = TransportConfig(**transport_data)
        boundaries = [BoundaryConfig(**b) for b in data.get("boundaries", [])]
        turbulence = TurbulenceConfig(**turbulence_data) if turbulence_data else None

        from source.solvers.FVM.sampling.base import sampler_from_dict

        samplers = tuple(
            sampler_from_dict(d) for d in data.get("samplers", []) if isinstance(d, dict)
        )

        return cls(
            case_name=data["case_name"],
            cores=data.get("cores", 1),
            mesh=mesh,
            execution=execution,
            output=output,
            acceptance=acceptance,
            logging=log,
            time=time,
            schemes=SchemesConfig(**data.get("schemes", {})),
            linear=LinearSolverConfig(**data.get("linear", {})),
            pimple=PimpleControl(**data.get("pimple", {})),
            samplers=samplers,
            transport=transport,
            dynamic_mesh=dynamic_mesh,
            boundaries=boundaries,
            turbulence=turbulence,
            initial_velocity=data.get("initial_velocity", [0.0, 0.0, 0.0]),
            initial_p=data.get("initial_p", 0.0),
        )
