from dataclasses import asdict, dataclass, field
import json
import os
import re
from typing import Literal


def _collect_field_patches(filepath: str, type_key: str, value_key: str, patches: dict) -> None:
    """Read a single OpenFOAM field file and merge its BC entries into *patches*.

    Args:
        filepath:  Path to an OpenFOAM field file (e.g. ``0/U``).
        type_key:  Key under which the boundary type is stored (e.g. ``"type_U"``).
        value_key: Key under which the boundary value is stored (e.g. ``"value_U"``).
        patches:   Dict of patch-name → attributes, mutated in place.
    """
    from ..fields.field_io import parse_field_boundary_patches

    if not os.path.exists(filepath):
        return
    for k, v in parse_field_boundary_patches(filepath).items():
        patches.setdefault(k, {})[type_key] = v.get("type")
        patches.setdefault(k, {})[value_key] = v.get("value")


def _build_boundary_config(name: str, vals: dict) -> "BoundaryConfig":
    """Construct a :class:`BoundaryConfig` from merged patch data.

    Iterates over the ``(type_attr, value_attr)`` pairs for U, p, and nut,
    setting only those that appear in *vals*.

    Args:
        name: Patch name.
        vals: Merged attributes dict (from :func:`_collect_field_patches`).

    Returns:
        A new :class:`BoundaryConfig`.
    """
    missing = [key for key in ("type_U", "type_p") if vals.get(key) is None]
    if missing:
        raise ValueError(
            f"Boundary patch {name!r} is missing required field definitions: {', '.join(missing)}"
        )
    b = BoundaryConfig(name=name)
    for type_attr, value_attr in (
        ("type_U", "value_U"),
        ("type_p", "value_p"),
        ("type_nut", "value_nut"),
    ):
        t = vals.get(type_attr)
        v = vals.get(value_attr)
        if t is not None:
            setattr(b, type_attr, t)
            if v is not None:
                setattr(b, value_attr, v)
    return b


def _resolve_system_dir(path: str) -> str:
    """Resolve a path to the OpenFOAM ``system`` directory.

    If *path* is a case directory (a directory whose basename is not
    ``system``), appends ``"system"``.  If *path* is a file, returns its
    parent directory.

    Args:
        path: Candidate path (directory or file).

    Returns:
        Absolute path to the ``system`` directory.
    """
    if os.path.isdir(path) and os.path.basename(path) != "system":
        return os.path.join(path, "system")
    return path if os.path.isdir(path) else os.path.dirname(path)


def _parse_yplus_patches(content: str) -> list[str] | None:
    """Extract the y+ patch list from the ``functions`` block of a ``controlDict``.

    Parses the ``functions`` sub-dictionary for an entry matching ``"yplus"``
    (case-insensitive) and extracts the ``patches ( ... )`` list.

    Args:
        content: Raw text of the ``controlDict`` file.

    Returns:
        List of patch names, or ``None`` if no y+ function is found.
    """
    funcs_match = re.search(r"functions\s*\{(.+?)\}\s*", content, re.DOTALL)
    if not funcs_match:
        return None
    funcs_content = funcs_match.group(1)
    for fm in re.finditer(r"(\w+)\s*\{([^}]*)\}", funcs_content, re.DOTALL):
        if "yplus" not in fm.group(1).lower():
            continue
        pm = re.search(r"patches\s*\(\s*([^\)]+)\)", fm.group(2))
        if pm:
            tokens = [t.strip() for t in pm.group(1).split() if t.strip()]
            if tokens:
                return tokens
    return None


def _extract_foam_number(v) -> float | None:
    """Parse a numeric value from an OpenFOAM dictionary entry.

    Handles scalars, single-element lists, and strings containing numbers
    (takes the last numeric token found).

    Args:
        v: Raw value from a parsed OpenFOAM dictionary (scalar, list, or string).

    Returns:
        Float value, or ``None`` if no number could be extracted.
    """
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, list) and v:
        return float(v[-1])
    if isinstance(v, str):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)
        if nums:
            return float(nums[-1])
    return None


def _normalise_openfoam_linear_solver(name: str, field_name: str) -> str:
    """Map supported OpenFOAM solver names to the FVM linear API."""
    mapping = {
        "gamg": "amg",
        "pcg": "cg",
        "cg": "cg",
        "pbicgstab": "bicgstab",
        "bicgstab": "bicgstab",
        "gmres": "gmres",
        "spsolve": "spsolve",
    }
    key = str(name).strip().lower()
    if key not in mapping:
        raise ValueError(
            f"Unsupported OpenFOAM linear solver {name!r} for {field_name}; "
            f"supported names: {sorted(mapping)}"
        )
    return mapping[key]


def _find_supported_scheme(
    value: str,
    mapping: dict[str, str],
    label: str,
    allowed_modifiers: set[str] | None = None,
) -> str:
    """Resolve a supported OpenFOAM scheme token from a complete entry."""
    tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", value)]
    allowed = set(mapping) | (allowed_modifiers or set())
    unsupported = [token for token in tokens if token not in allowed]
    if unsupported:
        raise ValueError(f"Unsupported {label} modifier(s) {unsupported!r} in entry {value!r}")
    for token in reversed(tokens):
        if token in mapping:
            return mapping[token]
    raise ValueError(f"Unsupported {label} entry {value!r}; supported schemes: {sorted(mapping)}")


def _detect_turbulence_model(data: dict, filepath: str) -> str:
    """Determine the turbulence model name from parsed data and raw file text.

    Checks ``simulationType``, ``LESModel``, and ``model`` keys.  If the
    simulation type is ``LES`` but the model name is not present in the
    parsed data, falls back to regex on the raw file content.

    Args:
        data:     Parsed dictionary from ``turbulenceProperties``.
        filepath: Original file path (read again if regex fallback needed).

    Returns:
        Model name string (e.g. ``"Smagorinsky"``, ``"None"``).
    """
    sim_type = data.get("simulationType") or data.get("simulationtype")
    model = data.get("LESModel") or data.get("model")
    if model is None and sim_type and str(sim_type).upper() == "LES":
        with open(filepath) as f:
            txt = f.read()
        m = re.search(r"LESModel\s+(\w+);", txt)
        if m:
            return m.group(1)
    return str(model or sim_type or "None")


def _find_turbulence_coeffs(data: dict, filepath: str) -> tuple[float, bool]:
    """Return ``(Cs, dynamic)`` turbulence coefficients.

    Checks for ``Cs`` (or variants ``C_s``, ``c_s``) and ``dynamic`` in the
    parsed data.  If missing, falls back to regex on the raw file content.

    Args:
        data:     Parsed dictionary from ``turbulenceProperties``.
        filepath: Original file path (used for regex fallback).

    Returns:
        Tuple ``(Cs, is_dynamic)``; defaults ``(0.17, False)`` if unset.
    """
    cs_candidates = [data.get("Cs"), data.get("C_s"), data.get("c_s")]
    cs_val = next((c for c in cs_candidates if c is not None), None)
    dynamic_val = data.get("dynamic")
    if cs_val is None or dynamic_val is None:
        with open(filepath) as f:
            txt = f.read()
        if cs_val is None:
            m = re.search(r"Cs\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?);", txt)
            if m:
                cs_val = float(m.group(1))
        if dynamic_val is None:
            dm = re.search(r"dynamic\s+(\w+);", txt)
            if dm:
                dynamic_val = dm.group(1)
    dynamic = (
        dynamic_val.strip().lower() in ("true", "on", "yes", "1")
        if isinstance(dynamic_val, str)
        else bool(dynamic_val)
    )
    return float(cs_val) if cs_val is not None else 0.17, dynamic


@dataclass
class BoundaryConfig:
    """
    Configuration for a boundary patch.
    """

    name: str  # e.g. "inlet", "outlet"
    type_U: str = "fixedValue"  # fixedValue, zeroGradient, etc.
    value_U: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    type_p: str = "zeroGradient"
    value_p: float = 0.0
    type_phi: str = "zeroGradient"  # For scalar transport
    value_phi: float = 0.0
    # Nut (turbulent viscosity) specific BC (OpenFOAM supports 'calculated')
    type_nut: str = "calculated"
    value_nut: float = 0.0
    neighbour_patch: str | None = None
    mesh_type: Literal["patch", "wall", "empty", "cyclic"] | None = None

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
            name=name, type_U="fixedValue", value_U=velocity, type_p="zeroGradient"
        )

    @staticmethod
    def outlet(name: str, p: float = 0.0) -> "BoundaryConfig":
        """Create an outlet with fixed pressure and ``inletOutlet`` velocity.

        Uses ``inletOutlet`` for velocity: zero-gradient on outflow, zero
        inflow on reverse flow.  This is bounded and does not rely on
        patch-name heuristics.

        Args:
            name: Patch name (e.g. ``"outlet"``).
            p:    Prescribed static pressure (default 0.0).

        Returns:
            A new :class:`BoundaryConfig` suitable for outflow boundaries.
        """
        return BoundaryConfig(name=name, type_U="inletOutlet", type_p="fixedValue", value_p=p)

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
            name=name, type_U="freestream", value_U=velocity, type_p="freestream", value_p=p
        )

    @staticmethod
    def cyclic(name: str, neighbour_patch: str) -> "BoundaryConfig":
        """Create one side of a translational periodic patch pair."""
        return BoundaryConfig(
            name=name,
            type_U="cyclic",
            type_p="cyclic",
            neighbour_patch=neighbour_patch,
            mesh_type="cyclic",
        )

    @staticmethod
    def wall(name: str) -> "BoundaryConfig":
        """Create a no-slip wall boundary condition.

        Sets velocity to zero (fixed value) and pressure to zero-gradient.
        Turbulent viscosity uses the ``calculated`` type (OpenFOAM convention).

        Args:
            name: Patch name (e.g. ``"bottomWall"``).

        Returns:
            A new :class:`BoundaryConfig` for solid walls.
        """
        return BoundaryConfig(
            name=name,
            type_U="fixedValue",
            value_U=[0.0, 0.0, 0.0],
            type_p="zeroGradient",
            type_nut="calculated",
            mesh_type="wall",
        )

    @staticmethod
    def slip(name: str) -> "BoundaryConfig":
        """Create an impermeable, zero-shear boundary."""
        return BoundaryConfig(
            name=name,
            type_U="slip",
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
            type_U="empty",
            value_U=[0.0, 0.0, 0.0],
            type_p="empty",
            type_nut="zeroGradient",
            mesh_type="empty",
        )

    @staticmethod
    def load_from_time_dir(case_dir: str, time_dir: str = "0") -> list["BoundaryConfig"]:
        """Load boundary conditions from an OpenFOAM time directory.

        Reads ``U``, ``p``, and ``nut`` field files (if present) and
        constructs a :class:`BoundaryConfig` for every patch found.

        Args:
            case_dir: Case root directory.
            time_dir: Time subdirectory name (default ``"0"``).

        Returns:
            List of :class:`BoundaryConfig` objects, one per patch.

        Example:
            >>> bcs = BoundaryConfig.load_from_time_dir("/path/to/case", "0")
            >>> bcs[0].name, bcs[0].type_U
            ('inlet', 'fixedValue')
        """
        base = os.path.join(case_dir, time_dir)
        patches: dict = {}
        _collect_field_patches(os.path.join(base, "U"), "type_U", "value_U", patches)
        _collect_field_patches(os.path.join(base, "p"), "type_p", "value_p", patches)
        _collect_field_patches(os.path.join(base, "nut"), "type_nut", "value_nut", patches)
        return [_build_boundary_config(name, vals) for name, vals in patches.items()]


@dataclass
class MeshConfig:
    """
    Configuration for mesh generation.
    Supports OpenFOAM's blockMesh and cfMesh (cartesianMesh).
    """

    method: Literal["blockMesh", "cartesianMesh"] = "blockMesh"
    surface_file: str | None = None  # For cartesianMesh: Expects this in constant/triSurface/
    max_cell_size: float = 1.0  # For cartesianMesh
    boundary_cell_size: float | None = None  # For cartesianMesh
    min_cell_size: float | None = None  # For cartesianMesh
    local_refinement: dict[str, float] = field(
        default_factory=dict
    )  # patch -> size (cartesianMesh)
    max_non_orthogonality_deg: float | None = None
    max_skewness: float | None = None
    max_aspect_ratio: float | None = None
    max_lsq_condition: float | None = None

    @staticmethod
    def block_mesh() -> "MeshConfig":
        """Create a default ``blockMesh`` configuration.

        Returns:
            :class:`MeshConfig` configured for OpenFOAM's ``blockMesh``.
        """
        return MeshConfig(method="blockMesh")

    @staticmethod
    def cartesian(surface_file: str, max_cell_size: float) -> "MeshConfig":
        """Create a cfMesh ``cartesianMesh`` configuration.

        Args:
            surface_file:  STL/OBJ surface file in ``constant/triSurface/``.
            max_cell_size: Maximum isotropic cell size.

        Returns:
            :class:`MeshConfig` configured for cfMesh ``cartesianMesh``.
        """
        return MeshConfig(
            method="cartesianMesh", surface_file=surface_file, max_cell_size=max_cell_size
        )

    def generate_mesh_dict(self) -> str:
        """Generate the content of ``system/meshDict`` for cfMesh cartesianMesh.

        Produces an OpenFOAM-format dictionary with ``surfaceFile``,
        ``maxCellSize``, and optional ``boundaryCellSize``, ``minCellSize``,
        and ``localRefinement`` entries.

        Returns:
            Complete ``meshDict`` content as a string, or ``""`` if the
            configured method is not ``cartesianMesh``.
        """
        if self.method != "cartesianMesh":
            return ""

        content = [
            "/*--------------------------------*- C++ -*----------------------------------*\\",
            "| =========                 |                                                 |",
            "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |",
            "|  \\\\    /   O peration     | Version:  v2006                                 |",
            "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |",
            "|    \\\\/     M anipulation  |                                                 |",
            "\\*---------------------------------------------------------------------------*/",
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            "    object      meshDict;",
            "}",
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //",
            "",
            f'surfaceFile "{self.surface_file}";',
            "",
            f"maxCellSize {self.max_cell_size};",
            "",
        ]

        if self.boundary_cell_size is not None:
            content.append(f"boundaryCellSize {self.boundary_cell_size};")

        if self.min_cell_size is not None:
            content.append(f"minCellSize {self.min_cell_size};")

        if self.local_refinement:
            content.append("localRefinement")
            content.append("{")
            for patch, size in self.local_refinement.items():
                content.append(f"    {patch}")
                content.append("    {")
                content.append(f"        cellSize {size};")
                content.append("    }")
            content.append("}")

        content.append("")
        content.append(
            "// ************************************************************************* //"
        )
        return "\n".join(content)


@dataclass
class TimeConfig:
    """
    Time control configuration.
    """

    delta_t: float = 0.01
    start_time: float = 0.0
    end_time: float = 1.0
    write_interval: int = 10
    write_interval_time: float | None = None
    adjust_timestep: bool = False
    max_cfl: float = 1.0
    max_delta_t: float = 0.1
    min_delta_t: float = 1e-4
    dt_adjust_coeff: float = 1.2

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

    @classmethod
    def from_control_dict(cls, path: str) -> "TimeConfig":
        """Create a TimeConfig from an OpenFOAM `controlDict` file or case dir.

        This reads `deltaT` (or `deltaT`), `startTime`, and `endTime`.
        """
        from ..fields.field_io import parse_simple_dictionary

        filepath = os.path.join(path, "system", "controlDict") if os.path.isdir(path) else path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"controlDict not found: {filepath}")
        data = parse_simple_dictionary(filepath)
        missing = [key for key in ("deltaT", "endTime") if key not in data]
        if missing:
            raise ValueError(
                f"controlDict {filepath!r} is missing required entries: {', '.join(missing)}"
            )
        dt = float(data.get("deltaT", cls().delta_t))
        start = float(data.get("startTime", cls().start_time))
        end = float(data.get("endTime", cls().end_time))
        w = int(data.get("writeInterval", cls().write_interval))
        return cls(delta_t=dt, start_time=start, end_time=end, write_interval=w)


@dataclass
class SolverParams:
    """
    Algorithm parameters for Simple/Pimple.
    """

    algorithm: Literal["SIMPLE", "PIMPLE", "PISO"] = "PIMPLE"
    n_correctors: int = 2
    n_outer_correctors: int = 1
    n_orthogonal_correctors: int = 0
    min_outer_correctors: int = 1
    outer_residual_tolerance: float | None = None
    outer_continuity_tolerance: float | None = None
    max_iter: int = 20
    tolerance: float = 1e-6
    alpha_u: float = 0.7
    alpha_p: float = 0.3
    linear_solver: str = "bicgstab"  # Defaulting to iterative for performance
    # Equation-specific methods. None inherits ``linear_solver`` for backward
    # compatibility with existing Python configurations.
    momentum_solver: str | None = None
    pressure_solver: str | None = None
    pressure_nullspace_policy: Literal["auto", "reference", "petsc"] = "auto"
    linear_failure_policy: Literal["raise", "direct_fallback"] = "raise"
    reuse_ilu: bool = True  # Enable ILU preconditioning reuse by default

    # y+ patches: user can specify patches to compute y+ for (None => auto-select wall patches)
    yplus_patches: list[str] | None = None

    # Momentum solver tunables
    momentum_tol: float = 1e-4
    momentum_maxiter: int = 1000
    pressure_tol: float = 1e-8
    pressure_maxiter: int = 500
    # ``None`` makes AMG inherit the pressure tolerance / iteration limit.
    amg_tol: float | None = None
    amg_maxiter: int | None = None
    amg_reuse_tol: float = 0.05
    ilu_drop_tol: float = 1e-4
    ilu_fill_factor: float = 10.0
    # ILU reuse heuristic: if set (e.g., 1e-3) the ILU cache will be reused only if the
    # relative change in diagonal is <= this threshold. If None, always reuse when pattern matches.
    ilu_reuse_tol: float | None = None

    # Force computation patches: None = auto-detect walls
    force_patches: list[str] | None = None
    # Reference velocity magnitude for Cd/Cl/Cz
    ref_velocity: float = 1.0
    # Reference area for Cd/Cl/Cz
    ref_area: float = 1.0
    # Reference length for Cm (pitching moment)
    ref_length: float = 1.0
    # Log forces every N steps (defaults to write_interval if None)
    force_log_interval: int | None = None
    # Moment centre for Cm computation (default [0,0,0])
    moment_centre: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Immersed boundary: multidirect residual-forcing iterations per outer
    # corrector (only used when bodies are attached via set_immersed_bodies).
    ibm_forcing_loops: int = 2
    ibm_second_solve: bool = True

    # Schemes (OpenFOAM-like)
    # Default to a bounded 2nd-order TVD scheme: far less diffusive than upwind
    # (the previous default) yet oscillation-free, unlike pure central.  Other
    # options: "upwind", "central"/"linear" (energy-conserving, unbounded),
    # "deferred", "LUST", "vanLeer", "MUSCL", "minmod", "superbee".
    convection_scheme: str = "limitedLinear"
    time_scheme: str = "euler_implicit"  # "euler_implicit" (BDF1) | "backward" (BDF2)
    gradient_scheme: str = "lsq"  # "gauss" | "lsq"

    @staticmethod
    def pimple(
        n_correctors: int = 2,
        n_outer: int = 1,
        n_non_orthogonal: int = 0,
        linear_solver: str = "spsolve",
        alpha_u: float = 1.0,
        alpha_p: float = 1.0,
        convection_scheme: str = "deferred",
        gradient_scheme: str = "lsq",
        time_scheme: str = "euler_implicit",
        momentum_tol: float = 1e-4,
        momentum_maxiter: int = 1000,
        pressure_tol: float = 1e-8,
        pressure_maxiter: int = 500,
        momentum_solver: str | None = None,
        pressure_solver: str | None = None,
        outer_residual_tolerance: float | None = None,
        outer_continuity_tolerance: float | None = None,
    ) -> "SolverParams":
        """Create PIMPLE solver parameters with transient-appropriate defaults.

        For transient (PIMPLE/PISO) simulations, under-relaxation should be
        1.0 (no relaxation) unless outer correctors are used.  The default
        ``"deferred"`` convection scheme is second-order central applied via
        deferred correction (energy-conserving but *unbounded*); for
        under-resolved LES prefer a bounded TVD scheme (e.g. ``"limitedLinear"``)
        or pair ``"deferred"`` with enough correctors.

        ``time_scheme`` selects the time discretisation: ``"euler_implicit"``
        (BDF1, first order, default) or ``"backward"`` (BDF2, second order).
        BDF2 needs ≥1 outer/PISO corrector per step for the deferred convection
        correction to keep pace; pair it with ``n_correctors>=2``.
        """
        return SolverParams(
            algorithm="PIMPLE",
            n_correctors=n_correctors,
            n_outer_correctors=n_outer,
            n_orthogonal_correctors=n_non_orthogonal,
            linear_solver=linear_solver,
            momentum_solver=momentum_solver,
            pressure_solver=pressure_solver,
            alpha_u=alpha_u,
            alpha_p=alpha_p,
            convection_scheme=convection_scheme,
            gradient_scheme=gradient_scheme,
            time_scheme=time_scheme,
            momentum_tol=momentum_tol,
            momentum_maxiter=momentum_maxiter,
            pressure_tol=pressure_tol,
            pressure_maxiter=pressure_maxiter,
            outer_residual_tolerance=outer_residual_tolerance,
            outer_continuity_tolerance=outer_continuity_tolerance,
        )

    @staticmethod
    def piso(
        n_correctors: int = 2,
        n_non_orthogonal: int = 0,
        linear_solver: str = "spsolve",
        convection_scheme: str = "deferred",
        gradient_scheme: str = "lsq",
        time_scheme: str = "euler_implicit",
        momentum_solver: str | None = None,
        pressure_solver: str | None = None,
    ) -> "SolverParams":
        """Create transient PISO parameters with exactly one outer corrector."""
        params = SolverParams.pimple(
            n_correctors=n_correctors,
            n_outer=1,
            n_non_orthogonal=n_non_orthogonal,
            linear_solver=linear_solver,
            convection_scheme=convection_scheme,
            gradient_scheme=gradient_scheme,
            time_scheme=time_scheme,
            momentum_solver=momentum_solver,
            pressure_solver=pressure_solver,
        )
        params.algorithm = "PISO"
        return params

    @staticmethod
    def simple(
        alpha_u: float = 0.7,
        alpha_p: float = 0.3,
        linear_solver: str = "spsolve",
        gradient_scheme: str = "lsq",
    ) -> "SolverParams":
        """Create SIMPLE solver parameters with steady-state defaults.

        Args:
            alpha_u:        Velocity under-relaxation factor (default 0.7).
            alpha_p:        Pressure under-relaxation factor (default 0.3).
            linear_solver:  Linear system solver (default ``"spsolve"``).
            gradient_scheme: Gradient scheme, ``"lsq"`` or ``"gauss"``.

        Returns:
            :class:`SolverParams` configured for steady SIMPLE.
        """
        return SolverParams(
            algorithm="SIMPLE",
            alpha_u=alpha_u,
            alpha_p=alpha_p,
            linear_solver=linear_solver,
            gradient_scheme=gradient_scheme,
        )

    @classmethod
    def from_system(cls, path: str) -> "SolverParams":
        """Attempt to construct solver parameters from `system/` dictionary files.

        Args:
            path: Case directory or directly the `system` path (e.g., 'system').
        """
        from ..fields.field_io import parse_simple_dictionary

        system_dir = _resolve_system_dir(path)
        control = os.path.join(system_dir, "controlDict")
        fvsolution = os.path.join(system_dir, "fvSolution")
        fvschemes = os.path.join(system_dir, "fvSchemes")
        params = cls()
        if os.path.exists(control):
            data = parse_simple_dictionary(control)
            dt = data.get("deltaT")
            if dt is not None:
                params.time_scheme = "euler_implicit" if float(dt) > 0 else params.time_scheme
            with open(control) as f:
                content = f.read()
            patches = _parse_yplus_patches(content)
            if patches:
                params.yplus_patches = patches
        if not os.path.exists(fvsolution):
            raise FileNotFoundError(f"fvSolution not found: {fvsolution}")
        if not os.path.exists(fvschemes):
            raise FileNotFoundError(f"fvSchemes not found: {fvschemes}")

        from ..fields.field_io import _extract_braced_block, _strip_comments

        with open(fvsolution) as f:
            content = _strip_comments(f.read())

        algorithm_blocks = []
        for algorithm in ("PIMPLE", "PISO", "SIMPLE"):
            try:
                body = _extract_braced_block(content, algorithm)
            except ValueError:
                continue
            algorithm_blocks.append((algorithm, body))
        if len(algorithm_blocks) != 1:
            names = ", ".join(name for name, _ in algorithm_blocks)
            detail = f"found {names}" if names else "found none"
            raise ValueError(
                f"fvSolution must define exactly one PIMPLE/PISO/SIMPLE block; {detail}"
            )
        algorithm, body = algorithm_blocks[0]
        params.algorithm = algorithm
        entries = {key: value.strip() for key, value in re.findall(r"(\w+)\s+([^;{}]+);", body)}
        if "nCorrectors" in entries:
            params.n_correctors = int(entries["nCorrectors"])
        if "nOuterCorrectors" in entries:
            params.n_outer_correctors = int(entries["nOuterCorrectors"])
        if "nNonOrthogonalCorrectors" in entries:
            params.n_orthogonal_correctors = int(entries["nNonOrthogonalCorrectors"])

        try:
            solvers_body = _extract_braced_block(content, "solvers")
        except ValueError as error:
            raise ValueError(f"fvSolution {fvsolution!r} has no solvers dictionary") from error

        selected = {}
        for unsupported_name in ("UFinal", "pFinal"):
            try:
                _extract_braced_block(solvers_body, unsupported_name)
            except ValueError:
                continue
            raise ValueError(
                f"fvSolution entry {unsupported_name!r} is unsupported because the Python "
                "solver has no separate final linear-solve stage"
            )
        for field_name, attribute in (("U", "momentum_solver"), ("p", "pressure_solver")):
            try:
                field_body = _extract_braced_block(solvers_body, field_name)
            except ValueError:
                continue
            solver_match = re.search(r"\bsolver\s+(\w+)\s*;", field_body)
            if solver_match is None:
                raise ValueError(f"fvSolution solver entry {field_name!r} has no solver method")
            method = _normalise_openfoam_linear_solver(solver_match.group(1), field_name)
            setattr(params, attribute, method)
            selected[field_name] = method
            tolerance_match = re.search(
                r"\btolerance\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*;", field_body
            )
            if tolerance_match:
                setattr(
                    params,
                    "momentum_tol" if field_name == "U" else "pressure_tol",
                    float(tolerance_match.group(1)),
                )
            reltol_match = re.search(
                r"\brelTol\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*;", field_body
            )
            if reltol_match and float(reltol_match.group(1)) != 0.0:
                raise ValueError(
                    f"fvSolution {field_name} relTol is unsupported; set relTol 0 and use tolerance"
                )
            maxiter_match = re.search(r"\bmaxIter\s+(\d+)\s*;", field_body)
            if maxiter_match:
                setattr(
                    params,
                    "momentum_maxiter" if field_name == "U" else "pressure_maxiter",
                    int(maxiter_match.group(1)),
                )
        missing_solvers = sorted({"U", "p"} - set(selected))
        if missing_solvers:
            raise ValueError(
                f"fvSolution must define explicit solver blocks for: {', '.join(missing_solvers)}"
            )
        params.linear_solver = selected["U"]

        with open(fvschemes) as f:
            schemes_content = _strip_comments(f.read())
        try:
            ddt_body = _extract_braced_block(schemes_content, "ddtSchemes")
            grad_body = _extract_braced_block(schemes_content, "gradSchemes")
            div_body = _extract_braced_block(schemes_content, "divSchemes")
        except ValueError as error:
            raise ValueError(
                f"fvSchemes {fvschemes!r} is missing a required scheme block"
            ) from error

        ddt_match = re.search(r"\bdefault\s+([^;]+);", ddt_body)
        grad_match = re.search(r"\bdefault\s+([^;]+);", grad_body)
        div_match = re.search(r"div\s*\(\s*phi\s*,\s*U\s*\)\s+([^;]+);", div_body)
        if ddt_match is None or grad_match is None or div_match is None:
            raise ValueError(
                "fvSchemes must define ddtSchemes/default, gradSchemes/default, "
                "and divSchemes/div(phi,U)"
            )
        params.time_scheme = _find_supported_scheme(
            ddt_match.group(1),
            {"euler": "euler_implicit", "backward": "backward"},
            "time",
        )
        params.gradient_scheme = _find_supported_scheme(
            grad_match.group(1),
            {"linear": "gauss", "leastsquares": "lsq"},
            "gradient",
            {"gauss"},
        )
        params.convection_scheme = _find_supported_scheme(
            div_match.group(1),
            {
                "upwind": "upwind",
                "linear": "central",
                "limitedlinear": "limitedLinear",
                "lust": "LUST",
                "vanleer": "vanLeer",
                "muscl": "MUSCL",
                "minmod": "minmod",
                "superbee": "superbee",
            },
            "convection",
            {"bounded", "gauss"},
        )
        return params


@dataclass
class TransportConfig:
    """
    Transport and fluid properties.
    """

    density: float = 1.225
    nu: float = 1.5e-5

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

    @classmethod
    def from_foam_file(cls, path: str) -> "TransportConfig":
        """Load transport properties from an OpenFOAM-style dictionary.

        Args:
            path: Path to `transportProperties` file or a case directory that
                  contains `constant/transportProperties`.
        """
        from ..fields.field_io import parse_simple_dictionary

        if os.path.isdir(path):
            filepath = os.path.join(path, "constant", "transportProperties")
        else:
            filepath = path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Transport file not found: {filepath}")

        data = parse_simple_dictionary(filepath)
        rho_raw = data["rho"] if "rho" in data else data.get("Rho")
        rho_val = _extract_foam_number(rho_raw) if rho_raw is not None else cls().density
        if rho_val is None or not float(rho_val) > 0.0:
            raise ValueError(f"Density must be positive; got {rho_raw!r}")
        nu_raw = data.get("nu")
        nu_val = _extract_foam_number(nu_raw) if nu_raw is not None else None
        if nu_val is None:
            mu_val = _extract_foam_number(data.get("mu"))
            if mu_val is not None:
                nu_val = float(mu_val) / float(rho_val)
        if nu_val is None:
            raise ValueError(
                f"Transport file {filepath!r} must define kinematic viscosity 'nu' "
                "or dynamic viscosity 'mu'"
            )
        if not float(nu_val) > 0.0:
            raise ValueError(f"Kinematic viscosity must be positive; got {nu_val!r}")
        return cls(density=float(rho_val), nu=float(nu_val))


@dataclass
class DynamicMeshConfig:
    """
    Configuration for predictable dynamic mesh motion.
    """

    method: Literal["static", "rigidMotion"] = "static"
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [vx, vy, vz]
    omega: float = 0.0  # rad/s (rotation around axis)
    axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    origin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

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
    """Configuration for turbulence/LES models (FVM).

    This mirrors an OpenFOAM-style 'turbulenceProperties' dictionary.
    """

    # Model name (case-insensitive): "None"/"ILES", "Smagorinsky", "WALE",
    # "sigma", "dynamicSmagorinsky".  ``Cs`` carries the model coefficient
    # (Smagorinsky Cs, WALE Cw, sigma Cσ) — use the factories below to get the
    # right default for each model.
    model: str = "None"
    Cs: float = 0.17  # model coefficient (meaning depends on model)
    dynamic: bool = False  # Smagorinsky only: use the Germano/Lilly dynamic procedure

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

    @classmethod
    def from_foam_file(cls, path: str) -> "TurbulenceConfig":
        """Load turbulenceProperties from case directory or file path.

        Looks for `constant/turbulenceProperties` and reads `simulationType` or
        `LESModel`/`model` fields. Supports a simple Smagorinsky dictionary layout.
        """
        from ..fields.field_io import parse_simple_dictionary

        if os.path.isdir(path):
            filepath = os.path.join(path, "constant", "turbulenceProperties")
        else:
            filepath = path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"turbulenceProperties not found: {filepath}")
        data = parse_simple_dictionary(filepath)
        model = _detect_turbulence_model(data, filepath)
        cs, dynamic = _find_turbulence_coeffs(data, filepath)
        return cls(model=model, Cs=cs, dynamic=dynamic)


@dataclass
class ExecutionConfig:
    """Execution and sparse-linear-algebra backend selection.

    ``petsc_partitioned`` stores owned cells plus one halo layer per rank and
    assembles only owned PETSc rows. ``petsc_replicated`` remains available as
    a compatibility/reference mode.
    """

    operator_backend: Literal["numpy", "numba", "taichi"] = "numpy"
    linear_backend: Literal["scipy", "petsc"] = "scipy"
    parallel_mode: Literal["serial", "petsc_replicated", "petsc_partitioned"] = "serial"
    device: Literal["cpu", "cuda", "metal", "vulkan"] = "cpu"
    precision: Literal["float64"] = "float64"
    output_mode: Literal["synchronous", "threaded"] = "synchronous"

    @staticmethod
    def petsc_replicated() -> "ExecutionConfig":
        """Use replicated NumPy assembly with collective PETSc solves."""
        return ExecutionConfig(linear_backend="petsc", parallel_mode="petsc_replicated")

    @staticmethod
    def petsc_partitioned() -> "ExecutionConfig":
        """Use owned-plus-halo fields and owned-row PETSc solves."""
        return ExecutionConfig(linear_backend="petsc", parallel_mode="petsc_partitioned")


@dataclass
class RunAcceptancePolicy:
    """Warning and abort thresholds applied to structured step diagnostics.

    Abort thresholds are evaluated over ``sustained_steps`` consecutive
    solves. Non-finite fields and failed linear solves always abort
    immediately.
    """

    sustained_steps: int = 1
    continuity_warning: float | None = None
    continuity_abort: float | None = None
    residual_warning: float | None = None
    residual_abort: float | None = None
    cfl_warning: float | None = None
    cfl_abort: float | None = None
    velocity_warning: float | None = None
    velocity_abort: float | None = None


@dataclass
class FVMConfig:
    """
    Top-level FVM configuration.
    """

    case_name: str
    mesh: MeshConfig = field(default_factory=MeshConfig.block_mesh)
    execution: "ExecutionConfig" = field(default_factory=lambda: ExecutionConfig())
    acceptance: "RunAcceptancePolicy" = field(default_factory=lambda: RunAcceptancePolicy())
    time: TimeConfig = field(default_factory=TimeConfig)
    solver: SolverParams = field(default_factory=SolverParams)
    transport: TransportConfig = field(default_factory=TransportConfig)
    dynamic_mesh: DynamicMeshConfig = field(default_factory=DynamicMeshConfig.static)
    boundaries: list[BoundaryConfig] = field(default_factory=list)
    turbulence: TurbulenceConfig | None = None
    initial_U: list[float] | None = field(default_factory=lambda: [0.0, 0.0, 0.0])
    initial_p: float | None = 0.0

    def save(self, filepath: str):
        """Serialise this configuration to a JSON file.

        Args:
            filepath: Path for the output JSON file.
        """
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, filepath: str):
        """Load a configuration from a JSON file.

        Args:
            filepath: Path to a JSON file produced by :meth:`save`.

        Returns:
            A new :class:`FVMConfig` instance.
        """
        with open(filepath) as f:
            data = json.load(f)
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown top-level FVMConfig field(s): {', '.join(unknown)}")

        # Manual reconstruction of nested dataclasses
        mesh_data = data.get("mesh", {})
        execution_data = data.get("execution", {})
        acceptance_data = data.get("acceptance", {})
        time_data = data.get("time", {})
        solver_data = data.get("solver", {})
        transport_data = data.get("transport", {})
        dynamic_mesh_data = data.get("dynamic_mesh", {"method": "static"})
        turbulence_data = data.get("turbulence", None)

        mesh = MeshConfig(**mesh_data)
        execution = ExecutionConfig(**execution_data)
        acceptance = RunAcceptancePolicy(**acceptance_data)
        time = TimeConfig(**time_data)
        dynamic_mesh = DynamicMeshConfig(**dynamic_mesh_data)

        # Dataclass defaults provide backward compatibility for fields absent
        # from old files.  Passing the complete dictionary preserves every
        # supported setting and rejects misspelled or obsolete keys.
        solver = SolverParams(**solver_data)
        transport = TransportConfig(**transport_data)
        boundaries = [BoundaryConfig(**b) for b in data.get("boundaries", [])]
        turbulence = TurbulenceConfig(**turbulence_data) if turbulence_data else None

        return cls(
            case_name=data["case_name"],
            mesh=mesh,
            execution=execution,
            acceptance=acceptance,
            time=time,
            solver=solver,
            transport=transport,
            dynamic_mesh=dynamic_mesh,
            boundaries=boundaries,
            turbulence=turbulence,
            initial_U=data.get("initial_U", [0.0, 0.0, 0.0]),
            initial_p=data.get("initial_p", 0.0),
        )
