import contextlib
from dataclasses import asdict, dataclass, field
import json
import os
import re
from typing import Literal


def _collect_field_patches(filepath: str, type_key: str, value_key: str, patches: dict) -> None:
    """Read a single OpenFOAM field file and merge its BC entries into `patches`."""
    from ..fields.field_io import parse_field_boundary_patches

    if not os.path.exists(filepath):
        return
    for k, v in parse_field_boundary_patches(filepath).items():
        patches.setdefault(k, {})[type_key] = v.get("type")
        patches.setdefault(k, {})[value_key] = v.get("value")


def _build_boundary_config(name: str, vals: dict) -> "BoundaryConfig":
    """Construct a BoundaryConfig from merged patch data for one patch name."""
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
    """Resolve path to the OpenFOAM `system` directory."""
    if os.path.isdir(path) and os.path.basename(path) != "system":
        return os.path.join(path, "system")
    return path if os.path.isdir(path) else os.path.dirname(path)


def _parse_yplus_patches(content: str) -> list[str] | None:
    """Extract yplus patch list from the functions block of a controlDict text."""
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
    """Parse a numeric value from an OpenFOAM dictionary entry (scalar, list, or string)."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list) and v:
        return float(v[-1])
    if isinstance(v, str):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)
        if nums:
            return float(nums[-1])
    return None


def _detect_turbulence_model(data: dict, filepath: str) -> str:
    """Determine the turbulence model name from parsed data and file text."""
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
    """Return (Cs, dynamic) turbulence coefficients, falling back to regex on the file."""
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

    @staticmethod
    def inlet(name: str, velocity: list[float]) -> "BoundaryConfig":
        return BoundaryConfig(
            name=name, type_U="fixedValue", value_U=velocity, type_p="zeroGradient"
        )

    @staticmethod
    def outlet(name: str, p: float = 0.0) -> "BoundaryConfig":
        return BoundaryConfig(name=name, type_U="zeroGradient", type_p="fixedValue", value_p=p)

    @staticmethod
    def freestream(name: str, velocity: list[float]) -> "BoundaryConfig":
        return BoundaryConfig(
            name=name, type_U="fixedValue", value_U=velocity, type_p="zeroGradient"
        )

    @staticmethod
    def wall(name: str) -> "BoundaryConfig":
        return BoundaryConfig(
            name=name,
            type_U="fixedValue",
            value_U=[0.0, 0.0, 0.0],
            type_p="zeroGradient",
            type_nut="calculated",
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
        )

    @staticmethod
    def load_from_time_dir(case_dir: str, time_dir: str = "0") -> list["BoundaryConfig"]:
        """Load boundary conditions from OpenFOAM time directory (e.g., '0').

        This reads `U`, `p`, and `nut` if present and constructs a list of
        BoundaryConfig objects for all patches found in the files.
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

    @staticmethod
    def block_mesh() -> "MeshConfig":
        return MeshConfig(method="blockMesh")

    @staticmethod
    def cartesian(surface_file: str, max_cell_size: float) -> "MeshConfig":
        return MeshConfig(
            method="cartesianMesh", surface_file=surface_file, max_cell_size=max_cell_size
        )

    def generate_mesh_dict(self) -> str:
        """Generate content for system/meshDict (for cartesianMesh)."""
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
        return TimeConfig(delta_t=1, start_time=0, end_time=max_iter, write_interval=write_interval)

    @staticmethod
    def transient(dt: float, duration: float, write_interval: int = 10) -> "TimeConfig":
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

    algorithm: Literal["SIMPLE", "PIMPLE"] = "PIMPLE"
    n_correctors: int = 2
    n_outer_correctors: int = 1
    max_iter: int = 20
    tolerance: float = 1e-6
    alpha_u: float = 0.7
    alpha_p: float = 0.3
    linear_solver: str = "bicgstab"  # Defaulting to iterative for performance
    reuse_ilu: bool = True  # Enable ILU preconditioning reuse by default

    # y+ patches: user can specify patches to compute y+ for (None => auto-select wall patches)
    yplus_patches: list[str] | None = None

    # Momentum solver tunables
    momentum_tol: float = 1e-4
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
        linear_solver: str = "spsolve",
        alpha_u: float = 1.0,
        alpha_p: float = 1.0,
        convection_scheme: str = "deferred",
        gradient_scheme: str = "lsq",
        time_scheme: str = "euler_implicit",
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
            linear_solver=linear_solver,
            alpha_u=alpha_u,
            alpha_p=alpha_p,
            convection_scheme=convection_scheme,
            gradient_scheme=gradient_scheme,
            time_scheme=time_scheme,
        )

    @staticmethod
    def simple(
        alpha_u: float = 0.7,
        alpha_p: float = 0.3,
        linear_solver: str = "spsolve",
        gradient_scheme: str = "lsq",
    ) -> "SolverParams":
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
        params = cls()
        if os.path.exists(control):
            data = parse_simple_dictionary(control)
            dt = data.get("deltaT")
            if dt is not None:
                params.time_scheme = "euler_implicit" if float(dt) > 0 else params.time_scheme
            try:
                with open(control) as f:
                    content = f.read()
                patches = _parse_yplus_patches(content)
                if patches:
                    params.yplus_patches = patches
            except Exception:
                pass
        if os.path.exists(fvsolution):
            data = parse_simple_dictionary(fvsolution)
            if "linearSolver" in data:
                params.linear_solver = data.get("linearSolver")
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
        return TransportConfig(density=1.225, nu=1.5e-5)

    @staticmethod
    def water() -> "TransportConfig":
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
        rho_val = _extract_foam_number(data.get("rho") or data.get("Rho")) or cls().density
        nu_raw = data.get("nu")
        nu_val = _extract_foam_number(nu_raw) if nu_raw is not None else None
        if nu_val is None:
            mu_val = _extract_foam_number(data.get("mu"))
            if mu_val is not None:
                with contextlib.suppress(Exception):
                    nu_val = float(mu_val) / float(rho_val)
        return cls(density=float(rho_val), nu=float(nu_val if nu_val is not None else cls().nu))


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
        return DynamicMeshConfig(method="static")

    @staticmethod
    def rigid(
        velocity: list[float] | None = None,
        omega: float = 0,
        axis: list[float] | None = None,
        origin: list[float] | None = None,
    ) -> "DynamicMeshConfig":
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
        return TurbulenceConfig(model="Smagorinsky", Cs=Cs, dynamic=dynamic)

    @staticmethod
    def wale(Cw: float = 0.325) -> "TurbulenceConfig":
        """Wall-adapting WALE model (recommended for wall-bounded LES)."""
        return TurbulenceConfig(model="WALE", Cs=Cw)

    @staticmethod
    def sigma(Csigma: float = 1.35) -> "TurbulenceConfig":
        """sigma model — zero ν_t in 2D / pure-shear / solid-rotation regions."""
        return TurbulenceConfig(model="sigma", Cs=Csigma)

    @staticmethod
    def dynamic_smagorinsky() -> "TurbulenceConfig":
        """Germano–Lilly dynamic Smagorinsky (globally averaged coefficient)."""
        return TurbulenceConfig(model="dynamicSmagorinsky", dynamic=True)

    @staticmethod
    def none() -> "TurbulenceConfig":
        """No subgrid model (ILES / DNS); pair with a low-dissipation scheme."""
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
class FVMConfig:
    """
    Top-level FVM configuration.
    """

    case_name: str
    mesh: MeshConfig = field(default_factory=MeshConfig.block_mesh)
    time: TimeConfig = field(default_factory=TimeConfig)
    solver: SolverParams = field(default_factory=SolverParams)
    transport: TransportConfig = field(default_factory=TransportConfig)
    dynamic_mesh: DynamicMeshConfig = field(default_factory=DynamicMeshConfig.static)
    boundaries: list[BoundaryConfig] = field(default_factory=list)
    turbulence: TurbulenceConfig | None = None
    initial_U: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    initial_p: float = 0.0

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath) as f:
            data = json.load(f)

        # Manual reconstruction of nested dataclasses
        mesh_data = data.get("mesh", {})
        time_data = data.get("time", {})
        solver_data = data.get("solver", {})
        transport_data = data.get("transport", {})
        dynamic_mesh_data = data.get("dynamic_mesh", {"method": "static"})
        turbulence_data = data.get("turbulence", None)

        mesh = MeshConfig(**mesh_data)
        time = TimeConfig(**time_data)
        dynamic_mesh = DynamicMeshConfig(**dynamic_mesh_data)

        # Pull extra fields that might not be in older saved configs
        solver_obj_data = {
            "algorithm": solver_data.get("algorithm", "PIMPLE"),
            "n_correctors": solver_data.get("n_correctors", 2),
            "n_outer_correctors": solver_data.get("n_outer_correctors", 1),
            "max_iter": solver_data.get("max_iter", 20),
            "tolerance": solver_data.get("tolerance", 1e-6),
            "alpha_u": solver_data.get("alpha_u", 0.7),
            "alpha_p": solver_data.get("alpha_p", 0.3),
            "linear_solver": solver_data.get("linear_solver", "spsolve"),
            "convection_scheme": solver_data.get("convection_scheme", "limitedLinear"),
            "time_scheme": solver_data.get("time_scheme", "euler_implicit"),
        }
        solver = SolverParams(**solver_obj_data)
        transport = TransportConfig(**transport_data)
        boundaries = [BoundaryConfig(**b) for b in data.get("boundaries", [])]
        turbulence = TurbulenceConfig(**turbulence_data) if turbulence_data else None

        return cls(
            case_name=data["case_name"],
            mesh=mesh,
            time=time,
            solver=solver,
            transport=transport,
            dynamic_mesh=dynamic_mesh,
            boundaries=boundaries,
            turbulence=turbulence,
            initial_U=data.get("initial_U", [0.0, 0.0, 0.0]),
            initial_p=data.get("initial_p", 0.0),
        )
