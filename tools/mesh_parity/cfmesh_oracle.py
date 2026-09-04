"""Reproducible cfMesh case generation and native OpenONDA execution.

This is a development oracle only.  The native ``CartesianMesher`` is invoked
directly in-process; it never shells out to cfMesh.  Conversely, the cfMesh
case receives a generated multi-region STL containing the *same input surface
triangles* plus the requested outer-domain shell, because cartesianMesh defines
its external boundary through the input surface.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

import numpy as np

from .openfoam_poly_mesh import PolyMesh, read_poly_mesh, write_poly_mesh


class ParityConfigurationError(ValueError):
    """Raised when a parity specification cannot define identical inputs."""


class CfMeshUnavailableError(RuntimeError):
    """Raised when the local cfMesh executable is not available."""


class CfMeshExecutionError(RuntimeError):
    """Raised when cfMesh exits unsuccessfully or does not write a polyMesh."""


class OpenONDAUnsupportedForParityError(RuntimeError):
    """Raised instead of silently comparing a feature OpenONDA does not implement."""


_WORD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DOMAIN_KEYS = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _finite_positive(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ParityConfigurationError(f"{name} must be finite and positive")
    return result


def _bounds(value: object, name: str) -> tuple[float, float, float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 6:
        raise ParityConfigurationError(f"{name} must contain six coordinates")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ParityConfigurationError(f"{name} must contain finite coordinates")
    if not all(result[2 * axis] < result[2 * axis + 1] for axis in range(3)):
        raise ParityConfigurationError(f"{name} must have positive extent along every axis")
    return result  # type: ignore[return-value]


def _word(value: object, name: str) -> str:
    result = str(value)
    if not _WORD.fullmatch(result):
        raise ParityConfigurationError(
            f"{name} must be an OpenFOAM word (letters, digits, and underscores)"
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _foam_scalar(value: float) -> str:
    """Emit the shortest decimal that round-trips to the requested float.

    The renderer must preserve a user-visible request such as ``0.6667``;
    formatting every binary float to 17 significant digits would misleadingly
    make it look as if the harness had changed the value.
    """
    return repr(float(value))


@dataclass(frozen=True, slots=True)
class SurfaceSpec:
    """One authoritative closed STL and the patch it supplies."""

    path: Path
    patch: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        object.__setattr__(self, "patch", _word(self.patch, "surface.patch"))
        if not self.path.is_file():
            raise ParityConfigurationError(f"Surface STL does not exist: {self.path}")


@dataclass(frozen=True, slots=True)
class BoxRefinementSpec:
    """An axis-aligned box refinement that maps to cfMesh ``objectRefinements``."""

    name: str
    bounds: tuple[float, float, float, float, float, float]
    cell_size: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _word(self.name, "box refinement name"))
        object.__setattr__(self, "bounds", _bounds(self.bounds, f"{self.name}.bounds"))
        object.__setattr__(
            self, "cell_size", _finite_positive(self.cell_size, f"{self.name}.cell_size")
        )


@dataclass(frozen=True, slots=True)
class PatchRefinementSpec:
    """A cfMesh ``localRefinement`` request for one supplied surface patch."""

    patch: str
    cell_size: float
    refinement_thickness: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "patch", _word(self.patch, "patch refinement patch"))
        object.__setattr__(
            self, "cell_size", _finite_positive(self.cell_size, f"{self.patch}.cell_size")
        )
        if self.refinement_thickness is not None:
            object.__setattr__(
                self,
                "refinement_thickness",
                _finite_positive(self.refinement_thickness, f"{self.patch}.refinement_thickness"),
            )


@dataclass(frozen=True, slots=True)
class ParitySpec:
    """Fully explicit meshing inputs shared by the cfMesh and OpenONDA paths."""

    name: str
    domain_bounds: tuple[float, float, float, float, float, float]
    domain_patches: Mapping[str, str]
    surfaces: tuple[SurfaceSpec, ...]
    max_cell_size: float
    boundary_cell_size: float
    min_cell_size: float
    box_refinements: tuple[BoxRefinementSpec, ...] = ()
    patch_refinements: tuple[PatchRefinementSpec, ...] = ()
    patch_types: Mapping[str, str] | None = None
    surface_may_cross_domain_boundary: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _word(self.name, "case name"))
        object.__setattr__(self, "domain_bounds", _bounds(self.domain_bounds, "domain.bounds"))
        object.__setattr__(
            self, "max_cell_size", _finite_positive(self.max_cell_size, "max_cell_size")
        )
        object.__setattr__(
            self,
            "boundary_cell_size",
            _finite_positive(self.boundary_cell_size, "boundary_cell_size"),
        )
        object.__setattr__(
            self, "min_cell_size", _finite_positive(self.min_cell_size, "min_cell_size")
        )
        if self.boundary_cell_size > self.max_cell_size:
            raise ParityConfigurationError("boundary_cell_size must not exceed max_cell_size")
        if self.min_cell_size > self.boundary_cell_size:
            raise ParityConfigurationError("min_cell_size must not exceed boundary_cell_size")
        patches = {
            str(key): _word(value, f"domain.patches.{key}")
            for key, value in self.domain_patches.items()
        }
        if tuple(sorted(patches)) != tuple(sorted(_DOMAIN_KEYS)):
            raise ParityConfigurationError(
                "domain.patches must define xmin, xmax, ymin, ymax, zmin, and zmax"
            )
        object.__setattr__(self, "domain_patches", patches)
        surfaces = tuple(self.surfaces)
        if not surfaces:
            raise ParityConfigurationError("At least one STL surface is required")
        surface_patches = [surface.patch for surface in surfaces]
        if len(set(surface_patches)) != len(surface_patches):
            raise ParityConfigurationError("Surface patch names must be unique")
        if set(surface_patches) & set(patches.values()):
            raise ParityConfigurationError("Surface and outer-domain patch names must be distinct")
        object.__setattr__(self, "surfaces", surfaces)
        boxes = tuple(self.box_refinements)
        if len({box.name for box in boxes}) != len(boxes):
            raise ParityConfigurationError("Box refinement names must be unique")
        for box in boxes:
            if not all(
                self.domain_bounds[2 * axis] <= box.bounds[2 * axis]
                and box.bounds[2 * axis + 1] <= self.domain_bounds[2 * axis + 1]
                for axis in range(3)
            ):
                raise ParityConfigurationError(
                    f"Box refinement {box.name!r} lies outside the domain"
                )
        object.__setattr__(self, "box_refinements", boxes)
        local = tuple(self.patch_refinements)
        if len({item.patch for item in local}) != len(local):
            raise ParityConfigurationError("Patch refinements must target unique patches")
        unknown = {item.patch for item in local} - set(surface_patches)
        if unknown:
            raise ParityConfigurationError(
                "Patch refinement must target an input surface patch, not an outer box patch: "
                + ", ".join(sorted(unknown))
            )
        object.__setattr__(self, "patch_refinements", local)
        if not isinstance(self.surface_may_cross_domain_boundary, bool):
            raise ParityConfigurationError("surface_may_cross_domain_boundary must be boolean")
        if self.surface_may_cross_domain_boundary:
            raise ParityConfigurationError(
                "A parity case must use a closed surface strictly inside its outer domain. "
                "Clip/cap and freeze a normalized STL before comparing it with cfMesh."
            )
        merged_types = dict.fromkeys(set(patches.values()), "patch")
        merged_types.update({surface.patch: "wall" for surface in surfaces})
        for name, patch_type in (self.patch_types or {}).items():
            name = _word(name, "patch_types key")
            if name not in merged_types:
                raise ParityConfigurationError(f"patch_types names unknown patch {name!r}")
            merged_types[name] = _word(patch_type, f"patch_types.{name}")
        object.__setattr__(self, "patch_types", merged_types)

    @property
    def all_patch_names(self) -> tuple[str, ...]:
        """All unique supplied patch labels in canonical order."""
        return tuple(
            sorted(set(self.domain_patches.values()) | {item.patch for item in self.surfaces})
        )

    def effective_config(self) -> dict[str, Any]:
        """Return every behavior-affecting input in portable JSON form."""
        return {
            "name": self.name,
            "domain": {"bounds": list(self.domain_bounds), "patches": dict(self.domain_patches)},
            "surfaces": [
                {"path": str(surface.path), "patch": surface.patch, "sha256": _sha256(surface.path)}
                for surface in self.surfaces
            ],
            "max_cell_size": self.max_cell_size,
            "boundary_cell_size": self.boundary_cell_size,
            "min_cell_size": self.min_cell_size,
            "box_refinements": [asdict(item) for item in self.box_refinements],
            "patch_refinements": [asdict(item) for item in self.patch_refinements],
            "patch_types": dict(self.patch_types or {}),
            "surface_may_cross_domain_boundary": self.surface_may_cross_domain_boundary,
        }


def load_parity_spec(path: Path | str) -> ParitySpec:
    """Load an explicit parity specification from JSON, resolving STL paths."""
    path = Path(path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ParityConfigurationError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ParityConfigurationError("Parity specification root must be an object")
    try:
        domain = raw["domain"]
        surfaces = raw["surfaces"]
    except KeyError as error:
        raise ParityConfigurationError(f"Parity specification lacks {error.args[0]!r}") from error
    if not isinstance(domain, dict) or not isinstance(surfaces, list):
        raise ParityConfigurationError("domain must be an object and surfaces must be a list")
    surface_specs = tuple(
        SurfaceSpec(path=(path.parent / item["path"]).resolve(), patch=item["patch"])
        for item in surfaces
    )
    boxes: list[BoxRefinementSpec] = []
    patches: list[PatchRefinementSpec] = []
    for item in raw.get("refinements", []):
        if not isinstance(item, dict):
            raise ParityConfigurationError("Every refinement must be an object")
        kind = item.get("type")
        if kind == "box":
            boxes.append(
                BoxRefinementSpec(
                    item["name"], _bounds(item["bounds"], "box.bounds"), item["cell_size"]
                )
            )
        elif kind == "patch":
            patches.append(
                PatchRefinementSpec(
                    item["patch"], item["cell_size"], item.get("refinement_thickness")
                )
            )
        else:
            raise ParityConfigurationError(
                f"Unsupported refinement type {kind!r}; use 'box' or 'patch'"
            )
    return ParitySpec(
        name=raw.get("name", path.stem),
        domain_bounds=_bounds(domain["bounds"], "domain.bounds"),
        domain_patches=domain["patches"],
        surfaces=surface_specs,
        max_cell_size=raw["max_cell_size"],
        boundary_cell_size=raw["boundary_cell_size"],
        min_cell_size=raw["min_cell_size"],
        box_refinements=tuple(boxes),
        patch_refinements=tuple(patches),
        patch_types=raw.get("patch_types"),
        surface_may_cross_domain_boundary=raw.get("surface_may_cross_domain_boundary", False),
    )


def _foam_header(object_name: str, location: str) -> str:
    return (
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        "    class       dictionary;\n"
        f'    location    "{location}";\n'
        f"    object      {object_name};\n"
        "}\n\n"
    )


def render_control_dict() -> str:
    """Render the ASCII-only control dictionary used for reproducible oracle files."""
    return _foam_header("controlDict", "system") + (
        "application     cartesianMesh;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        "endTime         1;\n"
        "deltaT          1;\n"
        "writeControl    timeStep;\n"
        "writeInterval   1;\n"
        "writeFormat     ascii;\n"
        "writePrecision  15;\n"
        "timeFormat      general;\n"
        "timePrecision   15;\n"
        "runTimeModifiable false;\n"
    )


def render_mesh_dict(spec: ParitySpec, *, geometry_name: str, stop_after: str | None = None) -> str:
    """Render the cfMesh subset corresponding exactly to :class:`ParitySpec`.

    The box syntax follows cfMesh's documented ``objectRefinements`` contract:
    a centre plus the three axis-aligned lengths.  ``localRefinement`` maps
    patch controls one-to-one.  No mesh-size fitting or hidden refinement is
    introduced here.
    """
    lines = [
        _foam_header("meshDict", "system").rstrip(),
        f'surfaceFile "{geometry_name}";',
        f"maxCellSize {_foam_scalar(spec.max_cell_size)};",
        f"boundaryCellSize {_foam_scalar(spec.boundary_cell_size)};",
        f"minCellSize {_foam_scalar(spec.min_cell_size)};",
        "",
    ]
    if spec.patch_refinements:
        lines.extend(("localRefinement", "{"))
        for refinement in spec.patch_refinements:
            lines.extend(
                (
                    f"    {refinement.patch}",
                    "    {",
                    f"        cellSize {_foam_scalar(refinement.cell_size)};",
                    *(
                        (
                            "        refinementThickness "
                            f"{_foam_scalar(refinement.refinement_thickness)};",
                        )
                        if refinement.refinement_thickness is not None
                        else ()
                    ),
                    "    }",
                )
            )
        lines.extend(("}", ""))
    if spec.box_refinements:
        lines.extend(("objectRefinements", "{"))
        for refinement in spec.box_refinements:
            lower = np.asarray(refinement.bounds[::2], dtype=np.float64)
            upper = np.asarray(refinement.bounds[1::2], dtype=np.float64)
            centre = 0.5 * (lower + upper)
            lengths = upper - lower
            lines.extend(
                (
                    f"    {refinement.name}",
                    "    {",
                    "        type box;",
                    f"        cellSize {_foam_scalar(refinement.cell_size)};",
                    "        centre ("
                    f"{_foam_scalar(centre[0])} {_foam_scalar(centre[1])} "
                    f"{_foam_scalar(centre[2])});",
                    f"        lengthX {_foam_scalar(lengths[0])};",
                    f"        lengthY {_foam_scalar(lengths[1])};",
                    f"        lengthZ {_foam_scalar(lengths[2])};",
                    "    }",
                )
            )
        lines.extend(("}", ""))
    lines.extend(("renameBoundary", "{", "    newPatchNames", "    {"))
    for patch_name in spec.all_patch_names:
        lines.extend(
            (
                f"        {patch_name}",
                "        {",
                f"            newName {patch_name};",
                f"            type {spec.patch_types[patch_name]};",
                "        }",
            )
        )
    lines.extend(("    }", "}", ""))
    if stop_after is not None:
        _word(stop_after, "workflowControls.stopAfter")
        lines.extend(("workflowControls", "{", f"    stopAfter {stop_after};", "}", ""))
    return "\n".join(lines)


def _box_patch_triangles(
    bounds: tuple[float, float, float, float, float, float], patches: Mapping[str, str]
) -> dict[str, list[np.ndarray]]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    corners = {
        "000": np.array((xmin, ymin, zmin)),
        "001": np.array((xmin, ymin, zmax)),
        "010": np.array((xmin, ymax, zmin)),
        "011": np.array((xmin, ymax, zmax)),
        "100": np.array((xmax, ymin, zmin)),
        "101": np.array((xmax, ymin, zmax)),
        "110": np.array((xmax, ymax, zmin)),
        "111": np.array((xmax, ymax, zmax)),
    }
    outward_quads = {
        "xmin": ("000", "001", "011", "010"),
        "xmax": ("100", "110", "111", "101"),
        "ymin": ("000", "100", "101", "001"),
        "ymax": ("010", "011", "111", "110"),
        "zmin": ("000", "010", "110", "100"),
        "zmax": ("001", "101", "111", "011"),
    }
    result: dict[str, list[np.ndarray]] = defaultdict(list)
    for side, corner_ids in outward_quads.items():
        values = [corners[item] for item in corner_ids]
        result[patches[side]].extend(
            (
                np.asarray((values[0], values[1], values[2])),
                np.asarray((values[0], values[2], values[3])),
            )
        )
    return result


def _surface_triangles(spec: ParitySpec) -> dict[str, np.ndarray]:
    """Load authority triangles with the same strict validation as OpenONDA."""
    from source.solvers.fvm.mesh.cartesian.surface import load_surface

    result: dict[str, np.ndarray] = {}
    for surface in spec.surfaces:
        triangles = np.asarray(load_surface(surface.path).triangles, dtype=np.float64)
        lower = triangles.min(axis=(0, 1))
        upper = triangles.max(axis=(0, 1))
        if not all(
            spec.domain_bounds[2 * axis] < lower[axis]
            and upper[axis] < spec.domain_bounds[2 * axis + 1]
            for axis in range(3)
        ):
            raise ParityConfigurationError(
                f"Surface {surface.patch!r} must lie strictly within the outer domain for a "
                "same-input cfMesh comparison"
            )
        result[surface.patch] = triangles
    return result


def _write_ascii_stl(groups: Mapping[str, Sequence[np.ndarray]], path: Path) -> None:
    """Write multi-region triangles, preserving every input coordinate exactly."""
    lines: list[str] = []
    for name in sorted(groups):
        lines.append(f"solid {name}")
        for triangle in groups[name]:
            values = np.asarray(triangle, dtype=np.float64)
            normal = np.cross(values[1] - values[0], values[2] - values[0])
            length = float(np.linalg.norm(normal))
            normal = normal / length if length else np.zeros(3)
            lines.append(f"  facet normal {normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}")
            lines.append("    outer loop")
            lines.extend(
                f"      vertex {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}" for point in values
            )
            lines.extend(("    endloop", "  endfacet"))
        lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_cfmesh_case(
    spec: ParitySpec, directory: Path | str, *, stop_after: str | None = None
) -> tuple[Path, dict[str, np.ndarray]]:
    """Create a self-contained ASCII cfMesh case and return surface authority arrays."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty cfMesh case directory: {directory}")
    (directory / "system").mkdir(parents=True, exist_ok=True)
    tri_surface = directory / "constant" / "triSurface"
    tri_surface.mkdir(parents=True, exist_ok=True)
    triangles = _surface_triangles(spec)
    groups: dict[str, list[np.ndarray]] = _box_patch_triangles(
        spec.domain_bounds, spec.domain_patches
    )
    for patch, values in triangles.items():
        groups[patch] = list(values)
    geometry_name = "openonda_parity_geometry.stl"
    geometry_path = tri_surface / geometry_name
    _write_ascii_stl(groups, geometry_path)
    (directory / "system" / "controlDict").write_text(render_control_dict(), encoding="ascii")
    (directory / "system" / "meshDict").write_text(
        render_mesh_dict(
            spec,
            geometry_name=f"constant/triSurface/{geometry_name}",
            stop_after=stop_after,
        ),
        encoding="ascii",
    )
    (directory / "openonda_effective_config.json").write_text(
        json.dumps(spec.effective_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return geometry_path, triangles


def discover_cartesian_mesh(executable: Path | str | None = None) -> Path | None:
    """Return an explicit, environment, or PATH cfMesh binary without guessing roots."""
    candidate = (
        executable or os.environ.get("CFMESH_CARTESIAN_MESH") or shutil.which("cartesianMesh")
    )
    if candidate is None:
        return None
    path = Path(candidate).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise CfMeshUnavailableError(f"cartesianMesh executable is not executable: {path}")
    return path


def discover_cfmesh_launcher(launcher: Path | str | None = None) -> Path | None:
    """Return an optional environment launcher used before ``cartesianMesh``.

    Native OpenFOAM.app installations on macOS keep their shared libraries on
    a mounted read-only volume.  Their versioned ``openfoam`` launcher mounts
    that volume and supplies the required runtime environment.  Linux builds
    normally need no launcher at all.
    """
    candidate = launcher or os.environ.get("CFMESH_LAUNCHER")
    if candidate is None:
        return None
    path = Path(candidate).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise CfMeshUnavailableError(f"cfMesh launcher is not executable: {path}")
    return path


def _cfmesh_command(executable: Path, launcher: Path | None = None) -> list[str]:
    command = [str(executable)]
    if launcher is not None:
        command.insert(0, str(launcher))
    return command


def executable_metadata(executable: Path, launcher: Path | None = None) -> dict[str, Any]:
    """Capture a stable executable hash plus best-effort version/help output."""
    command = _cfmesh_command(executable, launcher)
    try:
        help_run = subprocess.run(
            [*command, "-help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        version_text = (help_run.stdout + help_run.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        version_text = f"unavailable: {error}"
    metadata: dict[str, Any] = {
        "path": str(executable),
        "sha256": _sha256(executable),
        "version_or_help": version_text,
    }
    if launcher is not None:
        metadata["launcher"] = {
            "path": str(launcher),
            "sha256": _sha256(launcher),
        }
    return metadata


@dataclass(frozen=True, slots=True)
class CfMeshRun:
    directory: Path
    poly_mesh_directory: Path
    mesh: PolyMesh
    log_path: Path
    metadata: Mapping[str, Any]


def run_cfmesh(
    spec: ParitySpec,
    directory: Path | str,
    *,
    executable: Path | str | None = None,
    launcher: Path | str | None = None,
    stop_after: str | None = None,
    timeout_seconds: float = 3_600.0,
) -> tuple[CfMeshRun, dict[str, np.ndarray]]:
    """Generate and run an isolated cfMesh case, retaining its complete log."""
    binary = discover_cartesian_mesh(executable)
    if binary is None:
        raise CfMeshUnavailableError(
            "cartesianMesh was not found. Set CFMESH_CARTESIAN_MESH or pass --cfmesh-executable."
        )
    environment_launcher = discover_cfmesh_launcher(launcher)
    command = _cfmesh_command(binary, environment_launcher)
    directory = Path(directory)
    _geometry, triangles = write_cfmesh_case(spec, directory, stop_after=stop_after)
    log_path = directory / "cartesianMesh.log"
    environment = os.environ.copy()
    environment["FOAM_CASE"] = str(directory)
    # cfMesh writes shared surface vertices from OpenMP loops. At an exact
    # geometric tie (for example an aligned cube edge), different schedules
    # can select different but equally close surface faces. A differential
    # oracle needs a reproducible reference, so use one thread unless the
    # dedicated parity override explicitly requests another count.
    omp_num_threads = os.environ.get("CFMESH_PARITY_OMP_NUM_THREADS", "1")
    environment["OMP_NUM_THREADS"] = omp_num_threads
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            run = subprocess.run(
                command,
                cwd=directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired as error:
        raise CfMeshExecutionError(
            f"cfMesh exceeded {timeout_seconds:g} seconds; complete log: {log_path}"
        ) from error
    elapsed = time.monotonic() - started
    if run.returncode != 0:
        raise CfMeshExecutionError(
            f"cfMesh exited {run.returncode}; inspect complete log at {log_path}"
        )
    candidates = (directory / "constant" / "polyMesh", directory / "0" / "polyMesh")
    poly_mesh_directory = next((item for item in candidates if (item / "points").is_file()), None)
    if poly_mesh_directory is None:
        stage_note = f" after stopAfter {stop_after!r}" if stop_after else ""
        raise CfMeshExecutionError(
            f"cfMesh wrote no readable polyMesh{stage_note}; inspect complete log at {log_path}"
        )
    mesh = read_poly_mesh(poly_mesh_directory)
    metadata = {
        "executable": executable_metadata(binary, environment_launcher),
        "command": command,
        "return_code": run.returncode,
        "elapsed_seconds": elapsed,
        "environment": {"OMP_NUM_THREADS": omp_num_threads},
        "stop_after": stop_after,
        "generated_surface_sha256": _sha256(
            directory / "constant" / "triSurface" / "openonda_parity_geometry.stl"
        ),
    }
    return CfMeshRun(directory, poly_mesh_directory, mesh, log_path, metadata), triangles


@dataclass(frozen=True, slots=True)
class OpenONDARun:
    directory: Path
    poly_mesh_directory: Path
    mesh: PolyMesh
    metadata: Mapping[str, Any]


def run_openonda(
    spec: ParitySpec, directory: Path | str, *, stop_after: str | None = None
) -> OpenONDARun:
    """Build OpenONDA natively and persist the comparable ASCII polyMesh."""
    if spec.patch_refinements:
        names = ", ".join(item.patch for item in spec.patch_refinements)
        raise OpenONDAUnsupportedForParityError(
            "OpenONDA has no patch-refinement control yet, so a same-input parity run cannot "
            f"continue for: {names}"
        )
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty OpenONDA run directory: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    import openonda.fvm.mesher as msh

    domain = msh.BoxDomain(
        bounds=spec.domain_bounds,
        patches=msh.BoxPatches(**dict(spec.domain_patches)),
    )
    surfaces = tuple(msh.STLSurface(surface.path, patch=surface.patch) for surface in spec.surfaces)
    refinements = tuple(
        msh.BoxRefinement(item.name, item.bounds, item.cell_size) for item in spec.box_refinements
    )
    mesher = msh.CartesianMesher(
        domain=domain,
        surfaces=surfaces,
        max_cell_size=spec.max_cell_size,
        boundary_cell_size=spec.boundary_cell_size,
        min_cell_size=spec.min_cell_size,
        refinements=refinements,
        surface_may_cross_domain_boundary=spec.surface_may_cross_domain_boundary,
    )
    started = time.monotonic()
    if stop_after is None:
        mesh_data = mesher.build()
    else:
        # The parity runner deliberately does not substitute the final mesh for
        # an intermediate cfMesh checkpoint.  A future native stage API must
        # opt in explicitly, otherwise this remains a visible upstream gate.
        if stop_after not in getattr(mesher, "supported_workflow_stages", ()):
            raise OpenONDAUnsupportedForParityError(
                f"OpenONDA cannot expose the cfMesh checkpoint {stop_after!r} yet"
            )
        mesh_data = mesher.build(stop_after=stop_after)
        if stop_after == "boundaryLayerRefinement":
            # This is cfMesh's final workflow step, so execution continues
            # through replaceBoundaries() instead of saving a temporary mesh.
            # Apply the same explicit patch-type dictionary to the native
            # checkpoint; earlier intermediate stages deliberately stay empty.
            patch_types = dict(spec.patch_types or {})
            for patch in mesh_data["boundary"]:
                patch["type"] = patch_types[str(patch["name"])]
    elapsed = time.monotonic() - started
    mesh = PolyMesh.from_openonda(mesh_data)
    poly_mesh_directory = write_poly_mesh(mesh, directory / "constant" / "polyMesh")
    report = mesher.report.as_dict() if mesher.report is not None else None
    (directory / "openonda_generation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return OpenONDARun(
        directory=directory,
        poly_mesh_directory=poly_mesh_directory,
        mesh=mesh,
        metadata={"elapsed_seconds": elapsed, "report": report, "stop_after": stop_after},
    )


__all__ = [
    "BoxRefinementSpec",
    "CfMeshExecutionError",
    "CfMeshRun",
    "CfMeshUnavailableError",
    "OpenONDARun",
    "OpenONDAUnsupportedForParityError",
    "ParityConfigurationError",
    "ParitySpec",
    "PatchRefinementSpec",
    "SurfaceSpec",
    "discover_cartesian_mesh",
    "discover_cfmesh_launcher",
    "executable_metadata",
    "load_parity_spec",
    "render_control_dict",
    "render_mesh_dict",
    "run_cfmesh",
    "run_openonda",
    "write_cfmesh_case",
]
