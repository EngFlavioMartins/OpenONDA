"""
OpenVSP DegenGeom import helpers for OpenONDA VLM surfaces.

The VLM solver operates on OpenONDA's Aircraft/Wing/WingSegment primitives.
This module keeps OpenVSP optional by using DegenGeom CSV as the runtime-free
exchange format and only importing the OpenVSP Python API inside the .vsp3
export path.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from .aircraft import Aircraft, Wing, WingSegment
from .surface_io import save_surface


@dataclass
class OpenVSPImportConfig:
    """Configuration for importing OpenVSP DegenGeom data into OpenONDA VLM surfaces.

    Controls which components are imported, how they are discretised, and
    how coordinate transformations are applied during the conversion.

    Fields
    ------
    set_id
        OpenVSP set to import (default: ``'ALL'``).
    include_components
        Only import components whose name matches one of these substrings
        (``None`` = all components).
    exclude_components
        Skip components whose name matches one of these substrings.
    target_surface_types
        Surface-type(s) to import (default: wing, prop, rotor).
    panels_chord
        Override the number of chordwise panels (``None`` = use VSP settings).
    panels_span
        Override the number of spanwise panels (``None`` = use VSP settings).
    preserve_vsp_paneling
        Use VSP-native panel distribution when no chord/span override is
        given (default: ``True``).
    symmetry
        Symmetry-plane treatment (default: ``'from_openvsp'``).
    length_scale
        Factor applied to all coordinates (default: ``1.0``).
    coordinate_transform
        Affine transformation matrix (4×4) applied to imported node
        coordinates (``None`` = identity).
    strict
        Raise on unrecognised surface types (default: ``True``).

    Examples
    --------
    .. code-block:: python

        # Default import (all components, VSP-native panelling)
        cfg = OpenVSPImportConfig()

        # Import only wings with a custom 20×10 panel distribution
        cfg = OpenVSPImportConfig(
            include_components=["wing"],
            panels_chord=20,
            panels_span=10,
            preserve_vsp_paneling=False,
        )
    """

    set_id: int | str = "ALL"
    include_components: list[str] | None = None
    exclude_components: list[str] | None = None
    target_surface_types: tuple[str, ...] = ("wing", "prop", "rotor")
    panels_chord: int | None = None
    panels_span: int | None = None
    preserve_vsp_paneling: bool = True
    symmetry: str = "from_openvsp"
    length_scale: float = 1.0
    coordinate_transform: np.ndarray | None = None
    strict: bool = True


@dataclass
class _PointTable:
    name: str
    surface_type: str
    section: str
    points: np.ndarray
    u: np.ndarray | None = None
    w: np.ndarray | None = None
    dims: tuple[int, int] | None = None


_POINT_SECTIONS = {
    "plate",
    "surface",
    "lifting_surface",
    "vspaero_surface",
}
_NON_SURFACE_SECTIONS = {
    "surface_node",
    "surface_face",
    "stick_node",
    "stick_face",
    "point",
    "disk",
}


def export_openvsp_degengeom(
    vsp_file: str | Path,
    output_csv: str | Path,
    config: OpenVSPImportConfig | None = None,
) -> Path:
    """
    Export an OpenVSP ``.vsp3`` model to a DegenGeom CSV file.

    Uses the optional ``openvsp`` Python API when available.  Falls back to
    launching a subprocess with a system-installed OpenVSP Python interpreter.
    When neither is available, the user must export the CSV manually from
    OpenVSP and call :func:`load_degengeom_csv` directly.

    Parameters
    ----------
    vsp_file : str | Path
        Path to the ``.vsp3`` input file.
    output_csv : str | Path
        Path for the generated DegenGeom CSV file.
    config : OpenVSPImportConfig | None
        Import configuration (see :class:`OpenVSPImportConfig`).

    Returns
    -------
    Path
        Path to the generated CSV file.

    Raises
    ------
    FileNotFoundError
        If the ``.vsp3`` file does not exist.
    RuntimeError
        If the export completes but no CSV file is produced.
    ImportError
        If OpenVSP Python API is unavailable and no subprocess fallback exists.

    Examples
    --------
    >>> csv_path = export_openvsp_degengeom("model.vsp3", "degen.csv")
    >>> csv_path.exists()
    True
    """

    cfg = config or OpenVSPImportConfig()
    vsp_path = Path(vsp_file)
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not vsp_path.exists():
        raise FileNotFoundError(f"OpenVSP file not found: {vsp_path}")

    try:
        vsp = _import_openvsp()
    except ImportError:
        return _export_openvsp_degengeom_external(vsp_path, csv_path, cfg)

    _call_if_available(vsp, "ClearVSPModel")
    vsp.ReadVSPFile(str(vsp_path))
    _call_if_available(vsp, "Update")

    if all(
        hasattr(vsp, name)
        for name in ("SetComputationFileName", "ComputeDegenGeom", "DEGEN_GEOM_CSV_TYPE")
    ):
        set_id = getattr(vsp, "SET_ALL", 0) if cfg.set_id == "ALL" else int(cfg.set_id)
        vsp.SetComputationFileName(vsp.DEGEN_GEOM_CSV_TYPE, str(csv_path))
        vsp.ComputeDegenGeom(set_id, vsp.DEGEN_GEOM_CSV_TYPE)
        if csv_path.exists():
            return csv_path

    analysis = "DegenGeom"
    if hasattr(vsp, "SetAnalysisInputDefaults"):
        vsp.SetAnalysisInputDefaults(analysis)

    if cfg.set_id != "ALL":
        _try_set_analysis_input(vsp, analysis, "GeomSet", int(cfg.set_id))
        _try_set_analysis_input(vsp, analysis, "Set", int(cfg.set_id))

    _try_set_analysis_input(vsp, analysis, "WriteCSVFlag", 1)
    _try_set_analysis_input(vsp, analysis, "WriteMFileFlag", 0)
    _try_set_analysis_input(vsp, analysis, "WriteTXTFlag", 0)
    for key in ("DegenGeomCSVFile", "CSVFileName", "FileName"):
        _try_set_analysis_input(vsp, analysis, key, str(csv_path))

    vsp.ExecAnalysis(analysis)
    if not csv_path.exists():
        raise RuntimeError(
            "OpenVSP DegenGeom export completed but did not create the requested "
            f"CSV file: {csv_path}. Check the installed OpenVSP API version and "
            "export DegenGeom CSV manually if necessary."
        )
    return csv_path


def load_degengeom_csv(
    csv_file: str | Path,
    config: OpenVSPImportConfig | None = None,
) -> Aircraft:
    """Parse an OpenVSP DegenGeom CSV file into an OpenONDA
    :class:`~.aircraft.Aircraft`.

    Parameters
    ----------
    csv_file : str | Path
        Path to the DegenGeom CSV file.
    config : OpenVSPImportConfig | None
        Import configuration (see :class:`OpenVSPImportConfig`).  When
        ``None``, defaults are used.

    Returns
    -------
    Aircraft
        Populated aircraft model with wings and segments derived from the
        CSV lifting-surface tables.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If no lifting-surface point tables are found.

    Examples
    --------
    >>> aircraft = load_degengeom_csv("degen.csv")
    >>> len(aircraft.wings)
    2
    """

    cfg = config or OpenVSPImportConfig()
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"DegenGeom CSV file not found: {csv_path}")

    tables = _read_degengeom_sections(csv_path, cfg)
    if not tables:
        raise ValueError(
            f"No lifting-surface point tables were found in {csv_path}. "
            "Expected a DegenGeom PLATE or SURFACE table with X/Y/Z coordinates."
        )

    aircraft = Aircraft(uid=_sanitize_uid(csv_path.stem or "openvsp_aircraft"))
    for index, table in enumerate(tables):
        wing = _point_table_to_wing(table, cfg, index)
        aircraft.add_wing(wing)

    aircraft.compute_default_refs()
    return aircraft


def load_openvsp_surface(
    vsp_file_or_csv: str | Path,
    config: OpenVSPImportConfig | None = None,
    save_json: str | Path | None = None,
) -> Aircraft:
    """
    Load an OpenVSP ``.vsp3`` file or DegenGeom ``.csv`` into an ``Aircraft``.

    ``.csv`` files are parsed directly and do not require OpenVSP. ``.vsp3``
    files are first exported to DegenGeom CSV using the optional OpenVSP Python
    API.
    """

    cfg = config or OpenVSPImportConfig()
    input_path = Path(vsp_file_or_csv)
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        aircraft = load_degengeom_csv(input_path, cfg)
    elif suffix == ".vsp3":
        with tempfile.TemporaryDirectory(prefix="openonda_openvsp_") as tmpdir:
            csv_path = Path(tmpdir) / f"{input_path.stem}_degengeom.csv"
            export_openvsp_degengeom(input_path, csv_path, cfg)
            aircraft = load_degengeom_csv(csv_path, cfg)
    else:
        raise ValueError(
            f"Unsupported OpenVSP surface input '{input_path}'. "
            "Use a .csv DegenGeom file or a .vsp3 file."
        )

    if save_json is not None:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        save_surface(aircraft, str(save_json))
    return aircraft


def _import_openvsp() -> Any:
    message = (
        "Direct .vsp3 import requires the OpenVSP Python API (`openvsp`). "
        "Install OpenVSP with Python bindings, or export DegenGeom CSV "
        "manually from OpenVSP and call load_degengeom_csv(...)."
    )
    try:
        import openvsp as vsp
    except ImportError as exc:
        raise ImportError(message) from exc
    required_api = ("GetVSPVersion", "ClearVSPModel", "AddGeom")
    if any(not hasattr(vsp, name) for name in required_api):
        raise ImportError(message)
    return vsp


def _export_openvsp_degengeom_external(
    vsp_path: Path,
    csv_path: Path,
    cfg: OpenVSPImportConfig,
) -> Path:
    command = _openvsp_python_command()
    if command is None:
        raise ImportError(
            "Direct .vsp3 import requires either an importable OpenVSP Python API "
            "or an `openvsp-python` helper on PATH. Run "
            "`scripts/install/install_openvsp.sh`, or export DegenGeom CSV from "
            "OpenVSP manually and call load_degengeom_csv(...)."
        )

    set_id = "SET_ALL" if cfg.set_id == "ALL" else str(int(cfg.set_id))
    script = f"""
from pathlib import Path
import openvsp as vsp

vsp_path = Path({str(vsp_path)!r})
csv_path = Path({str(csv_path)!r})
csv_path.parent.mkdir(parents=True, exist_ok=True)

vsp.ClearVSPModel()
vsp.ReadVSPFile(str(vsp_path))
vsp.Update()
set_id = getattr(vsp, "SET_ALL", 0) if {set_id!r} == "SET_ALL" else int({set_id!r})
vsp.SetComputationFileName(vsp.DEGEN_GEOM_CSV_TYPE, str(csv_path))
vsp.ComputeDegenGeom(set_id, vsp.DEGEN_GEOM_CSV_TYPE)
if not csv_path.exists():
    raise SystemExit(f"OpenVSP did not create {{csv_path}}")
"""
    subprocess.run([*command, "-c", script], check=True, env=_openvsp_subprocess_env())
    return csv_path


def _openvsp_python_command() -> list[str] | None:
    configured = os.environ.get("OPENONDA_OPENVSP_PYTHON") or os.environ.get("OPENVSP_PYTHON")
    candidates = [
        configured,
        shutil.which("openvsp-python"),
        str(Path.home() / "anaconda3/envs/openonda-openvsp/bin/python"),
        str(Path.home() / "miniforge3/envs/openonda-openvsp/bin/python"),
        str(Path.home() / "mambaforge/envs/openonda-openvsp/bin/python"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return [candidate]
    return None


def _openvsp_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    root = _openvsp_root()
    if root is not None:
        python_paths = [
            root / "python/openvsp",
            root / "python/degen_geom",
            root / "python/openvsp_config",
            root / "python/utilities",
        ]
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(path) for path in python_paths if path.exists()]
            + ([existing_pythonpath] if existing_pythonpath else [])
        )
        existing_ld_library_path = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(root / "lib")] + ([existing_ld_library_path] if existing_ld_library_path else [])
        )
    return env


def _openvsp_root() -> Path | None:
    candidates = [
        os.environ.get("OPENVSP_ROOT"),
        os.environ.get("OPENVSP_PATH"),
        str(Path.home() / "OpenVSP-3.51.0"),
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if (path / "python/openvsp").exists():
                return path
    return None


def _call_if_available(module: Any, name: str) -> None:
    fn = getattr(module, name, None)
    if callable(fn):
        fn()


def _try_set_analysis_input(vsp: Any, analysis: str, key: str, value: int | str) -> bool:
    setters: tuple[tuple[str, Any], ...]
    if isinstance(value, str):
        setters = (("SetStringAnalysisInput", [value]),)
    else:
        setters = (
            ("SetIntAnalysisInput", [int(value)]),
            ("SetDoubleAnalysisInput", [float(value)]),
        )

    for setter_name, payload in setters:
        setter = getattr(vsp, setter_name, None)
        if not callable(setter):
            continue
        try:
            setter(analysis, key, payload)
            return True
        except Exception:
            continue
    return False


def _read_degengeom_sections(csv_path: Path, config: OpenVSPImportConfig) -> list[_PointTable]:
    rows = _read_csv_rows(csv_path)
    tables: list[_PointTable] = []
    current_name = csv_path.stem
    current_type = "wing"
    current_section = ""
    pending_dims: tuple[int, int] | None = None
    i = 0

    while i < len(rows):
        row = rows[i]
        first = _norm(row[0]) if row else ""

        if not row:
            pending_dims = None
            i += 1
            continue

        if first in {"geom", "component", "lifting_surface"}:
            name, surface_type = _parse_component_metadata(row, current_name, current_type)
            current_name = name
            current_type = surface_type
            current_section = first
            i += 1
            continue

        if first in _POINT_SECTIONS or first in _NON_SURFACE_SECTIONS:
            current_section = first
            pending_dims = _parse_section_dims(row) if first in _POINT_SECTIONS else None
            i += 1
            continue

        if _is_num_dims_header(row) and i + 1 < len(rows):
            dims = _parse_num_dims(rows[i + 1])
            if dims is not None:
                pending_dims = dims
                i += 2
                continue

        header = [_norm(cell) for cell in row]
        if {"x", "y", "z"}.issubset(set(header)):
            table, next_i = _parse_point_table(
                rows,
                i,
                current_name,
                current_type,
                current_section,
                pending_dims,
                config,
            )
            if table is not None and _component_is_selected(table, config):
                tables.append(table)
            pending_dims = None
            i = next_i
            continue

        i += 1

    return tables


def _read_csv_rows(csv_path: Path) -> list[list[str]]:
    with csv_path.open(newline="") as stream:
        reader = csv.reader(stream)
        rows = []
        for row in reader:
            cleaned = [cell.strip().lstrip("\ufeff") for cell in row]
            while cleaned and cleaned[-1] == "":
                cleaned.pop()
            rows.append(cleaned)
        return rows


def _parse_component_metadata(
    row: list[str],
    default_name: str,
    default_type: str,
) -> tuple[str, str]:
    if len(row) == 1:
        return default_name, default_type

    marker = _norm(row[0])
    name = (
        (row[1] or default_name)
        if marker in {"geom", "component", "lifting_surface"} and len(row) >= 2
        else default_name
    )

    surface_type = default_type
    for idx, cell in enumerate(row):
        key = _norm(cell)
        if key in {"name", "comp_name"} and idx + 1 < len(row):
            name = row[idx + 1] or name
        elif key in {"type", "surf_type", "surftype"} and idx + 1 < len(row):
            surface_type = _infer_surface_type(row[idx + 1], name)

    if marker == "lifting_surface" and len(row) >= 3 and row[2]:
        name = f"{name}_{_sanitize_uid(row[2], fallback='0')}"
        surface_type = _infer_surface_type(marker, name)
    elif len(row) >= 3:
        surface_type = _infer_surface_type(row[-1], name)
    else:
        surface_type = _infer_surface_type(surface_type, name)
    return name, surface_type


def _is_num_dims_header(row: list[str]) -> bool:
    header = {_norm(cell) for cell in row}
    return "num_u" in header and "num_w" in header


def _parse_num_dims(row: list[str]) -> tuple[int, int] | None:
    nums = []
    for cell in row:
        try:
            nums.append(int(float(cell)))
        except ValueError:
            continue
    if len(nums) >= 3:
        return nums[-2], nums[-1]
    if len(nums) == 2:
        return nums[0], nums[1]
    return None


def _parse_section_dims(row: list[str]) -> tuple[int, int] | None:
    if len(row) < 3:
        return None
    try:
        return int(float(row[1])), int(float(row[2]))
    except ValueError:
        return None


def _parse_point_table(
    rows: list[list[str]],
    header_index: int,
    current_name: str,
    current_type: str,
    current_section: str,
    dims: tuple[int, int] | None,
    config: OpenVSPImportConfig,
) -> tuple[_PointTable | None, int]:
    header = [_norm(cell) for cell in rows[header_index]]
    col = {name: idx for idx, name in enumerate(header)}
    x_idx, y_idx, z_idx = col["x"], col["y"], col["z"]
    u_idx = col.get("u")
    w_idx = col.get("w")
    name_idx = _first_present_col(col, "name", "comp_name")
    type_idx = _first_present_col(col, "type", "surf_type", "surftype")

    points: list[list[float]] = []
    u_values: list[float] = []
    w_values: list[float] = []
    table_name = current_name
    table_type = current_type
    i = header_index + 1

    while i < len(rows):
        row = rows[i]
        first = _norm(row[0]) if row else ""
        if (
            not row
            or first in _POINT_SECTIONS
            or first in _NON_SURFACE_SECTIONS
            or first.startswith("degengeom_type")
            or first in {"geom", "component", "lifting_surface"}
            or _is_num_dims_header(row)
        ):
            break

        try:
            points.append([float(row[x_idx]), float(row[y_idx]), float(row[z_idx])])
            if u_idx is not None and u_idx < len(row):
                u_values.append(float(row[u_idx]))
            if w_idx is not None and w_idx < len(row):
                w_values.append(float(row[w_idx]))
            if name_idx is not None and name_idx < len(row) and row[name_idx]:
                table_name = row[name_idx]
            if type_idx is not None and type_idx < len(row) and row[type_idx]:
                table_type = _infer_surface_type(row[type_idx], table_name)
        except (IndexError, ValueError):
            if config.strict:
                raise ValueError(
                    f"Invalid DegenGeom point row after header line {header_index + 1}: {row}"
                ) from None
            break
        i += 1

    if not points or current_section in _NON_SURFACE_SECTIONS:
        return None, i

    points_np = _apply_coordinate_transform(np.asarray(points, dtype=float), config)
    u_np = np.asarray(u_values, dtype=float) if len(u_values) == len(points) else None
    w_np = np.asarray(w_values, dtype=float) if len(w_values) == len(points) else None
    return (
        _PointTable(
            name=table_name,
            surface_type=_infer_surface_type(table_type, table_name),
            section=current_section or "surface",
            points=points_np,
            u=u_np,
            w=w_np,
            dims=dims,
        ),
        i,
    )


def _point_table_to_wing(table: _PointTable, config: OpenVSPImportConfig, index: int) -> Wing:
    grid = _point_table_to_grid(table, config)
    n_chord, n_span = grid.shape[:2]
    if n_chord < 2 or n_span < 2:
        raise ValueError(
            f"OpenVSP surface '{table.name}' needs at least 2 chord and 2 span stations; "
            f"got {n_chord} by {n_span}."
        )

    wing_uid = _sanitize_uid(table.name, fallback=f"openvsp_surface_{index}")
    wing = Wing(uid=wing_uid, symmetry=_symmetry_to_int(config.symmetry))
    span_panel_counts = _infer_span_panel_counts(n_span - 1, config)
    panels_chord = _infer_chord_panel_count(n_chord, config)

    for station in range(n_span - 1):
        segment = WingSegment(
            uid=f"{wing_uid}_segment_{station}",
            vertices={
                "a": grid[0, station].copy(),
                "b": grid[0, station + 1].copy(),
                "c": grid[-1, station + 1].copy(),
                "d": grid[-1, station].copy(),
            },
            panels_chord=panels_chord,
            panels_span=span_panel_counts[station],
            airfoils={"inner": "flat", "outer": "flat"},
            geometry={"source": "openvsp_degengeom"},
        )
        _validate_segment_orientation(segment, table.name)
        wing.add_segment(segment)

    return wing


def _point_table_to_grid(table: _PointTable, config: OpenVSPImportConfig) -> np.ndarray:
    if table.u is not None and table.w is not None:
        u_unique = _unique_sorted(table.u)
        w_unique = _unique_sorted(table.w)
        grid = np.full((len(u_unique), len(w_unique), 3), np.nan)
        for point, u_value, w_value in zip(table.points, table.u, table.w, strict=False):
            u_idx = int(np.argmin(np.abs(u_unique - u_value)))
            w_idx = int(np.argmin(np.abs(w_unique - w_value)))
            grid[u_idx, w_idx] = point
        if np.isnan(grid).any():
            message = f"DegenGeom table '{table.name}' has an incomplete U/W point grid."
            if config.strict:
                raise ValueError(message)
            grid = _drop_incomplete_grid_lines(grid)
        return grid

    if table.dims is not None:
        n_u, n_w = table.dims
        expected = n_u * n_w
        if table.points.shape[0] < expected:
            raise ValueError(
                f"DegenGeom table '{table.name}' declares {expected} points "
                f"({n_u} x {n_w}) but only {table.points.shape[0]} were parsed."
            )
        grid = table.points[:expected].reshape(n_u, n_w, 3)
        if table.section == "plate":
            return np.transpose(grid, (1, 0, 2))
        return grid

    raise ValueError(
        f"DegenGeom table '{table.name}' does not provide U/W columns or Num_U/Num_W dimensions."
    )


def _apply_coordinate_transform(points: np.ndarray, config: OpenVSPImportConfig) -> np.ndarray:
    transformed = np.asarray(points, dtype=float) * float(config.length_scale)
    transform = config.coordinate_transform
    if transform is None:
        return transformed

    matrix = np.asarray(transform, dtype=float)
    if matrix.shape == (3, 3):
        return transformed @ matrix.T
    if matrix.shape == (4, 4):
        homogeneous = np.ones((transformed.shape[0], 4), dtype=float)
        homogeneous[:, :3] = transformed
        return (homogeneous @ matrix.T)[:, :3]
    raise ValueError("coordinate_transform must be a 3x3 or 4x4 matrix.")


def _drop_incomplete_grid_lines(grid: np.ndarray) -> np.ndarray:
    valid_u = ~np.isnan(grid).any(axis=(1, 2))
    valid_w = ~np.isnan(grid).any(axis=(0, 2))
    return grid[valid_u][:, valid_w]


def _infer_chord_panel_count(n_chord_stations: int, config: OpenVSPImportConfig) -> int:
    if config.panels_chord is not None:
        return max(1, int(config.panels_chord))
    if config.preserve_vsp_paneling:
        return max(1, n_chord_stations - 1)
    return 1


def _infer_span_panel_counts(num_segments: int, config: OpenVSPImportConfig) -> list[int]:
    if config.panels_span is None:
        return [1] * num_segments

    total = max(num_segments, int(config.panels_span))
    counts = np.ones(num_segments, dtype=int)
    remaining = total - num_segments
    for idx in range(remaining):
        counts[idx % num_segments] += 1
    return counts.tolist()


def _validate_segment_orientation(segment: WingSegment, surface_name: str) -> None:
    if not np.isfinite(segment.area) or segment.area <= 1e-12:
        raise ValueError(
            f"OpenVSP surface '{surface_name}' produced a degenerate segment "
            f"'{segment.uid}' with area {segment.area:.6e}."
        )

    chord_root = segment.vertices["d"] - segment.vertices["a"]
    chord_tip = segment.vertices["c"] - segment.vertices["b"]
    if np.linalg.norm(chord_root) <= 1e-12 or np.linalg.norm(chord_tip) <= 1e-12:
        raise ValueError(
            f"OpenVSP surface '{surface_name}' produced segment '{segment.uid}' "
            "with coincident leading/trailing-edge vertices."
        )


def _component_is_selected(table: _PointTable, config: OpenVSPImportConfig) -> bool:
    name = table.name.lower()
    include = config.include_components
    exclude = config.exclude_components
    targets = {item.lower() for item in config.target_surface_types}

    if include and not any(item.lower() in name for item in include):
        return False
    if exclude and any(item.lower() in name for item in exclude):
        return False
    return table.surface_type.lower() in targets


def _infer_surface_type(value: str, name: str) -> str:
    text = f"{value} {name}".lower()
    if "rotor" in text or "prop" in text or "blade" in text:
        return "rotor" if "prop" not in text else "prop"
    if "wing" in text or "lift" in text:
        return "wing"
    return value.lower() if value and value.lower() in {"wing", "prop", "rotor"} else "wing"


def _symmetry_to_int(symmetry: str) -> int:
    value = str(symmetry).strip().lower()
    if value in {"none", "off", "false", "0", "from_openvsp"}:
        return 0
    mapping = {"xy": 1, "xz": 2, "yz": 3}
    if value in mapping:
        return mapping[value]
    try:
        integer = int(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported symmetry option: {symmetry!r}") from exc
    if integer not in (0, 1, 2, 3):
        raise ValueError(f"symmetry must map to 0, 1, 2, or 3; got {symmetry!r}")
    return integer


def _unique_sorted(values: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(values, dtype=float), decimals=12)
    return np.unique(rounded)


def _first_present_col(columns: dict[str, int], *names: str) -> int | None:
    for name in names:
        if name in columns:
            return columns[name]
    return None


def _sanitize_uid(value: str, fallback: str = "openvsp_surface") -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _norm(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")


__all__ = [
    "OpenVSPImportConfig",
    "export_openvsp_degengeom",
    "load_degengeom_csv",
    "load_openvsp_surface",
]
