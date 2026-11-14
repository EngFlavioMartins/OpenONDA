"""Import of OpenVSP DegenGeom CSV files as VLM surfaces.

The VLM solver works on the Aircraft, Wing and WingSegment primitives. The CSV
is read without the OpenVSP Python API, so a surface can be imported where
OpenVSP is not installed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .aircraft import Aircraft, Wing, WingSegment


@dataclass
class _PointTable:
    name: str
    surface_type: str
    section: str
    points: np.ndarray
    u: np.ndarray | None = None
    w: np.ndarray | None = None
    dims: tuple[int, int] | None = None


# Surface types imported from a DegenGeom file.
_SURFACE_TYPES = {"wing", "prop", "rotor"}
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


def load_degengeom_csv(csv_file: str | Path) -> Aircraft:
    """Read the wing, propeller and rotor surfaces of an OpenVSP DegenGeom CSV file.

    Each surface becomes one wing without symmetry. Its VSP cross sections give
    the chordwise panels, and every spanwise cell between two cross sections is
    one segment of one spanwise panel.

    Parameters
    ----------
    csv_file : str | Path
        Path to the DegenGeom CSV file.

    Returns
    -------
    Aircraft
        Aircraft with one wing per imported surface.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the file has no lifting-surface point table, a point row or a U/W
        grid is incomplete, or a segment is degenerate.
    """
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"DegenGeom CSV file not found: {csv_path}")

    tables = _read_degengeom_sections(csv_path)
    if not tables:
        raise ValueError(
            f"No lifting-surface point tables were found in {csv_path}. "
            "Expected a DegenGeom PLATE or SURFACE table with X/Y/Z coordinates."
        )

    aircraft = Aircraft(uid=_sanitize_uid(csv_path.stem or "openvsp_aircraft"))
    for index, table in enumerate(tables):
        aircraft.add_wing(_point_table_to_wing(table, index))

    aircraft.compute_default_refs()
    return aircraft


def _read_degengeom_sections(csv_path: Path) -> list[_PointTable]:
    """Parse selected surface point tables while tracking component names and dimensions."""
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
            )
            if table is not None and table.surface_type in _SURFACE_TYPES:
                tables.append(table)
            pending_dims = None
            i = next_i
            continue

        i += 1

    return tables


def _read_csv_rows(csv_path: Path) -> list[list[str]]:
    """Read CSV cells, removing surrounding whitespace, BOMs and trailing empty columns."""
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
    """Resolve component name and lifting-surface type from a DegenGeom marker row."""
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
    """Recognize a normalized Num_U/Num_W dimension header."""
    header = {_norm(cell) for cell in row}
    return "num_u" in header and "num_w" in header


def _parse_num_dims(row: list[str]) -> tuple[int, int] | None:
    """Extract the last two numeric grid dimensions, or return None if absent."""
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
    """Read U/W dimensions from a section marker, or return None if malformed."""
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
) -> tuple[_PointTable | None, int]:
    """Read a coordinate table; return it and the next unconsumed row index."""
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
            raise ValueError(
                f"Invalid DegenGeom point row after header line {header_index + 1}: {row}"
            ) from None
        i += 1

    if not points or current_section in _NON_SURFACE_SECTIONS:
        return None, i

    points_np = np.asarray(points, dtype=float)
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


def _point_table_to_wing(table: _PointTable, index: int) -> Wing:
    """Build the wing segments of a structured imported point grid."""
    grid = _point_table_to_grid(table)
    n_chord, n_span = grid.shape[:2]
    if n_chord < 2 or n_span < 2:
        raise ValueError(
            f"OpenVSP surface '{table.name}' needs at least 2 chord and 2 span stations; "
            f"got {n_chord} by {n_span}."
        )

    wing_uid = _sanitize_uid(table.name, fallback=f"openvsp_surface_{index}")
    wing = Wing(uid=wing_uid)

    for station in range(n_span - 1):
        segment = WingSegment(
            uid=f"{wing_uid}_segment_{station}",
            vertex_position={
                "a": grid[0, station].copy(),
                "b": grid[0, station + 1].copy(),
                "c": grid[-1, station + 1].copy(),
                "d": grid[-1, station].copy(),
            },
            n_chordwise_panels=n_chord - 1,
            n_spanwise_panels=1,
        )
        _validate_segment_orientation(segment, table.name)
        wing.add_segment(segment)

    return wing


def _point_table_to_grid(table: _PointTable) -> np.ndarray:
    """Arrange U/W-indexed or dimensioned points as a chord-by-span coordinate grid."""
    if table.u is not None and table.w is not None:
        u_unique = _unique_sorted(table.u)
        w_unique = _unique_sorted(table.w)
        grid = np.full((len(u_unique), len(w_unique), 3), np.nan)
        for point, u_value, w_value in zip(table.points, table.u, table.w, strict=False):
            u_idx = int(np.argmin(np.abs(u_unique - u_value)))
            w_idx = int(np.argmin(np.abs(w_unique - w_value)))
            grid[u_idx, w_idx] = point
        if np.isnan(grid).any():
            raise ValueError(f"DegenGeom table '{table.name}' has an incomplete U/W point grid.")
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


def _validate_segment_orientation(segment: WingSegment, surface_name: str) -> None:
    """Reject nonfinite/zero segment area and coincident leading/trailing edges."""
    if not np.isfinite(segment.area) or segment.area <= 1e-12:
        raise ValueError(
            f"OpenVSP surface '{surface_name}' produced a degenerate segment "
            f"'{segment.uid}' with area {segment.area:.6e}."
        )

    chord_root = segment.vertex_position["d"] - segment.vertex_position["a"]
    chord_tip = segment.vertex_position["c"] - segment.vertex_position["b"]
    if np.linalg.norm(chord_root) <= 1e-12 or np.linalg.norm(chord_tip) <= 1e-12:
        raise ValueError(
            f"OpenVSP surface '{surface_name}' produced segment '{segment.uid}' "
            "with coincident leading/trailing-edge vertex_position."
        )


def _infer_surface_type(value: str, name: str) -> str:
    """Infer wing, rotor or propeller type from normalized component metadata."""
    text = f"{value} {name}".lower()
    if "rotor" in text or "prop" in text or "blade" in text:
        return "rotor" if "prop" not in text else "prop"
    if "wing" in text or "lift" in text:
        return "wing"
    return value.lower() if value and value.lower() in {"wing", "prop", "rotor"} else "wing"


def _unique_sorted(values: np.ndarray) -> np.ndarray:
    """Return sorted coordinate labels rounded to twelve decimal places."""
    rounded = np.round(np.asarray(values, dtype=float), decimals=12)
    return np.unique(rounded)


def _first_present_col(columns: dict[str, int], *names: str) -> int | None:
    """Return the first matching named column index, or None."""
    for name in names:
        if name in columns:
            return columns[name]
    return None


def _sanitize_uid(value: str, fallback: str = "openvsp_surface") -> str:
    """Make a nonempty identifier from imported component text."""
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _norm(value: str) -> str:
    """Normalize a CSV label to lowercase alphanumeric words separated by underscores."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")


__all__ = ["load_degengeom_csv"]
