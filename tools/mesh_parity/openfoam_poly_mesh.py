"""Small, strict reader and writer for ASCII OpenFOAM ``polyMesh`` files.

The parity harness must be able to inspect a mesh without relying on OpenFOAM
Python bindings.  This module deliberately supports the portable ASCII form
written by the harness (``points``, ``faces``, ``owner``, ``neighbour`` and
``boundary``), not binary or gzip-compressed OpenFOAM files.  A binary result is
reported as an actionable error rather than being parsed incorrectly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np


class PolyMeshFormatError(ValueError):
    """Raised when an ASCII polyMesh file is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class BoundaryPatch:
    """One contiguous OpenFOAM boundary-face range."""

    name: str
    type: str
    start_face: int
    n_faces: int

    def __post_init__(self) -> None:
        if not self.name:
            raise PolyMeshFormatError("Boundary-patch names must be non-empty")
        if not self.type:
            raise PolyMeshFormatError(f"Boundary patch {self.name!r} has no type")
        if self.start_face < 0 or self.n_faces < 0:
            raise PolyMeshFormatError(f"Boundary patch {self.name!r} has an invalid face range")


@dataclass(frozen=True, slots=True)
class PolyMesh:
    """Numbering-preserving OpenFOAM polyhedral mesh representation."""

    points: np.ndarray
    faces: tuple[np.ndarray, ...]
    owner: np.ndarray
    neighbour: np.ndarray
    boundary: tuple[BoundaryPatch, ...]
    n_cells: int

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        owner = np.asarray(self.owner, dtype=np.int64)
        neighbour = np.asarray(self.neighbour, dtype=np.int64)
        faces = tuple(np.asarray(face, dtype=np.int64) for face in self.faces)
        if points.ndim != 2 or points.shape[1] != 3:
            raise PolyMeshFormatError("points must have shape (n_points, 3)")
        if not np.isfinite(points).all():
            raise PolyMeshFormatError("points contain non-finite coordinates")
        if len(owner) != len(faces):
            raise PolyMeshFormatError("owner length must equal the number of faces")
        if len(neighbour) > len(faces):
            raise PolyMeshFormatError("neighbour length exceeds the number of faces")
        if self.n_cells < 0:
            raise PolyMeshFormatError("n_cells must be non-negative")
        if np.any(owner < 0) or np.any(owner >= self.n_cells):
            raise PolyMeshFormatError("owner contains a cell index outside n_cells")
        if np.any(neighbour < 0) or np.any(neighbour >= self.n_cells):
            raise PolyMeshFormatError("neighbour contains a cell index outside n_cells")
        if np.any(owner[: len(neighbour)] == neighbour):
            raise PolyMeshFormatError("an internal face cannot own and neighbour the same cell")
        for face_index, face in enumerate(faces):
            if face.ndim != 1 or len(face) < 3:
                raise PolyMeshFormatError(f"face {face_index} has fewer than three vertices")
            if np.any(face < 0) or np.any(face >= len(points)):
                raise PolyMeshFormatError(f"face {face_index} references an invalid point")
            if len(np.unique(face)) != len(face):
                raise PolyMeshFormatError(f"face {face_index} repeats a point")
        boundary_start = len(neighbour)
        expected_start = boundary_start
        seen_names: set[str] = set()
        for patch in self.boundary:
            if patch.name in seen_names:
                raise PolyMeshFormatError(f"Duplicate boundary patch {patch.name!r}")
            seen_names.add(patch.name)
            if patch.start_face != expected_start:
                raise PolyMeshFormatError(
                    "Boundary patches must cover contiguous faces in OpenFOAM order: "
                    f"expected {expected_start}, got {patch.start_face} for {patch.name!r}"
                )
            expected_start += patch.n_faces
        if expected_start != len(faces):
            raise PolyMeshFormatError(
                f"Boundary patches cover faces through {expected_start}, expected {len(faces)}"
            )
        points = np.ascontiguousarray(points)
        owner = np.ascontiguousarray(owner)
        neighbour = np.ascontiguousarray(neighbour)
        points.setflags(write=False)
        owner.setflags(write=False)
        neighbour.setflags(write=False)
        for face in faces:
            face.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "neighbour", neighbour)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "boundary", tuple(self.boundary))

    @property
    def n_faces(self) -> int:
        """Total number of internal and boundary faces."""
        return len(self.faces)

    @property
    def n_internal_faces(self) -> int:
        """Number of owner/neighbour faces."""
        return len(self.neighbour)

    @property
    def n_boundary_faces(self) -> int:
        """Number of one-sided faces."""
        return self.n_faces - self.n_internal_faces

    @classmethod
    def from_openonda(cls, mesh_data: Mapping[str, Any]) -> PolyMesh:
        """Adapt OpenONDA's native face-based mesh dictionary without reordering it."""
        try:
            points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
            faces = tuple(np.asarray(face, dtype=np.int64) for face in mesh_data["faces"])
            owner = np.asarray(mesh_data["owners"], dtype=np.int64)
            neighbour = np.asarray(mesh_data["neighbours"], dtype=np.int64)
            raw_boundary = mesh_data["boundary"]
            n_cells = int(mesh_data["n_cells"])
        except KeyError as error:
            raise PolyMeshFormatError(f"OpenONDA mesh is missing {error.args[0]!r}") from error
        boundary = tuple(
            BoundaryPatch(
                name=str(item["name"]),
                type=str(item.get("type", "patch")),
                start_face=int(item["start_face"]),
                n_faces=int(item["n_faces"]),
            )
            for item in raw_boundary
        )
        return cls(points, faces, owner, neighbour, boundary, n_cells)


_TOKEN_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"|[(){};]|[^\s(){};]+', re.DOTALL)
_LINE_COMMENT_PATTERN = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.DOTALL)


def _tokens(path: Path) -> list[str]:
    """Return OpenFOAM tokens after checking that the file is ASCII."""
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise PolyMeshFormatError(
            f"{path} is binary; rerun cfMesh with writeFormat ascii for parity comparison"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PolyMeshFormatError(
            f"{path} is not UTF-8 ASCII-like text; rerun cfMesh with writeFormat ascii"
        ) from error
    text = _LINE_COMMENT_PATTERN.sub("", text)
    text = _BLOCK_COMMENT_PATTERN.sub("", text)
    return _TOKEN_PATTERN.findall(text)


def _unquote(token: str) -> str:
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    return token


def _is_integer(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def _skip_group(tokens: Sequence[str], start: int, opening: str, closing: str) -> int:
    if start >= len(tokens) or tokens[start] != opening:
        raise PolyMeshFormatError(f"Expected {opening!r} while parsing OpenFOAM header")
    depth = 0
    for index in range(start, len(tokens)):
        if tokens[index] == opening:
            depth += 1
        elif tokens[index] == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    raise PolyMeshFormatError(f"Unclosed {opening!r} group")


def _after_header(tokens: Sequence[str]) -> int:
    """Find the data section after an optional ``FoamFile`` header."""
    if not tokens:
        raise PolyMeshFormatError("Empty OpenFOAM file")
    if tokens[0] != "FoamFile":
        return 0
    if len(tokens) < 2:
        raise PolyMeshFormatError("Truncated FoamFile header")
    return _skip_group(tokens, 1, "{", "}")


def _expect_counted_list(tokens: Sequence[str], start: int, label: str) -> tuple[int, int]:
    if start >= len(tokens) or not _is_integer(tokens[start]):
        got = tokens[start] if start < len(tokens) else "end of file"
        raise PolyMeshFormatError(f"Expected {label} count, got {got!r}")
    count = int(tokens[start])
    if count < 0:
        raise PolyMeshFormatError(f"{label} count must be non-negative")
    opening = start + 1
    if opening >= len(tokens) or tokens[opening] != "(":
        raise PolyMeshFormatError(f"Expected '(' after {label} count")
    return count, opening + 1


def _parse_points(path: Path) -> np.ndarray:
    tokens = _tokens(path)
    count, index = _expect_counted_list(tokens, _after_header(tokens), "point")
    points = np.empty((count, 3), dtype=np.float64)
    for point_index in range(count):
        if index >= len(tokens) or tokens[index] != "(":
            raise PolyMeshFormatError(f"Expected point {point_index} in {path}")
        try:
            points[point_index] = [
                float(tokens[index + 1]),
                float(tokens[index + 2]),
                float(tokens[index + 3]),
            ]
        except (IndexError, ValueError) as error:
            raise PolyMeshFormatError(f"Malformed point {point_index} in {path}") from error
        if index + 4 >= len(tokens) or tokens[index + 4] != ")":
            raise PolyMeshFormatError(
                f"Point {point_index} in {path} does not have three coordinates"
            )
        index += 5
    if index >= len(tokens) or tokens[index] != ")":
        raise PolyMeshFormatError(f"Point list in {path} is not closed")
    return points


def _parse_faces(path: Path) -> tuple[np.ndarray, ...]:
    tokens = _tokens(path)
    count, index = _expect_counted_list(tokens, _after_header(tokens), "face")
    faces: list[np.ndarray] = []
    for face_index in range(count):
        if index >= len(tokens) or not _is_integer(tokens[index]):
            raise PolyMeshFormatError(f"Expected vertex count for face {face_index} in {path}")
        n_vertices = int(tokens[index])
        index += 1
        if n_vertices < 3 or index >= len(tokens) or tokens[index] != "(":
            raise PolyMeshFormatError(f"Malformed face {face_index} in {path}")
        index += 1
        values = tokens[index : index + n_vertices]
        if len(values) != n_vertices or not all(_is_integer(value) for value in values):
            raise PolyMeshFormatError(f"Malformed point labels for face {face_index} in {path}")
        index += n_vertices
        if index >= len(tokens) or tokens[index] != ")":
            raise PolyMeshFormatError(f"Face {face_index} in {path} is not closed")
        index += 1
        faces.append(np.asarray([int(value) for value in values], dtype=np.int64))
    if index >= len(tokens) or tokens[index] != ")":
        raise PolyMeshFormatError(f"Face list in {path} is not closed")
    return tuple(faces)


def _parse_labels(path: Path, label: str) -> np.ndarray:
    tokens = _tokens(path)
    count, index = _expect_counted_list(tokens, _after_header(tokens), label)
    values = tokens[index : index + count]
    if len(values) != count or not all(_is_integer(value) for value in values):
        raise PolyMeshFormatError(f"Malformed {label} list in {path}")
    index += count
    if index >= len(tokens) or tokens[index] != ")":
        raise PolyMeshFormatError(f"{label.capitalize()} list in {path} is not closed")
    return np.asarray([int(value) for value in values], dtype=np.int64)


def _parse_boundary(path: Path) -> tuple[BoundaryPatch, ...]:
    tokens = _tokens(path)
    count, index = _expect_counted_list(tokens, _after_header(tokens), "boundary-patch")
    patches: list[BoundaryPatch] = []
    for _ in range(count):
        if index >= len(tokens):
            raise PolyMeshFormatError(f"Boundary list in {path} ends before all patches")
        name = _unquote(tokens[index])
        index += 1
        if index >= len(tokens) or tokens[index] != "{":
            raise PolyMeshFormatError(f"Boundary patch {name!r} in {path} has no dictionary")
        index += 1
        entries: dict[str, str] = {}
        while index < len(tokens) and tokens[index] != "}":
            key = _unquote(tokens[index])
            index += 1
            if index >= len(tokens):
                raise PolyMeshFormatError(f"Boundary patch {name!r} ends unexpectedly")
            if tokens[index] in {"(", "{"}:
                opening = tokens[index]
                closing = ")" if opening == "(" else "}"
                end = _skip_group(tokens, index, opening, closing)
                entries[key] = " ".join(tokens[index:end])
                index = end
            else:
                entries[key] = _unquote(tokens[index])
                index += 1
            if index < len(tokens) and tokens[index] == ";":
                index += 1
        if index >= len(tokens) or tokens[index] != "}":
            raise PolyMeshFormatError(f"Boundary patch {name!r} in {path} is not closed")
        index += 1
        try:
            patches.append(
                BoundaryPatch(
                    name=name,
                    type=entries["type"],
                    start_face=int(entries["startFace"]),
                    n_faces=int(entries["nFaces"]),
                )
            )
        except KeyError as error:
            raise PolyMeshFormatError(
                f"Boundary patch {name!r} in {path} lacks {error.args[0]!r}"
            ) from error
        except ValueError as error:
            raise PolyMeshFormatError(
                f"Boundary patch {name!r} in {path} has invalid face labels"
            ) from error
    if index >= len(tokens) or tokens[index] != ")":
        raise PolyMeshFormatError(f"Boundary list in {path} is not closed")
    return tuple(patches)


def read_poly_mesh(directory: Path | str) -> PolyMesh:
    """Read the five ASCII files in an OpenFOAM ``polyMesh`` directory."""
    directory = Path(directory)
    required = ("points", "faces", "owner", "neighbour", "boundary")
    missing = [str(directory / name) for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError("polyMesh is missing required files: " + ", ".join(missing))
    points = _parse_points(directory / "points")
    faces = _parse_faces(directory / "faces")
    owner = _parse_labels(directory / "owner", "owner")
    neighbour = _parse_labels(directory / "neighbour", "neighbour")
    boundary = _parse_boundary(directory / "boundary")
    boundary_start = boundary[0].start_face if boundary else len(faces)
    if len(neighbour) == len(faces) and boundary_start < len(faces):
        # cfMesh's polyMeshGen writer emits a full-face neighbour list and
        # marks every boundary face with -1.  Normalize this unambiguous
        # dialect to OpenFOAM's usual internal-face-only representation.
        if len(neighbour) < boundary_start or np.any(neighbour[boundary_start:] != -1):
            raise PolyMeshFormatError(
                "A full-face neighbour list must use -1 for every boundary face"
            )
        neighbour = neighbour[:boundary_start]
    labels = np.concatenate((owner, neighbour))
    n_cells = int(labels.max(initial=-1) + 1)
    return PolyMesh(points, faces, owner, neighbour, boundary, n_cells)


def _foam_header(class_name: str, object_name: str) -> str:
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        '    location    "constant/polyMesh";\n'
        f"    object      {object_name};\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )


def _write_counted_lines(path: Path, header: str, lines: Iterable[str]) -> None:
    values = tuple(lines)
    path.write_text(header + f"{len(values)}\n(\n" + "\n".join(values) + "\n)\n", encoding="ascii")


def write_poly_mesh(mesh: PolyMesh, directory: Path | str) -> Path:
    """Write an ASCII polyMesh that can be reread by :func:`read_poly_mesh`.

    Existing mesh files are replaced only inside the caller-supplied directory;
    the directory itself is created when necessary.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write_counted_lines(
        directory / "points",
        _foam_header("vectorField", "points"),
        (f"({point[0]:.17g} {point[1]:.17g} {point[2]:.17g})" for point in mesh.points),
    )
    _write_counted_lines(
        directory / "faces",
        _foam_header("faceList", "faces"),
        (f"{len(face)}({' '.join(str(int(value)) for value in face)})" for face in mesh.faces),
    )
    _write_counted_lines(
        directory / "owner",
        _foam_header("labelList", "owner"),
        (str(int(value)) for value in mesh.owner),
    )
    _write_counted_lines(
        directory / "neighbour",
        _foam_header("labelList", "neighbour"),
        (str(int(value)) for value in mesh.neighbour),
    )
    entries = []
    for patch in mesh.boundary:
        entries.extend(
            (
                patch.name,
                "{",
                f"    type        {patch.type};",
                f"    nFaces      {patch.n_faces};",
                f"    startFace   {patch.start_face};",
                "}",
            )
        )
    (directory / "boundary").write_text(
        _foam_header("polyBoundaryMesh", "boundary")
        + f"{len(mesh.boundary)}\n(\n"
        + "\n".join(entries)
        + "\n)\n",
        encoding="ascii",
    )
    return directory


__all__ = [
    "BoundaryPatch",
    "PolyMesh",
    "PolyMeshFormatError",
    "read_poly_mesh",
    "write_poly_mesh",
]
