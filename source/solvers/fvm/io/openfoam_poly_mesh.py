# SPDX-License-Identifier: GPL-3.0-or-later
"""Small, dependency-free ASCII OpenFOAM ``polyMesh`` interchange layer.

The FVM solver uses a backend-neutral dictionary in memory.  This module keeps
the on-disk OpenFOAM representation useful for inspection and for exchanging a
generated mesh with tools that understand ``constant/polyMesh``; it does not
invoke OpenFOAM or require an OpenFOAM installation.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

from ..mesh.validation import validate_topology

_HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {foam_class};
    object      {object_name};
}}
"""


def _foam_list(values: list[str], foam_class: str, object_name: str) -> str:
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "| =========                 |                                                 |\\\n"
        "| \\      /  F ield         | OpenONDA native export                            |\\\n"
        "|  \\    /   O p e n F o a m|                                                 |\\\n"
        "|   \\  /    A p p l i c a t i o n |                                         |\\\n"
        "|    \\/     M e s h         |                                                 |\\\n"
        "\\*---------------------------------------------------------------------------*/\n"
        + _HEADER.format(foam_class=foam_class, object_name=object_name)
        + f"{len(values)}\n(\n"
        + "\n".join(values)
        + "\n)\n"
    )


def _format_point(point: np.ndarray) -> str:
    return "(" + " ".join(format(float(value), ".17g") for value in point) + ")"


def _format_face(face: Any) -> str:
    values = np.asarray(face, dtype=np.int64)
    return f"{len(values)}(" + " ".join(str(int(value)) for value in values) + ")"


def _write_boundary(mesh_data: dict[str, Any]) -> str:
    patches = mesh_data["boundary"]
    lines = [
        "/*--------------------------------*- C++ -*----------------------------------*\\",
        "| OpenONDA native ASCII polyMesh boundary export                              |",
        "\\*---------------------------------------------------------------------------*/",
        _HEADER.format(foam_class="polyBoundaryMesh", object_name="boundary").rstrip(),
        str(len(patches)),
        "(",
    ]
    for patch in patches:
        name = str(patch["name"])
        patch_type = str(patch.get("type", "patch"))
        lines.extend(
            (
                name,
                "{",
                f"    type            {patch_type};",
                f"    nFaces          {int(patch['n_faces'])};",
                f"    startFace       {int(patch['start_face'])};",
                "}",
            )
        )
    lines.extend((")", ""))
    return "\n".join(lines)


def write_poly_mesh(mesh_data: dict[str, Any], directory: str | Path) -> Path:
    """Write ``mesh_data`` to an ASCII ``constant/polyMesh`` directory.

    ``directory`` is the polyMesh directory itself.  Internal faces are kept
    in native owner order, and boundary faces follow the contiguous patch
    ranges required by both OpenONDA and OpenFOAM.
    """
    validate_topology(mesh_data)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = mesh_data["faces"]
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    (destination / "points").write_text(
        _foam_list(
            [_format_point(point) for point in points],
            "vectorField",
            "points",
        ),
        encoding="utf-8",
    )
    (destination / "faces").write_text(
        _foam_list([_format_face(face) for face in faces], "faceList", "faces"),
        encoding="utf-8",
    )
    (destination / "owner").write_text(
        _foam_list([str(int(value)) for value in owners], "labelList", "owner"),
        encoding="utf-8",
    )
    (destination / "neighbour").write_text(
        _foam_list([str(int(value)) for value in neighbours], "labelList", "neighbour"),
        encoding="utf-8",
    )
    (destination / "boundary").write_text(_write_boundary(mesh_data), encoding="utf-8")
    return destination


def _tokens(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    return re.findall(r"[^(){};\s]+|[(){};]", text)


def _read_list(path: Path) -> list[str]:
    tokens = _tokens(path)
    try:
        count = int(tokens[tokens.index("(") - 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid OpenFOAM list file: {path}") from exc
    open_index = tokens.index("(")
    close_index = len(tokens) - 1 - tokens[::-1].index(")")
    values = tokens[open_index + 1 : close_index]
    if len(values) != count and path.name != "faces":
        raise ValueError(f"OpenFOAM list count mismatch in {path}")
    return values


def _read_points(path: Path) -> np.ndarray:
    tokens = _tokens(path)
    open_index = tokens.index("(")
    close_index = len(tokens) - 1 - tokens[::-1].index(")")
    count = int(tokens[open_index - 1])
    # A vector is represented by a parenthesised tuple inside the list.
    points: list[tuple[float, float, float]] = []
    index = open_index + 1
    while index < close_index:
        if tokens[index] != "(":
            raise ValueError(f"Invalid point list in {path}")
        points.append(
            (
                float(tokens[index + 1]),
                float(tokens[index + 2]),
                float(tokens[index + 3]),
            )
        )
        if tokens[index + 4] != ")":
            raise ValueError(f"Invalid point tuple in {path}")
        index += 5
    if len(points) != count:
        raise ValueError(f"OpenFOAM point count mismatch in {path}")
    return np.asarray(points, dtype=np.float64)


def _read_faces(path: Path) -> list[np.ndarray]:
    tokens = _tokens(path)
    count = int(tokens[tokens.index("(") - 1])
    index = tokens.index("(") + 1
    result: list[np.ndarray] = []
    while index < len(tokens) and tokens[index] != ")":
        width = int(tokens[index])
        if tokens[index + 1] != "(":
            raise ValueError(f"Invalid face entry in {path}")
        values = [int(token) for token in tokens[index + 2 : index + 2 + width]]
        if tokens[index + 2 + width] != ")":
            raise ValueError(f"Invalid face closure in {path}")
        result.append(np.asarray(values, dtype=np.int32))
        index += width + 3
    if len(result) != count:
        raise ValueError(f"OpenFOAM face count mismatch in {path}")
    return result


def read_poly_mesh(directory: str | Path) -> dict[str, Any]:
    """Read the files produced by :func:`write_poly_mesh` into native data."""
    source = Path(directory)
    points = _read_points(source / "points")
    faces = _read_faces(source / "faces")
    owners = np.asarray([int(value) for value in _read_list(source / "owner")], dtype=np.int32)
    neighbours = np.asarray(
        [int(value) for value in _read_list(source / "neighbour")], dtype=np.int32
    )
    boundary_tokens = _tokens(source / "boundary")
    patches: list[dict[str, Any]] = []
    index = boundary_tokens.index("(") + 1
    while index < len(boundary_tokens) and boundary_tokens[index] != ")":
        name = boundary_tokens[index]
        if boundary_tokens[index + 1] != "{":
            raise ValueError(f"Invalid boundary patch entry in {source / 'boundary'}")
        end = boundary_tokens.index("}", index + 2)
        values = boundary_tokens[index + 2 : end]
        fields = {
            values[position]: values[position + 1]
            for position in range(0, len(values) - 1, 3)
            if values[position + 2] == ";"
        }
        patches.append(
            {
                "name": name,
                "type": fields.get("type", "patch"),
                "n_faces": int(fields["nFaces"]),
                "start_face": int(fields["startFace"]),
            }
        )
        index = end + 1
    mesh_data: dict[str, Any] = {
        "vertex_position": points,
        "faces": faces,
        "owners": owners,
        "neighbours": neighbours,
        "boundary": patches,
        "n_cells": int(owners.max(initial=-1)) + 1,
        "n_faces": len(faces),
        "n_interior_faces": len(neighbours),
        "n_points": len(points),
    }
    validate_topology(mesh_data)
    return mesh_data


__all__ = ["read_poly_mesh", "write_poly_mesh"]
