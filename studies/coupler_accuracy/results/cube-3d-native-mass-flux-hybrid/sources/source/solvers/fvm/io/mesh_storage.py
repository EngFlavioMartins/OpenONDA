"""Portable, backend-neutral storage for solver-native FVM meshes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .storage import require_free_space

FORMAT_VERSION = 1


def _jsonable(value: Any) -> Any:
    """Return metadata composed only of strict JSON scalar containers."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"Mesh metadata contains unsupported {type(value).__name__}")


def _packed_faces(faces: Any) -> tuple[np.ndarray, np.ndarray, bool]:
    """Pack fixed- or variable-width polygon connectivity without pickling."""
    fixed_width = isinstance(faces, np.ndarray) and faces.ndim == 2
    polygons = [np.asarray(face, dtype=np.int32) for face in faces]
    counts = np.fromiter((len(face) for face in polygons), dtype=np.int64, count=len(polygons))
    offsets = np.empty(len(polygons) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    vertices = (
        np.ascontiguousarray(np.concatenate(polygons), dtype=np.int32)
        if polygons
        else np.empty(0, dtype=np.int32)
    )
    return vertices, offsets, fixed_width


def save_native_mesh(mesh_data: dict[str, Any], path: str | Path) -> Path:
    """Atomically save a complete native mesh as a compressed ``.npz`` archive."""
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("Native FVM mesh files must use the '.npz' extension")

    face_vertices, face_offsets, fixed_width = _packed_faces(mesh_data["faces"])
    arrays: dict[str, np.ndarray] = {
        "face_vertices": face_vertices,
        "face_offsets": face_offsets,
    }
    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "faces_fixed_width": fixed_width,
    }
    for name, value in mesh_data.items():
        if name == "faces" or name.startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError(f"Mesh array {name!r} cannot use object dtype")
            arrays[name] = np.ascontiguousarray(value)
        else:
            metadata[name] = _jsonable(value)
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = sum(int(value.nbytes) + 4096 for value in arrays.values())
    require_free_space(destination, payload_bytes + (4 << 20))
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            # NumPy accepts arbitrary archive member names here; its stub models
            # one reserved keyword and consequently rejects a typed mapping.
            np.savez_compressed(stream, **arrays)  # pyrefly: ignore[bad-argument-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return destination


def load_native_mesh(path: str | Path) -> dict[str, Any]:
    """Load a mesh written by :func:`save_native_mesh` without object arrays."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {"metadata", "face_vertices", "face_offsets"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Invalid native FVM mesh; missing fields: {sorted(missing)}")
        metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
        version = int(metadata.pop("format_version", -1))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported native FVM mesh version {version}; expected {FORMAT_VERSION}"
            )
        fixed_width = bool(metadata.pop("faces_fixed_width", False))
        vertices = np.asarray(archive["face_vertices"], dtype=np.int32)
        offsets = np.asarray(archive["face_offsets"], dtype=np.int64)
        if offsets.ndim != 1 or len(offsets) == 0 or offsets[0] != 0:
            raise ValueError("Invalid native FVM face offsets")
        if np.any(np.diff(offsets) < 3) or offsets[-1] != len(vertices):
            raise ValueError("Invalid native FVM face connectivity")
        counts = np.diff(offsets)
        if fixed_width:
            if len(counts) and np.any(counts != counts[0]):
                raise ValueError("Fixed-width native FVM faces have inconsistent sizes")
            width = int(counts[0]) if len(counts) else 0
            faces: Any = np.ascontiguousarray(vertices.reshape(len(counts), width))
        else:
            faces = [vertices[offsets[i] : offsets[i + 1]].copy() for i in range(len(counts))]
        mesh_data: dict[str, Any] = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name not in required
        }
    mesh_data.update(metadata)
    mesh_data["faces"] = faces
    return mesh_data


__all__ = ["load_native_mesh", "save_native_mesh"]
