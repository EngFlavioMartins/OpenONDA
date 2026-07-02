"""
Surface geometry I/O for panel methods — JSON format mirroring ``vlm/geometry/surface_io.py``.

The panel method uses triangulated surfaces loaded from STL files, not parametric
wing definitions.  This module provides JSON serialisation of body metadata
(UID, STL path, transform, group ID) for scene-layout files, plus direct
vertex/face export for post-processing.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def save_surface_metadata(
    uid: str,
    stl_path: str,
    group_id: int = 0,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    filepath: str | Path = "",
) -> str:
    """Save surface metadata (UID, STL path, transform) to a JSON file.

    Parameters
    ----------
    uid : str
        Unique identifier for the surface body.
    stl_path : str
        Path to the STL file.
    group_id : int
        Group identifier (default ``0``).
    translation : tuple[float, float, float]
        Translation offset ``(x, y, z)`` (default ``(0, 0, 0)``).
    rotation_deg : tuple[float, float, float]
        Euler rotation angles in degrees ``(rx, ry, rz)`` (default ``(0, 0, 0)``).
    filepath : str | Path
        Output JSON path. When empty, returns the JSON string instead of
        writing to a file.

    Returns
    -------
    str
        Path to the saved file, or the JSON string if ``filepath`` is empty.

    Examples
    --------
    >>> save_surface_metadata("wing_1", "body.stl", filepath="wing.json")
    'wing.json'
    """
    metadata = {
        "uid": uid,
        "stl_path": str(stl_path),
        "group_id": group_id,
        "translation": list(translation),
        "rotation_deg": list(rotation_deg),
    }
    if filepath:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        return str(path)
    return json.dumps(metadata, indent=2)

def load_surface_metadata(filepath: str | Path) -> dict[str, Any]:
    """Load surface metadata from a JSON file written by
    :func:`save_surface_metadata`.

    Parameters
    ----------
    filepath : str | Path
        Path to the JSON metadata file.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``uid``, ``stl_path``, ``group_id``,
        ``translation``, and ``rotation_deg``.

    Examples
    --------
    >>> meta = load_surface_metadata("wing.json")
    >>> meta["uid"]
    'wing_1'
    """
    with open(filepath) as f:
        return json.load(f)

def save_scene(
    bodies: list[dict[str, Any]],
    filepath: str | Path,
    description: str = "",
) -> str:
    """Save a multi-body scene description to a JSON file.

    Each body in the ``bodies`` list should be a metadata dictionary
    (typically produced by :func:`save_surface_metadata`).

    Parameters
    ----------
    bodies : list[dict[str, Any]]
        List of body metadata dictionaries.
    filepath : str | Path
        Output JSON path.
    description : str
        Optional scene description string (default ``""``).

    Returns
    -------
    str
        Path to the saved file.

    Examples
    --------
    >>> save_scene([meta_wing, meta_fuselage], "scene.json", "Full aircraft")
    'scene.json'
    """
    scene = {
        "description": description,
        "bodies": bodies,
    }
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)
    return str(path)

def load_scene(filepath: str | Path) -> list[dict[str, Any]]:
    """Load a multi-body scene from a JSON file written by :func:`save_scene`.

    Parameters
    ----------
    filepath : str | Path
        Path to the scene JSON file.

    Returns
    -------
    list[dict[str, Any]]
        List of body metadata dictionaries.

    Examples
    --------
    >>> bodies = load_scene("scene.json")
    >>> len(bodies)
    2
    """
    with open(filepath) as f:
        scene = json.load(f)
    return scene.get("bodies", [])
