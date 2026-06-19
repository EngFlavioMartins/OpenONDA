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
    with open(filepath) as f:
        return json.load(f)


def save_scene(
    bodies: list[dict[str, Any]],
    filepath: str | Path,
    description: str = "",
) -> str:
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
    with open(filepath) as f:
        scene = json.load(f)
    return scene.get("bodies", [])
