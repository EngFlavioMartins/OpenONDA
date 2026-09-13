# SPDX-License-Identifier: GPL-3.0-or-later
"""Native mesh assembly helpers."""

from __future__ import annotations

from typing import Any


def require_native_mesh(mesh_data: dict[str, Any]) -> dict[str, Any]:
    """Check the minimum face-based mesh contract and return the same object."""
    required = {"vertex_position", "faces", "owners", "neighbours", "boundary"}
    missing = sorted(required - set(mesh_data))
    if missing:
        raise ValueError(f"Native mesh assembly is missing keys: {missing}")
    return mesh_data


__all__ = ["require_native_mesh"]
