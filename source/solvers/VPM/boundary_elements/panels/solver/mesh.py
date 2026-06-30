"""
Mesh upload and preprocessing for panel solver.
==================
Handles the conversion of STL geometry to PanelLattice GPU fields.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.stl_io import load_stl
from .lattice import PanelLattice

if TYPE_CHECKING:
    from ..coupling.kinematics import PanelKinematics

logger = logging.getLogger("vpm")

def add_body_from_mesh_stl(
    lattice: PanelLattice,
    uid: str,
    filepath: str,
    motion: PanelKinematics = None,
    group_id: int = 0,
    fix_normals: bool = True,
):
    """
    Main entry point for adding a panel body from an STL with topology data.
    """
    vertices, _ = load_stl(filepath)

    lattice.add_body(uid, vertices, motion=motion, group_id=group_id)

    # Initial geometry update at t=0
    lattice.update_geometry(0.0)

    if fix_normals:
        # Flip normals away from centroid
        centers_np = lattice.centers.to_numpy()[start : start + count]
        centroid = np.mean(centers_np, axis=0)
        lattice.flip_normals(start, count, centroid)
        logger.debug(f"Fixed normals for '{uid}' using centroid: {centroid}.")
