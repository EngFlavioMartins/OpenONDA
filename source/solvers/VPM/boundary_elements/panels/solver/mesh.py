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

from ..geometry.stl_io import build_mesh_topology, load_stl
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

    # Pre-calculate topology on CPU before upload
    neighbors, te, le = build_mesh_topology(vertices, max_neighbors=lattice.max_neighbors)

    # Add body to lattice (uploads vertices, group_id)
    lattice.add_body(uid, vertices, motion=motion, group_id=group_id)

    # Store start index for adding topology data
    start = lattice.bodies[-1].start_idx
    count = lattice.bodies[-1].count

    # Upload topology fields
    neighbor_np = lattice.neighbor_indices.to_numpy()
    neighbor_np[start : start + count, :] = neighbors
    lattice.neighbor_indices.from_numpy(neighbor_np)

    te_np = lattice.is_TE_panel.to_numpy()
    te_np[start : start + count] = te
    lattice.is_TE_panel.from_numpy(te_np)

    le_np = lattice.is_LE_panel.to_numpy()
    le_np[start : start + count] = le
    lattice.is_LE_panel.from_numpy(le_np)

    # Initial geometry update at t=0
    lattice.update_geometry(0.0)

    if fix_normals:
        # Flip normals away from centroid
        centers_np = lattice.centers.to_numpy()[start : start + count]
        centroid = np.mean(centers_np, axis=0)
        lattice.flip_normals(start, count, centroid)
        logger.debug(f"Fixed normals for '{uid}' using centroid: {centroid}.")
