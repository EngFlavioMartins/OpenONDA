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

from ..geometry.stl_audit import audit_stl_mesh, orient_components_by_signed_volume
from ..geometry.stl_io import load_stl
from .lattice import PanelLattice

if TYPE_CHECKING:
    from ..coupling.kinematics import PanelKinematics

logger = logging.getLogger("vpm")


def load_and_audit_body_stl(
    filepath: str,
    fix_normal_orientation: bool = True,
    validate: bool = True,
    max_panels: int | None = None,
    expected_components: int | None = None,
) -> np.ndarray:
    """Load, validate, and orient a body STL without allocating GPU state.

    This is deliberately separate from :func:`upload_body_to_lattice` so a
    caller can reject invalid geometry *before* a Taichi lattice and dense
    influence matrix are allocated for it.

    ``validate`` runs :func:`~..geometry.stl_audit.audit_stl_mesh`, rejecting
    non-finite coordinates, degenerate/duplicate triangles, open or
    non-manifold edges, inconsistent winding, panel-count overflow, and
    undeclared multi-body STLs. ``fix_normal_orientation`` then orients each
    closed connected component using its topological signed volume
    (:func:`~..geometry.stl_audit.orient_components_by_signed_volume`), which
    is correct for concave bodies, unlike a per-panel geometric-centre
    heuristic.
    """
    vertex_position, _ = load_stl(filepath)

    if validate:
        audit_stl_mesh(
            vertex_position, max_panels=max_panels, expected_components=expected_components
        )

    if fix_normal_orientation:
        vertex_position = orient_components_by_signed_volume(vertex_position)
    return vertex_position


def upload_body_to_lattice(
    lattice: PanelLattice,
    uid: str,
    vertex_position: np.ndarray,
    kinematics: PanelKinematics = None,
    group_id: int = 0,
) -> int:
    """Upload already-audited triangles to the lattice; returns the panel count."""
    # add_body appends panels at the end of the lattice; record the range of
    # the new body from the panel count before/after (add_body returns None).
    start = lattice.n_panels
    lattice.add_body(uid, vertex_position, kinematics=kinematics, group_id=group_id)
    count = lattice.n_panels - start

    # Initial geometry update at t=0
    lattice.update_geometry(0.0)
    return count


def add_body_from_mesh_stl(
    lattice: PanelLattice,
    uid: str,
    filepath: str,
    kinematics: PanelKinematics = None,
    group_id: int = 0,
    fix_normal_orientation: bool = True,
    validate: bool = True,
    max_panels: int | None = None,
    expected_components: int | None = None,
):
    """
    Main entry point for adding a panel body from an STL with topology data.

    Callers that own the lattice allocation should prefer
    :func:`load_and_audit_body_stl` followed by
    :func:`upload_body_to_lattice`, so an invalid STL is rejected before the
    lattice exists at all.
    """
    vertex_position = load_and_audit_body_stl(
        filepath,
        fix_normal_orientation=fix_normal_orientation,
        validate=validate,
        max_panels=max_panels,
        expected_components=expected_components,
    )
    count = upload_body_to_lattice(
        lattice, uid, vertex_position, kinematics=kinematics, group_id=group_id
    )
    logger.debug(f"Added body '{uid}' with {count} panels from '{filepath}'.")
