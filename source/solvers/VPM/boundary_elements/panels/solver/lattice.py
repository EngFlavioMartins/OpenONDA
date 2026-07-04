"""
Data module for VPM panel solver (Lattice structure).
==================
GPU-resident data structure for 3D panel geometry, strengths, and topology.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import taichi as ti

from ....config.constants import PANEL_EPSILON

if TYPE_CHECKING:
    from ..coupling.kinematics import PanelKinematics

logger = logging.getLogger("vpm")

@dataclass
class PanelBody:
    """Dataclass storing metadata for a collection of panels belonging to a single body."""

    uid: str
    start_idx: int
    count: int
    kinematics: "PanelKinematics"
    group_id: int = 0
    reference_area: float = 1.0

    @property
    def motion(self):
        return self.kinematics

@ti.data_oriented
class PanelLattice:
    """
    GPU-resident data structure representing the 3D panel lattice.
    Equivalent to VLMLattice but for arbitrary 3D triangular panel meshes.
    """

    def __init__(self, max_panels: int = 20000, float_dtype: str = "f32"):
        # Guard: Taichi must already be initialised before creating fields,
        # otherwise ti.field() triggers an auto-init with wrong precision.
        if ti.lang.impl.get_runtime().prog is None:
            raise RuntimeError(
                "PanelLattice must be created after ti.init(). "
                "Ensure the VPM Solver (which calls ti.init) is "
                "constructed before any PanelLattice instance."
            )
        self.max_panels = max_panels
        self.float_dtype = float_dtype
        self.ti_dtype = ti.f32 if float_dtype == "f32" else ti.f64
        self.num_panels = 0

        # Metadata
        self.bodies: list[PanelBody] = []

        self._init_fields()

    def _init_fields(self):
        dtype = self.ti_dtype
        N = self.max_panels

        # Geometry
        self.vertices = ti.Vector.field(3, dtype=dtype, shape=(N, 3))
        self.centers = ti.Vector.field(3, dtype=dtype, shape=N)
        self.normals = ti.Vector.field(3, dtype=dtype, shape=N)
        self.areas = ti.field(dtype=dtype, shape=N)
        self.local_vertices = ti.Vector.field(3, dtype=dtype, shape=(N, 3))

        # Kinematics
        self.velocities = ti.Vector.field(3, dtype=dtype, shape=N)

        # Strengths (Doublet / Potential)
        self.strengths = ti.field(dtype=dtype, shape=N)
        self.strengths_old = ti.field(dtype=dtype, shape=N)
        self.strengths_old2 = ti.field(dtype=dtype, shape=N)
        self.source_strengths = ti.field(dtype=dtype, shape=N)

        # Cumulative strengths for time-history integration
        self.cumulative_strengths = ti.field(dtype=dtype, shape=N)
        self.cumulative_strengths_old = ti.field(dtype=dtype, shape=N)
        self.cumulative_strengths_old2 = ti.field(dtype=dtype, shape=N)

        # Physics diagnostics
        self.phi_dot = ti.field(dtype=dtype, shape=N)
        self.Cp = ti.field(dtype=dtype, shape=N)

        # Panel identity
        self.panel_group_id = ti.field(ti.i32, shape=N)

    @ti.kernel
    def save_old_strengths(self):
        """Advance the timestep history of panel strengths."""
        for i in range(self.num_panels):
            self.strengths_old2[i] = self.strengths_old[i]
            self.strengths_old[i] = self.strengths[i]
            self.cumulative_strengths_old2[i] = self.cumulative_strengths_old[i]
            self.cumulative_strengths_old[i] = self.cumulative_strengths[i]

    @ti.kernel
    def _flip_normals_kernel(
        self, start_idx: int, count: int, ref_point: ti.types.vector(3, float)
    ):  # type: ignore
        for i in range(count):
            idx = start_idx + i
            r = self.centers[idx] - ref_point
            if r.dot(self.normals[idx]) < 0:
                self.normals[idx] = -self.normals[idx]
                v1, v2 = self.vertices[idx, 1], self.vertices[idx, 2]
                self.vertices[idx, 1], self.vertices[idx, 2] = v2, v1
                v1_loc, v2_loc = self.local_vertices[idx, 1], self.local_vertices[idx, 2]
                self.local_vertices[idx, 1], self.local_vertices[idx, 2] = v2_loc, v1_loc

    def flip_normals(self, start_idx: int, count: int, reference_point: np.ndarray):
        self._flip_normals_kernel(start_idx, count, ti.Vector(reference_point.tolist()))

    def update_geometry(self, t: float = 0.0):
        """Apply identity transform to initialise geometry from local_vertices."""
        n = self.num_panels
        if n == 0:
            return
        eye = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        zero = ti.Vector([0.0, 0.0, 0.0])
        self._update_geometry_kernel(0, n, eye, zero)

    @ti.kernel
    def _update_geometry_kernel(
        self,
        start_idx: int,
        count: int,
        R: ti.types.matrix(3, 3, float),
        T: ti.types.vector(3, float),
    ):  # type: ignore
        for i in range(count):
            idx = start_idx + i
            v0_l, v1_l, v2_l = (
                self.local_vertices[idx, 0],
                self.local_vertices[idx, 1],
                self.local_vertices[idx, 2],
            )
            v0, v1, v2 = R @ v0_l + T, R @ v1_l + T, R @ v2_l + T
            self.vertices[idx, 0], self.vertices[idx, 1], self.vertices[idx, 2] = v0, v1, v2

            cp = (v1 - v0).cross(v2 - v0)
            area = cp.norm() * 0.5
            self.areas[idx] = area
            if area > PANEL_EPSILON:
                self.normals[idx] = cp / (2.0 * area)
            else:
                self.normals[idx] = ti.Vector([0.0, 0.0, 1.0])
            self.centers[idx] = (v0 + v1 + v2) / 3.0

    @ti.kernel
    def _update_velocity_kernel(
        self,
        start_idx: int,
        count: int,
        v_lin: ti.types.vector(3, float),
        omega: ti.types.vector(3, float),
        center: ti.types.vector(3, float),
    ):  # type: ignore
        for i in range(count):
            idx = start_idx + i
            pos = self.centers[idx]
            self.velocities[idx] = v_lin + omega.cross(pos - center)

    def add_body(
        self,
        uid: str,
        vertices: np.ndarray,
        motion: Optional["PanelKinematics"] = None,
        group_id: int = 0,
    ):
        from ..coupling.kinematics import StaticPanel

        if motion is None:
            motion = StaticPanel()

        num_new = vertices.shape[0]
        if self.num_panels + num_new > self.max_panels:
            raise RuntimeError(
                f"Max panels exceeded: {self.num_panels + num_new} > {self.max_panels}"
            )

        start_idx = self.num_panels
        self.bodies.append(
            PanelBody(
                uid=uid, start_idx=start_idx, count=num_new, kinematics=motion, group_id=group_id
            )
        )

        # Upload
        v_flat = vertices.reshape(-1, 3).astype(
            np.float32 if self.float_dtype == "f32" else np.float64
        )
        self.local_vertices.from_numpy(v_flat.reshape(num_new, 3, 3))

        # Set initial group ID
        group_ids = np.full(num_new, group_id, dtype=np.int32)
        group_id_np = self.panel_group_id.to_numpy()
        group_id_np[start_idx : start_idx + num_new] = group_ids
        self.panel_group_id.from_numpy(group_id_np)

        self.num_panels += num_new
        logger.info(f"Added PanelBody '{uid}' with {num_new} panels at index {start_idx}.")
