"""
GPU-resident VLM lattice data structure (VLMLattice): panel geometry,
circulations, and topology.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
import taichi as ti


def _set_vtk_array_name(array, field_name: str) -> None:
    """Assign a canonical physical-field name to a VTK array."""
    array.SetName(field_name)


@ti.data_oriented
class VLMLattice:
    """
    Taichi-native VLM lattice data structure.

    Stores panel geometry, horseshoe vortex points, and solution data.
    All fields are pre-allocated for efficient GPU computation.

    Attributes:
        max_n_panels: Maximum number of panels
        n_panels: Current number of active panels
        dtype: Taichi data type (ti.f32 or ti.f64)
    """

    def __init__(self, max_n_panels: int = 5000, dtype=ti.f32):
        """
        Initialize VLM lattice.

        Args:
            max_n_panels: Maximum number of panels to allocate
            dtype: Data type for floating point (ti.f32 or ti.f64)

        Raises:
            RuntimeError: If called before ``ti.init()`` (Taichi must be
                initialised first so that fields use the correct precision
                and backend).
        """
        # Guard: Taichi must already be initialised before creating fields,
        # otherwise ti.field() triggers an auto-init with wrong precision.
        if ti.lang.impl.get_runtime().prog is None:
            raise RuntimeError(
                "VLMLattice must be created after ti.init(). "
                "Ensure the VPM Solver (which calls ti.init) is "
                "constructed before any VLMLattice instance."
            )
        self.max_n_panels = int(max_n_panels)
        self.dtype = dtype
        self.np_dtype = np.float32 if dtype == ti.f32 else np.float64
        self.n_panels = 0
        print(f"VLMLattice initialized with max_n_panels={self.max_n_panels}")

        # Panel corner points (N x 4 x 3)
        # Order: [P, Q, R, S] where P-Q is LE, R-S is TE
        #   P ------- Q
        #   |         |
        #   S ------- R
        self.panel_corner_position = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels, 4))

        # Horseshoe vortex points (N x 4 x 3)
        # Order: [V1, V2, V3, V4]
        #   V1 = left trailing leg far endpoint (downstream infinity)
        #   V2 = bound leg left endpoint (at 25% chord)
        #   V3 = bound leg right endpoint (at 25% chord)
        #   V4 = right trailing leg far endpoint (downstream infinity)
        self.vortex_point_position = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels, 4))

        # Collocation points (N x 3) - at 75% chord
        self.collocation_point = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))

        # Panel normal (N x 3) - unit vectors
        self.normal = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))

        # Panel area (N,)
        self.area = ti.field(dtype=dtype, shape=(max_n_panels,))

        # Bound leg midpoints (N x 3) - for force calculation
        self.bound_vortex_midpoint = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))

        # Trailing edge direction vectors (N x 2 x 3)
        # [0] = left trailing direction, [1] = right trailing direction
        self.trailing_direction = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels, 2))

        # Solution: circulation strength Γ (N,) - matching VPM convention
        # This is the PER-PANEL circulation (local vortex ring strength)
        self.circulation = ti.field(dtype=dtype, shape=(max_n_panels,))

        # Previous circulation strength Γ_old (N,) - for wake shedding (dcirculation/dt)
        self.circulation_old = ti.field(dtype=dtype, shape=(max_n_panels,))

        # Time-averaged circulation 0.5*(Γ + Γ_old) used by kj_smoothing to
        # compute bound_vortex_velocity so V_bound and the force kernel use the same
        # smoothed circulation (eliminates the 2Δt oscillation in KJ forces).
        self.smoothed_circulation = ti.field(dtype=dtype, shape=(max_n_panels,))

        # Cumulative circulation at each panel (sum of all upstream panels)
        # For TE panels: Γ_cumulative = Σ γ_i along chordwise direction
        # This is what should be used for trailing vortex shedding
        self.cumulative_circulation = ti.field(dtype=dtype, shape=(max_n_panels,))

        # Previous cumulative circulation (for delta-shedding)
        self.cumulative_circulation_old = ti.field(dtype=dtype, shape=(max_n_panels,))

        # Aerodynamic influence coefficient (aerodynamic_influence_coefficient) matrix (N x N)
        # aerodynamic_influence_coefficient[i,j] = downwash at panel i due to unit circulation on panel j
        self.aerodynamic_influence_coefficient = ti.field(
            dtype=dtype, shape=(max_n_panels, max_n_panels)
        )

        # Right-hand side (boundary condition) (N,)
        self.right_hand_side = ti.field(dtype=dtype, shape=(max_n_panels,))

        # Velocity at collocation_point points (N x 3)
        self.velocity = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))

        # Velocity at bound vortex midpoints (N x 3) - for correct K-J force
        self.bound_vortex_velocity = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))

        # Kinematic velocity at collocation_point points (N x 3) - 2D scalar field for better stability
        self.kinematic_velocity = ti.field(dtype=dtype, shape=(max_n_panels, 3))

        # Pressure coefficient (N,) - matching VPM convention
        self.pressure_coefficient = ti.field(dtype=dtype, shape=(max_n_panels,))

        # Panel forces (N x 3)
        self.panel_force = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))

        # External velocity field at collocation_point points (N x 3) - matching VPM convention
        self.external_velocity = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))

        # Panel center position (N x 3) - for visualization/export
        self.panel_centre = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))

        # Bookkeeping: which wing/segment each panel belongs to
        self.wing_id = ti.field(dtype=ti.i32, shape=(max_n_panels,))
        self.segment_id = ti.field(dtype=ti.i32, shape=(max_n_panels,))
        self.is_mirrored = ti.field(dtype=ti.i32, shape=(max_n_panels,))

        # Topology Connectivity (N x 4)
        # [0]=Left, [1]=Right, [2]=Upstream, [3]=Downstream
        # Value -1 indicates no neighbor (e.g., edge or tip)
        self.neighbor_indices = ti.field(dtype=ti.i32, shape=(max_n_panels, 4))

        # Index of the trailing-edge panel in each panel's own chordwise strip.
        # For every panel this points to the TE panel (i == n_chord-1) that shares
        # its spanwise station. Used by the coupled-mode influence kernels so each
        # panel's internal trailing legs run from its bound vortex all the way to
        # the WING trailing edge (where the VPM wake takes over), rather than only
        # to the panel's own downstream edge. The latter truncation breaks the
        # chordwise accumulation of trailing vorticity and collapses the
        # finite-wing downwash (flat, non-tapering spanwise loading). Stored as an
        # index (not a position) so it tracks corner motion automatically.
        self.trailing_edge_index = ti.field(dtype=ti.i32, shape=(max_n_panels,))

        # Flags
        # 1 if panel is at Trailing Edge (no downstream neighbor), 0 otherwise
        self.is_trailing_edge = ti.field(dtype=ti.i32, shape=(max_n_panels,))

        # 1 if panel is at Leading Edge (no upstream neighbor), 0 otherwise
        # Static topology — set once after mesh generation via mark_le_panels()
        self.is_leading_edge = ti.field(dtype=ti.i32, shape=(max_n_panels,))

        # Leading Edge Suction Parameter (per LE panel, recomputed every step)
        # Convention: LESP = |Γ_panel| / (chord_ref * V_ref)  (dimensionless, ≥ 0)
        # Shedding triggers when LESP > lesp_crit (Ramesh et al. 2014)
        self.leading_edge_suction_parameter = ti.field(dtype=dtype, shape=(max_n_panels,))

        # WAKE PARTICLE BUFFER (for direct Taichi-to-VPM transfer)
        # Pre-allocated buffer for wake particles shed per time step.
        # Avoids numpy intermediate for GPU-to-GPU transfer.
        # Factor 3×: longitudinal (left+right) + transverse per TE panel,
        # plus headroom for LEV particles when enabled.
        max_n_wake_particles_per_step = 3 * self.max_n_panels
        # Particle properties (for wake shedding)
        self.wake_position = ti.Vector.field(3, dtype=dtype, shape=(max_n_wake_particles_per_step,))
        self.wake_velocity = ti.Vector.field(3, dtype=dtype, shape=(max_n_wake_particles_per_step,))
        self.wake_vortex_strength = ti.Vector.field(
            3, dtype=dtype, shape=(max_n_wake_particles_per_step,)
        )
        self.wake_core_radius = ti.field(dtype=dtype, shape=(max_n_wake_particles_per_step,))
        self.wake_volume = ti.field(dtype=dtype, shape=(max_n_wake_particles_per_step,))
        self.wake_group_id = ti.field(dtype=ti.i32, shape=(max_n_wake_particles_per_step,))

        # Panel properties
        self.group_id = ti.field(dtype=ti.i32, shape=(max_n_panels,))

        # Atomic counter for number of wake particles in current buffer
        self.n_wake_particles = ti.field(dtype=ti.i32, shape=())

        # Maximum wake particles per step
        self._max_wake_per_step = max_n_wake_particles_per_step

    def reset_wake_buffer(self):
        """Reset wake particle buffer for new time step."""
        self.n_wake_particles[None] = 0

    @ti.kernel
    def mark_le_panels(self):
        """
        Mark all panels that have no upstream neighbor as Leading Edge panels.

        Must be called once after mesh generation (when neighbor_indices is
        populated) and after any topology change.  Result stored in is_leading_edge.
        """
        for i in range(self.n_panels):
            if self.neighbor_indices[i, 2] == -1:
                self.is_leading_edge[i] = 1
            else:
                self.is_leading_edge[i] = 0

    @ti.kernel
    def save_old_circulation(self):
        """Save current circulation as previous-step state."""
        for i in range(self.n_panels):
            self.circulation_old[i] = self.circulation[i]
            self.cumulative_circulation_old[i] = self.cumulative_circulation[i]

    @ti.kernel
    def apply_relaxation(self, relaxation_factor: float):
        """
        Apply under-relaxation to circulation: circulation = alpha * circulation_new + (1-alpha) * circulation_old.

        This stabilizes the VLM solution when coupled with strong wake influence.
        Called AFTER linear solve to blend new solution with previous values.

        Args:
            relaxation_factor: Relaxation factor in ``(0, 1]``.
                   1.0 = no relaxation (direct solve)
                   0.5 = 50% blend with previous step
        """
        for i in range(self.n_panels):
            circulation_new = self.circulation[i]
            circulation_old = self.circulation_old[i]
            self.circulation[i] = (
                relaxation_factor * circulation_new + (1.0 - relaxation_factor) * circulation_old
            )

    @ti.kernel
    def compute_cumulative_circulation(self):
        """
        Compute cumulative circulation for each panel by summing upstream.

        For trailing edge shedding, the bound circulation at each spanwise station
        is the sum of all chordwise panel circulations. This kernel walks upstream
        from each panel to compute this cumulative sum.

        Physical meaning:
        - circulation[i] = local vortex ring strength (per-panel)
        - cumulative_circulation[i] = sum of all upstream panels + self
          = bound circulation at this spanwise station for TE panels
        """
        for i in range(self.n_panels):
            # Start with this panel's circulation
            cumsum = self.circulation[i]

            # Walk upstream and accumulate
            current = i
            for _ in range(100):  # Max depth to prevent infinite loops
                upstream_idx = self.neighbor_indices[current, 2]  # Index 2 = upstream
                if upstream_idx == -1:
                    break  # Reached leading edge
                cumsum += self.circulation[upstream_idx]
                current = upstream_idx

            self.cumulative_circulation[i] = cumsum

    def get_wake_count(self) -> int:
        """Get number of wake particles in current buffer."""
        return self.n_wake_particles[None]

    def reset(self):
        """Reset lattice (clear all data)."""
        self.n_panels = 0
        self.circulation.fill(0.0)
        self.circulation_old.fill(0.0)
        self.smoothed_circulation.fill(0.0)
        self.cumulative_circulation.fill(0.0)
        self.cumulative_circulation_old.fill(0.0)
        self.aerodynamic_influence_coefficient.fill(0.0)
        self.right_hand_side.fill(0.0)
        self.right_hand_side.fill(0.0)
        self.kinematic_velocity.fill(0.0)
        self.external_velocity.fill(0.0)
        self.neighbor_indices.fill(-1)
        self.trailing_edge_index.fill(-1)
        self.is_trailing_edge.fill(0)
        self.is_leading_edge.fill(0)
        self.leading_edge_suction_parameter.fill(0.0)

    def get_collocation_points(self) -> np.ndarray:
        """Get collocation_point points as numpy array."""
        return self.collocation_point.to_numpy()[: self.n_panels]

    def get_circulation(self) -> np.ndarray:
        """Get circulation distribution as numpy array."""
        return self.circulation.to_numpy()[: self.n_panels]

    def get_velocity(self) -> np.ndarray:
        """Get velocity at collocation_point points as numpy array."""
        return self.velocity.to_numpy()[: self.n_panels]

    def get_bound_vortex_velocity(self) -> np.ndarray:
        """Get velocity at bound vortex midpoints as numpy array."""
        return self.bound_vortex_velocity.to_numpy()[: self.n_panels]

    def get_kinematic_velocity(self) -> np.ndarray:
        """Get kinematic velocity at collocation_point points as numpy array."""
        return self.kinematic_velocity.to_numpy()[: self.n_panels]

    def get_panel_centre(self) -> np.ndarray:
        """Get panel center position as numpy array."""
        return self.panel_centre.to_numpy()[: self.n_panels]

    def get_external_velocity(self) -> np.ndarray:
        """Get external velocity at collocation_point points as numpy array."""
        return self.external_velocity.to_numpy()[: self.n_panels]

    def get_pressure_coefficient(self) -> np.ndarray:
        """Get pressure coefficient as numpy array."""
        return self.pressure_coefficient.to_numpy()[: self.n_panels]

    # --- NumPy-based geometry update (avoids Taichi field dimension bugs) -----
    def translate_panels(self, displacement: np.ndarray, start_idx: int = 0, end_idx: int = None):
        """
        Translate panel geometry by *displacement* using NumPy (CPU).

        This is a robust alternative to the Taichi kernel
        ``update_geometry_translating_kernel`` that avoids field dimension
        issues when VPM and VLM Taichi fields coexist.

        Args:
            displacement: Translation vector [dx, dy, dz]
            start_idx: First panel index to update
            end_idx: One-past-last panel index (default: n_panels)
        """
        if end_idx is None:
            end_idx = self.n_panels
        if end_idx <= start_idx:
            return

        displacement = np.asarray(displacement, dtype=self.np_dtype)

        # Read → modify → write for 2D fields (panel_corner_position, vortex_point_position)
        corners_np = self.panel_corner_position.to_numpy().astype(self.np_dtype)
        vortex_np = self.vortex_point_position.to_numpy().astype(self.np_dtype)
        for j in range(4):
            corners_np[start_idx:end_idx, j] += displacement
            vortex_np[start_idx:end_idx, j] += displacement
        self.panel_corner_position.from_numpy(corners_np)
        self.vortex_point_position.from_numpy(vortex_np)

        # 1D vector fields
        coll_np = self.collocation_point.to_numpy().astype(self.np_dtype)
        coll_np[start_idx:end_idx] += displacement
        self.collocation_point.from_numpy(coll_np)

        bm_np = self.bound_vortex_midpoint.to_numpy().astype(self.np_dtype)
        bm_np[start_idx:end_idx] += displacement
        self.bound_vortex_midpoint.from_numpy(bm_np)

    def rotate_translate_panels(
        self,
        rotation_matrix: np.ndarray,
        origin: np.ndarray,
        displacement: np.ndarray,
        start_idx: int = 0,
        end_idx: int = None,
        update_normal: bool = True,
    ):
        """
        Rotate panels about *origin* then translate by *displacement* using NumPy.

        New position = rotation_matrix @ (old - origin) + origin + displacement

        Args:
            rotation_matrix: 3×3 rotation matrix
            origin: Centre of rotation [3]
            displacement: Translation after rotation [3]
            start_idx: First panel index
            end_idx: One-past-last panel index (default: n_panels)
            update_normal: If True, rotate normal as well
        """
        if end_idx is None:
            end_idx = self.n_panels
        if end_idx <= start_idx:
            return

        origin = np.asarray(origin, dtype=self.np_dtype)
        displacement = np.asarray(displacement, dtype=self.np_dtype)
        rotation_matrix = np.asarray(rotation_matrix, dtype=self.np_dtype)

        def _rotate(position):
            return (rotation_matrix @ (position - origin).T).T + origin + displacement

        corners_np = self.panel_corner_position.to_numpy().astype(self.np_dtype)
        vortex_np = self.vortex_point_position.to_numpy().astype(self.np_dtype)
        for j in range(4):
            corners_np[start_idx:end_idx, j] = _rotate(corners_np[start_idx:end_idx, j])
            vortex_np[start_idx:end_idx, j] = _rotate(vortex_np[start_idx:end_idx, j])
        self.panel_corner_position.from_numpy(corners_np)
        self.vortex_point_position.from_numpy(vortex_np)

        coll_np = self.collocation_point.to_numpy().astype(self.np_dtype)
        coll_np[start_idx:end_idx] = _rotate(coll_np[start_idx:end_idx])
        self.collocation_point.from_numpy(coll_np)

        bm_np = self.bound_vortex_midpoint.to_numpy().astype(self.np_dtype)
        bm_np[start_idx:end_idx] = _rotate(bm_np[start_idx:end_idx])
        self.bound_vortex_midpoint.from_numpy(bm_np)

        if update_normal:
            normal = self.normal.to_numpy().astype(self.np_dtype)
            normal[start_idx:end_idx] = (rotation_matrix @ normal[start_idx:end_idx].T).T
            self.normal.from_numpy(normal)

    def has_kinematic_velocity(self) -> bool:
        """
        Check if kinematic velocity has been set.

        Returns True if any panel has non-zero kinematic velocity.
        """
        if self.kinematic_velocity is None:
            return False
        kinematic_velocity = self.kinematic_velocity.to_numpy()[: self.n_panels]
        return np.any(np.abs(kinematic_velocity) > 1e-10)

    def set_kinematic_velocity(self, kinematic_velocity: np.ndarray):
        """
        Assign a pre-computed kinematic velocity field from a numpy array.
        (Nx3)
        """
        if kinematic_velocity is None:
            return

        # Validate shape
        if kinematic_velocity.shape[0] != self.n_panels:
            # If only 3 components, broadcast to all panels
            if kinematic_velocity.shape == (3,):
                kinematic_velocity = np.tile(kinematic_velocity, (self.n_panels, 1))
            else:
                # We can be lenient and just use the provided slice if n_panels is smaller
                n = min(kinematic_velocity.shape[0], self.n_panels)
                kinematic_velocity = kinematic_velocity[:n]

        # Ensure correct shape (N, 3)
        if kinematic_velocity.ndim == 1:
            kinematic_velocity = kinematic_velocity.reshape(-1, 3)

        # Use from_numpy for better stability than custom kernels for simple transfers
        full_kinematic_velocity = np.zeros((self.max_n_panels, 3), dtype=self.np_dtype)
        n = min(kinematic_velocity.shape[0], self.n_panels)
        full_kinematic_velocity[:n] = kinematic_velocity[:n]

        self.kinematic_velocity.from_numpy(full_kinematic_velocity)

    def set_external_velocity(self, external_velocity: np.ndarray):
        """
        Set external velocity at each panel.

        Args:
            external_velocity: Velocity array (N x 3)
        """
        if external_velocity.shape[0] < self.n_panels:
            raise ValueError(
                f"Expected at least {self.n_panels} velocity, got {external_velocity.shape[0]}"
            )

        # Ensure correct shape and dtype
        external_velocity = np.ascontiguousarray(
            external_velocity[: self.n_panels], dtype=self.np_dtype
        )
        if external_velocity.ndim != 2 or external_velocity.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {external_velocity.shape}")

        # Efficient batch copy using from_numpy
        full_external_velocity = np.zeros((self.external_velocity.shape[0], 3), dtype=self.np_dtype)
        full_external_velocity[: self.n_panels] = external_velocity
        self.external_velocity.from_numpy(full_external_velocity)

    # Removed duplicate get_pressure_coefficient method that used non-existent pressure-coefficient field

    def get_forces(self) -> np.ndarray:
        """Get panel forces as numpy array."""
        return self.panel_force.to_numpy()[: self.n_panels]

    def get_aerodynamic_influence_coefficient_matrix(self) -> np.ndarray:
        """Get aerodynamic_influence_coefficient matrix as numpy array."""
        n = self.n_panels
        return self.aerodynamic_influence_coefficient.to_numpy()[:n, :n]

    def set_circulation(self, circulation: np.ndarray) -> None:
        """Upload bound circulation values to the device.

        Parameters
        ----------
        circulation
            One value per panel, in square metres per second (m²/s). The input
            may have any shape but must contain exactly ``n_panels`` values.

        Raises
        ------
        ValueError
            If the input does not contain one value per panel.
        """
        circulation = np.asarray(circulation, dtype=self.np_dtype).reshape(-1)
        if circulation.size != self.n_panels:
            raise ValueError(f"Expected {self.n_panels} values, got {circulation.size}")

        circulation_full = np.zeros(self.circulation.shape[0], dtype=self.np_dtype)
        circulation_full[: self.n_panels] = circulation
        self.circulation.from_numpy(circulation_full)

    @ti.kernel
    def compute_panel_centre(self):
        """Compute each panel centre as the mean of its corner positions."""
        for i in range(self.n_panels):
            curr_pos = ti.Vector([0.0, 0.0, 0.0])
            for k in range(4):
                curr_pos += self.panel_corner_position[i, k]
            self.panel_centre[i] = curr_pos * 0.25

    def save_vtk(self, filename: str, time: float = 0.0):
        """
        Save lattice to VTK file for visualization.

        Args:
            filename: Output filename (without extension)
            time: Physical simulation time stored in the output field.
        """
        # Ensure all Taichi operations are complete before reading fields
        ti.sync()

        try:
            from vtk import (
                vtkCellArray,
                vtkDoubleArray,
                vtkPoints,
                vtkPolyData,
                vtkQuad,
                vtkXMLPolyDataWriter,
            )
            from vtk.util import numpy_support
        except ImportError:
            print("Warning: VTK not available. Cannot save lattice.")
            return

        # Ensure derived fields are computed
        self.compute_panel_centre()

        points = vtkPoints()
        cells = vtkCellArray()

        # Add panel panel_corner_position
        corners_np = self.panel_corner_position.to_numpy()[: self.n_panels]

        for panel_idx in range(self.n_panels):
            # Add 4 panel_corner_position
            p_ids = []
            for corner_idx in range(4):
                pt = corners_np[panel_idx, corner_idx]
                p_id = points.InsertNextPoint(pt[0], pt[1], pt[2])
                p_ids.append(p_id)

            # Create quad cell
            quad = vtkQuad()
            for i, p_id in enumerate(p_ids):
                quad.GetPointIds().SetId(i, p_id)
            cells.InsertNextCell(quad)

        # Create polydata
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        # Add scalar fields
        N = self.n_panels

        # -- Geometry ---------------------------------------------------------

        # Panel Areas
        areas_np = self.area.to_numpy()[:N].reshape(-1, 1)
        areas_vtk = numpy_support.numpy_to_vtk(areas_np.ravel())
        _set_vtk_array_name(areas_vtk, "area")
        polydata.GetCellData().AddArray(areas_vtk)

        # Panel Normal Vectors (unit normal)
        normals_np = self.normal.to_numpy()[:N]
        normals_vtk = numpy_support.numpy_to_vtk(normals_np)
        _set_vtk_array_name(normals_vtk, "normal")
        normals_vtk.SetNumberOfComponents(3)
        polydata.GetCellData().AddArray(normals_vtk)

        # Panel centre (geometric average of the four corner positions).
        pos_np = self.panel_centre.to_numpy()[:N]
        pos_vtk = numpy_support.numpy_to_vtk(pos_np)
        _set_vtk_array_name(pos_vtk, "panel_centre")
        pos_vtk.SetNumberOfComponents(3)
        polydata.GetCellData().AddArray(pos_vtk)

        # Panel chord length Δc = |TE_mid − LE_mid|  (scalar, metres)
        te_mid = 0.5 * (corners_np[:, 3] + corners_np[:, 2])  # (S + R) / 2
        le_mid = 0.5 * (corners_np[:, 0] + corners_np[:, 1])  # (P + Q) / 2
        panel_chord = np.linalg.norm(te_mid - le_mid, axis=1)
        pc_vtk = numpy_support.numpy_to_vtk(panel_chord)
        _set_vtk_array_name(pc_vtk, "panel_chord")
        polydata.GetCellData().AddArray(pc_vtk)

        # Bound-vortex leg l = V3 − V2 at 25% chord (root→tip vector)
        vp_np = self.vortex_point_position.to_numpy()[:N]
        bound_leg = vp_np[:, 2] - vp_np[:, 1]  # V3 − V2
        bl_vtk = numpy_support.numpy_to_vtk(bound_leg)
        _set_vtk_array_name(bl_vtk, "bound_vortex_leg")
        bl_vtk.SetNumberOfComponents(3)
        polydata.GetCellData().AddArray(bl_vtk)

        # Trailing-edge / leading-edge flags (integer, 0 or 1)
        is_te_np = self.is_trailing_edge.to_numpy()[:N].astype(np.int32)
        is_te_vtk = numpy_support.numpy_to_vtk(is_te_np)
        _set_vtk_array_name(is_te_vtk, "is_trailing_edge")
        polydata.GetCellData().AddArray(is_te_vtk)

        is_le_np = self.is_leading_edge.to_numpy()[:N].astype(np.int32)
        is_le_vtk = numpy_support.numpy_to_vtk(is_le_np)
        _set_vtk_array_name(is_le_vtk, "is_leading_edge")
        polydata.GetCellData().AddArray(is_le_vtk)

        # -- Circulation -------------------------------------------------------

        # Per-panel (horseshoe) circulation Γ  — used directly in K-J
        circulation_np = self.circulation.to_numpy()[:N]
        circulation_vtk = numpy_support.numpy_to_vtk(circulation_np)
        _set_vtk_array_name(circulation_vtk, "circulation")
        polydata.GetCellData().AddArray(circulation_vtk)

        # -- Velocity fields ---------------------------------------------------

        # Velocity at bound-vortex midpoints including wake-induced downwash.
        # This is the velocity used in the Kutta-Joukowski force kernel.
        bv_np = self.bound_vortex_velocity.to_numpy()[:N]
        bv_vtk = numpy_support.numpy_to_vtk(bv_np)
        _set_vtk_array_name(bv_vtk, "bound_vortex_velocity")
        bv_vtk.SetNumberOfComponents(3)
        polydata.GetCellData().AddArray(bv_vtk)

        # -- Pressure ----------------------------------------------------------

        # pressure_jump_coefficient: pressure-jump coefficient across the panel surface.
        #   ΔCp = 2Γ / (V∞ · Δc)
        # where Δc is the panel chord length (TE_mid − LE_mid) and V∞ is the
        # freestream speed taken from the magnitude of the kinematic velocity
        # (body velocity = −V∞ for a translating wing).
        # This IS the aerodynamically meaningful surface pressure coefficient
        # and can be integrated directly:  f_i = q∞ · ΔCp_i · A_i · n̂_i
        #
        # NOTE: bound_vortex_velocity is the INDUCED-ONLY velocity (no freestream).  It
        # must NOT be used to estimate V∞ — use kinematic_velocity instead.
        kin_np = self.kinematic_velocity.to_numpy()[:N]
        kin_mag = np.linalg.norm(kin_np, axis=1)
        freestream_speed = (
            float(np.median(kin_mag[kin_mag > 1e-8])) if kin_mag.max() > 1e-8 else 1.0
        )
        denom = freestream_speed * panel_chord
        delta_cp = np.where(denom > 1e-15, 2.0 * circulation_np / denom, 0.0)
        dcp_vtk = numpy_support.numpy_to_vtk(delta_cp)
        _set_vtk_array_name(dcp_vtk, "pressure_jump_coefficient")
        polydata.GetCellData().AddArray(dcp_vtk)

        # -- Per-panel forces (as computed by the solver) ----------------------

        # Total panel force F = F_KJ + F_unsteady  (ground-truth for cross-check)
        forces_np = self.panel_force.to_numpy()[:N]
        forces_vtk = numpy_support.numpy_to_vtk(forces_np)
        _set_vtk_array_name(forces_vtk, "panel_force")
        forces_vtk.SetNumberOfComponents(3)
        polydata.GetCellData().AddArray(forces_vtk)

        # Add physical time for visualization synchronization.
        time_array = vtkDoubleArray()
        _set_vtk_array_name(time_array, "time")
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, time)
        polydata.GetFieldData().AddArray(time_array)

        # Write to file
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(f"{filename}.vtp")
        writer.SetInputData(polydata)
        writer.Write()
