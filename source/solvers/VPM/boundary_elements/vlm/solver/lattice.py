"""
GPU-resident VLM lattice data structure (VLMLattice): panel geometry,
circulations, and topology.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
import taichi as ti


@ti.data_oriented
class VLMLattice:
    """
    Taichi-native VLM lattice data structure.

    Stores panel geometry, horseshoe vortex points, and solution data.
    All fields are pre-allocated for efficient GPU computation.

    Attributes:
        max_panels: Maximum number of panels
        num_panels: Current number of active panels
        dtype: Taichi data type (ti.f32 or ti.f64)
    """

    def __init__(self, max_panels: int = 5000, dtype=ti.f32):
        """
        Initialize VLM lattice.

        Args:
            max_panels: Maximum number of panels to allocate
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
        self.max_panels = int(max_panels)
        self.dtype = dtype
        self.np_dtype = np.float32 if dtype == ti.f32 else np.float64
        self.num_panels = 0
        print(f"VLMLattice initialized with max_panels={self.max_panels}")

        # Panel corner points (N x 4 x 3)
        # Order: [P, Q, R, S] where P-Q is LE, R-S is TE
        #   P ------- Q
        #   |         |
        #   S ------- R
        self.corners = ti.Vector.field(3, dtype=dtype, shape=(max_panels, 4))

        # Horseshoe vortex points (N x 4 x 3)
        # Order: [V1, V2, V3, V4]
        #   V1 = left trailing leg far endpoint (upstream infinity)
        #   V2 = bound leg left endpoint (at 25% chord)
        #   V3 = bound leg right endpoint (at 25% chord)
        #   V4 = right trailing leg far endpoint (upstream infinity)
        self.vortex_points = ti.Vector.field(3, dtype=dtype, shape=(max_panels, 4))

        # Collocation points (N x 3) - at 75% chord
        self.collocation = ti.Vector.field(3, dtype=dtype, shape=(max_panels,))

        # Panel normals (N x 3) - unit vectors
        self.normals = ti.Vector.field(3, dtype=dtype, shape=(max_panels,))

        # Panel areas (N,)
        self.areas = ti.field(dtype=dtype, shape=(max_panels,))

        # Bound leg midpoints (N x 3) - for force calculation
        self.bound_midpoints = ti.Vector.field(3, dtype=dtype, shape=(max_panels,))

        # Trailing edge direction vectors (N x 2 x 3)
        # [0] = left trailing direction, [1] = right trailing direction
        self.trailing_dirs = ti.Vector.field(3, dtype=dtype, shape=(max_panels, 2))

        # Solution: circulation strength Γ (N,) - matching VPM convention
        # This is the PER-PANEL circulation (local vortex ring strength)
        self.circulation = ti.field(dtype=dtype, shape=(max_panels,))

        # Previous circulation strength Γ_old (N,) - for wake shedding (dGamma/dt)
        self.circulation_old = ti.field(dtype=dtype, shape=(max_panels,))

        # Time-averaged circulation 0.5*(Γ + Γ_old) used by kj_smoothing to
        # compute bound_velocity so V_bound and the force kernel use the same
        # smoothed gamma (eliminates the 2Δt oscillation in KJ forces).
        self.circulation_smooth = ti.field(dtype=dtype, shape=(max_panels,))

        # Cumulative circulation at each panel (sum of all upstream panels)
        # For TE panels: Γ_cumulative = Σ γ_i along chordwise direction
        # This is what should be used for trailing vortex shedding
        self.cumulative_circulation = ti.field(dtype=dtype, shape=(max_panels,))

        # Previous cumulative circulation (for delta-shedding)
        self.cumulative_circulation_old = ti.field(dtype=dtype, shape=(max_panels,))

        # Aerodynamic influence coefficient (AIC) matrix (N x N)
        # AIC[i,j] = downwash at panel i due to unit circulation on panel j
        self.AIC = ti.field(dtype=dtype, shape=(max_panels, max_panels))

        # Right-hand side (boundary condition) (N,)
        self.rhs = ti.field(dtype=dtype, shape=(max_panels,))

        # Velocity at collocation points (N x 3)
        self.velocity = ti.Vector.field(3, dtype=dtype, shape=(max_panels,))

        # Velocity at bound vortex midpoints (N x 3) - for correct K-J force
        self.bound_velocity = ti.Vector.field(3, dtype=dtype, shape=(max_panels,))

        # Kinematic velocity at collocation points (N x 3) - 2D scalar field for better stability
        self.kinematic_velocity = ti.field(dtype=dtype, shape=(max_panels, 3))

        # Pressure coefficient (N,) - matching VPM convention
        self.pressure_coefficient = ti.field(dtype=dtype, shape=(max_panels,))

        # Panel forces (N x 3)
        self.forces = ti.Vector.field(3, dtype=dtype, shape=(max_panels,))

        # External velocity field at collocation points (N x 3) - matching VPM convention
        self.external_velocity = ti.Vector.field(3, dtype=dtype, shape=(max_panels,))

        # Panel center positions (N x 3) - for visualization/export
        self.position = ti.Vector.field(3, dtype=dtype, shape=(max_panels,))

        # Bookkeeping: which wing/segment each panel belongs to
        self.panel_wing_id = ti.field(dtype=ti.i32, shape=(max_panels,))
        self.panel_segment_id = ti.field(dtype=ti.i32, shape=(max_panels,))
        self.panel_is_mirrored = ti.field(dtype=ti.i32, shape=(max_panels,))

        # Topology Connectivity (N x 4)
        # [0]=Left, [1]=Right, [2]=Upstream, [3]=Downstream
        # Value -1 indicates no neighbor (e.g., edge or tip)
        self.neighbor_indices = ti.field(dtype=ti.i32, shape=(max_panels, 4))

        # Index of the trailing-edge panel in each panel's own chordwise strip.
        # For every panel this points to the TE panel (i == n_chord-1) that shares
        # its spanwise station. Used by the coupled-mode influence kernels so each
        # panel's internal trailing legs run from its bound vortex all the way to
        # the WING trailing edge (where the VPM wake takes over), rather than only
        # to the panel's own downstream edge. The latter truncation breaks the
        # chordwise accumulation of trailing vorticity and collapses the
        # finite-wing downwash (flat, non-tapering spanwise loading). Stored as an
        # index (not a position) so it tracks corner motion automatically.
        self.te_index = ti.field(dtype=ti.i32, shape=(max_panels,))

        # Flags
        # 1 if panel is at Trailing Edge (no downstream neighbor), 0 otherwise
        self.is_TE_panel = ti.field(dtype=ti.i32, shape=(max_panels,))

        # 1 if panel is at Leading Edge (no upstream neighbor), 0 otherwise
        # Static topology — set once after mesh generation via mark_le_panels()
        self.is_LE_panel = ti.field(dtype=ti.i32, shape=(max_panels,))

        # Leading Edge Suction Parameter (per LE panel, recomputed every step)
        # Convention: LESP = |Γ_panel| / (chord_ref * V_ref)  (dimensionless, ≥ 0)
        # Shedding triggers when LESP > lesp_crit (Ramesh et al. 2014)
        self.lesp = ti.field(dtype=dtype, shape=(max_panels,))

        # WAKE PARTICLE BUFFER (for direct Taichi-to-VPM transfer)
        # Pre-allocated buffer for wake particles shed per time step.
        # Avoids numpy intermediate for GPU-to-GPU transfer.
        # Factor 3×: longitudinal (left+right) + transverse per TE panel,
        # plus headroom for LEV particles when enabled.
        max_wake_per_step = 3 * self.max_panels
        # Particle properties (for wake shedding)
        self.wake_positions = ti.Vector.field(3, dtype=dtype, shape=(max_wake_per_step,))
        self.wake_velocities = ti.Vector.field(3, dtype=dtype, shape=(max_wake_per_step,))
        self.wake_strengths = ti.Vector.field(3, dtype=dtype, shape=(max_wake_per_step,))
        self.wake_radii = ti.field(dtype=dtype, shape=(max_wake_per_step,))
        self.wake_volumes = ti.field(dtype=dtype, shape=(max_wake_per_step,))
        self.wake_group_ids = ti.field(dtype=ti.i32, shape=(max_wake_per_step,))

        # Panel properties
        self.panel_group_id = ti.field(dtype=ti.i32, shape=(max_panels,))

        # Atomic counter for number of wake particles in current buffer
        self.num_wake_particles = ti.field(dtype=ti.i32, shape=())

        # Maximum wake particles per step
        self._max_wake_per_step = max_wake_per_step

    def reset_wake_buffer(self):
        """Reset wake particle buffer for new time step."""
        self.num_wake_particles[None] = 0

    @ti.kernel
    def mark_le_panels(self):
        """
        Mark all panels that have no upstream neighbor as Leading Edge panels.

        Must be called once after mesh generation (when neighbor_indices is
        populated) and after any topology change.  Result stored in is_LE_panel.
        """
        for i in range(self.num_panels):
            if self.neighbor_indices[i, 2] == -1:
                self.is_LE_panel[i] = 1
            else:
                self.is_LE_panel[i] = 0

    @ti.kernel
    def save_old_circulation(self):
        """Save current circulation as previous-step state."""
        for i in range(self.num_panels):
            self.circulation_old[i] = self.circulation[i]
            self.cumulative_circulation_old[i] = self.cumulative_circulation[i]

    @ti.kernel
    def apply_relaxation(self, alpha: float):
        """
        Apply under-relaxation to circulation: gamma = alpha * gamma_new + (1-alpha) * gamma_old.

        This stabilizes the VLM solution when coupled with strong wake influence.
        Called AFTER linear solve to blend new solution with previous values.

        Args:
            alpha: Relaxation factor (0 < alpha <= 1.0)
                   1.0 = no relaxation (direct solve)
                   0.5 = 50% blend with previous step
        """
        for i in range(self.num_panels):
            gamma_new = self.circulation[i]
            gamma_old = self.circulation_old[i]
            self.circulation[i] = alpha * gamma_new + (1.0 - alpha) * gamma_old

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
        for i in range(self.num_panels):
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
        return self.num_wake_particles[None]

    def reset(self):
        """Reset lattice (clear all data)."""
        self.num_panels = 0
        self.circulation.fill(0.0)
        self.circulation_old.fill(0.0)
        self.circulation_smooth.fill(0.0)
        self.cumulative_circulation.fill(0.0)
        self.cumulative_circulation_old.fill(0.0)
        self.AIC.fill(0.0)
        self.rhs.fill(0.0)
        self.rhs.fill(0.0)
        self.kinematic_velocity.fill(0.0)
        self.external_velocity.fill(0.0)
        self.neighbor_indices.fill(-1)
        self.te_index.fill(-1)
        self.is_TE_panel.fill(0)
        self.is_LE_panel.fill(0)
        self.lesp.fill(0.0)

    def get_collocation_points(self) -> np.ndarray:
        """Get collocation points as numpy array."""
        return self.collocation.to_numpy()[: self.num_panels]

    def get_circulation(self) -> np.ndarray:
        """Get circulation distribution as numpy array."""
        return self.circulation.to_numpy()[: self.num_panels]

    def get_velocity(self) -> np.ndarray:
        """Get velocity at collocation points as numpy array."""
        return self.velocity.to_numpy()[: self.num_panels]

    def get_bound_velocity(self) -> np.ndarray:
        """Get velocity at bound vortex midpoints as numpy array."""
        return self.bound_velocity.to_numpy()[: self.num_panels]

    def get_kinematic_velocity(self) -> np.ndarray:
        """Get kinematic velocity at collocation points as numpy array."""
        return self.kinematic_velocity.to_numpy()[: self.num_panels]

    def get_position(self) -> np.ndarray:
        """Get panel center positions as numpy array."""
        return self.position.to_numpy()[: self.num_panels]

    def get_external_velocity(self) -> np.ndarray:
        """Get external velocity at collocation points as numpy array."""
        return self.external_velocity.to_numpy()[: self.num_panels]

    def get_pressure_coefficient(self) -> np.ndarray:
        """Get pressure coefficient as numpy array."""
        return self.pressure_coefficient.to_numpy()[: self.num_panels]

    # --- NumPy-based geometry update (avoids Taichi field dimension bugs) -----
    def translate_panels(self, dX: np.ndarray, start_idx: int = 0, end_idx: int = None):
        """
        Translate panel geometry by dX using NumPy (CPU).

        This is a robust alternative to the Taichi kernel
        ``update_geometry_translating_kernel`` that avoids field dimension
        issues when VPM and VLM Taichi fields coexist.

        Args:
            dX: Translation vector [dx, dy, dz]
            start_idx: First panel index to update
            end_idx: One-past-last panel index (default: num_panels)
        """
        if end_idx is None:
            end_idx = self.num_panels
        if end_idx <= start_idx:
            return

        dX = np.asarray(dX, dtype=self.np_dtype)
        end_idx - start_idx

        # Read → modify → write for 2D fields (corners, vortex_points)
        corners_np = self.corners.to_numpy().astype(self.np_dtype)
        vortex_np = self.vortex_points.to_numpy().astype(self.np_dtype)
        for j in range(4):
            corners_np[start_idx:end_idx, j] += dX
            vortex_np[start_idx:end_idx, j] += dX
        self.corners.from_numpy(corners_np)
        self.vortex_points.from_numpy(vortex_np)

        # 1D vector fields
        coll_np = self.collocation.to_numpy().astype(self.np_dtype)
        coll_np[start_idx:end_idx] += dX
        self.collocation.from_numpy(coll_np)

        bm_np = self.bound_midpoints.to_numpy().astype(self.np_dtype)
        bm_np[start_idx:end_idx] += dX
        self.bound_midpoints.from_numpy(bm_np)

    def rotate_translate_panels(
        self,
        R: np.ndarray,
        origin: np.ndarray,
        dX: np.ndarray,
        start_idx: int = 0,
        end_idx: int = None,
        update_normals: bool = True,
    ):
        """
        Rotate panels about *origin* then translate by *dX* using NumPy.

        New position = R @ (old - origin) + origin + dX

        Args:
            R: 3×3 rotation matrix
            origin: Centre of rotation [3]
            dX: Translation after rotation [3]
            start_idx: First panel index
            end_idx: One-past-last panel index (default: num_panels)
            update_normals: If True, rotate normals as well
        """
        if end_idx is None:
            end_idx = self.num_panels
        if end_idx <= start_idx:
            return

        origin = np.asarray(origin, dtype=self.np_dtype)
        dX = np.asarray(dX, dtype=self.np_dtype)
        R = np.asarray(R, dtype=self.np_dtype)

        def _rotate(pts):
            return (R @ (pts - origin).T).T + origin + dX

        corners_np = self.corners.to_numpy().astype(self.np_dtype)
        vortex_np = self.vortex_points.to_numpy().astype(self.np_dtype)
        for j in range(4):
            corners_np[start_idx:end_idx, j] = _rotate(corners_np[start_idx:end_idx, j])
            vortex_np[start_idx:end_idx, j] = _rotate(vortex_np[start_idx:end_idx, j])
        self.corners.from_numpy(corners_np)
        self.vortex_points.from_numpy(vortex_np)

        coll_np = self.collocation.to_numpy().astype(self.np_dtype)
        coll_np[start_idx:end_idx] = _rotate(coll_np[start_idx:end_idx])
        self.collocation.from_numpy(coll_np)

        bm_np = self.bound_midpoints.to_numpy().astype(self.np_dtype)
        bm_np[start_idx:end_idx] = _rotate(bm_np[start_idx:end_idx])
        self.bound_midpoints.from_numpy(bm_np)

        if update_normals:
            normals_np = self.normals.to_numpy().astype(self.np_dtype)
            normals_np[start_idx:end_idx] = (R @ normals_np[start_idx:end_idx].T).T
            self.normals.from_numpy(normals_np)

    def has_kinematic_velocity(self) -> bool:
        """
        Check if kinematic velocity has been set.

        Returns True if any panel has non-zero kinematic velocity.
        """
        if self.kinematic_velocity is None:
            return False
        V_kin = self.kinematic_velocity.to_numpy()[: self.num_panels]
        return np.any(np.abs(V_kin) > 1e-10)

    def set_kinematic_velocity(self, V_kin: np.ndarray):
        """
        Assign a pre-computed kinematic velocity field from a numpy array.
        (Nx3)
        """
        if V_kin is None:
            return

        # Validate shape
        if V_kin.shape[0] != self.num_panels:
            # If only 3 components, broadcast to all panels
            if V_kin.shape == (3,):
                V_kin = np.tile(V_kin, (self.num_panels, 1))
            else:
                # We can be lenient and just use the provided slice if n_panels is smaller
                n = min(V_kin.shape[0], self.num_panels)
                V_kin = V_kin[:n]

        # Ensure correct shape (N, 3)
        if V_kin.ndim == 1:
            V_kin = V_kin.reshape(-1, 3)

        # Use from_numpy for better stability than custom kernels for simple transfers
        V_full = np.zeros((self.max_panels, 3), dtype=self.np_dtype)
        n = min(V_kin.shape[0], self.num_panels)
        V_full[:n] = V_kin[:n]

        self.kinematic_velocity.from_numpy(V_full)

    def set_external_velocity(self, V_ext: np.ndarray):
        """
        Set external velocity at each panel.

        Args:
            V_ext: Velocity array (N x 3)
        """
        if V_ext.shape[0] < self.num_panels:
            raise ValueError(
                f"Expected at least {self.num_panels} velocities, got {V_ext.shape[0]}"
            )

        # Ensure correct shape and dtype
        V_ext = np.ascontiguousarray(V_ext[: self.num_panels], dtype=self.np_dtype)
        if V_ext.ndim != 2 or V_ext.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {V_ext.shape}")

        # Efficient batch copy using from_numpy
        V_full = np.zeros((self.external_velocity.shape[0], 3), dtype=self.np_dtype)
        V_full[: self.num_panels] = V_ext
        self.external_velocity.from_numpy(V_full)

    # Removed duplicate get_pressure_coefficient method that used non-existent self.Cp

    def get_forces(self) -> np.ndarray:
        """Get panel forces as numpy array."""
        return self.forces.to_numpy()[: self.num_panels]

    def get_AIC_matrix(self) -> np.ndarray:
        """Get AIC matrix as numpy array."""
        n = self.num_panels
        return self.AIC.to_numpy()[:n, :n]

    def set_circulation(self, gamma: np.ndarray) -> None:
        """Upload bound circulation values to the device.

        Parameters
        ----------
        gamma
            One value per panel, in square metres per second (m²/s). The input
            may have any shape but must contain exactly ``num_panels`` values.

        Raises
        ------
        ValueError
            If the input does not contain one value per panel.
        """
        gamma = np.asarray(gamma, dtype=self.np_dtype).reshape(-1)
        if gamma.size != self.num_panels:
            raise ValueError(f"Expected {self.num_panels} values, got {gamma.size}")

        gamma_full = np.zeros(self.circulation.shape[0], dtype=self.np_dtype)
        gamma_full[: self.num_panels] = gamma
        self.circulation.from_numpy(gamma_full)

    @ti.kernel
    def compute_panel_centers(self):
        """Compute panel centers (centroid of corners)."""
        for i in range(self.num_panels):
            curr_pos = ti.Vector([0.0, 0.0, 0.0])
            for k in range(4):
                curr_pos += self.corners[i, k]
            self.position[i] = curr_pos * 0.25

    def save_vtk(self, filename: str, flow_time: float = 0.0):
        """
        Save lattice to VTK file for visualization.

        Args:
            filename: Output filename (without extension)
            flow_time: Simulation time for Paraview TimeValue
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
        self.compute_panel_centers()

        points = vtkPoints()
        cells = vtkCellArray()

        # Add panel corners
        corners_np = self.corners.to_numpy()[: self.num_panels]

        for panel_idx in range(self.num_panels):
            # Add 4 corners
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
        N = self.num_panels

        # -- Geometry ---------------------------------------------------------

        # Panel Areas
        areas_np = self.areas.to_numpy()[:N].reshape(-1, 1)
        areas_vtk = numpy_support.numpy_to_vtk(areas_np.ravel())
        areas_vtk.SetName("Area")
        polydata.GetCellData().AddArray(areas_vtk)

        # Panel Normal Vectors (unit normals)
        normals_np = self.normals.to_numpy()[:N]
        normals_vtk = numpy_support.numpy_to_vtk(normals_np)
        normals_vtk.SetName("Normal")
        normals_vtk.SetNumberOfComponents(3)
        polydata.GetCellData().AddArray(normals_vtk)

        # Panel centroid (geometric average of the 4 quad corners)
        pos_np = self.position.to_numpy()[:N]
        pos_vtk = numpy_support.numpy_to_vtk(pos_np)
        pos_vtk.SetName("PanelCenter")
        pos_vtk.SetNumberOfComponents(3)
        polydata.GetCellData().AddArray(pos_vtk)

        # Panel chord length Δc = |TE_mid − LE_mid|  (scalar, metres)
        te_mid = 0.5 * (corners_np[:, 3] + corners_np[:, 2])  # (S + R) / 2
        le_mid = 0.5 * (corners_np[:, 0] + corners_np[:, 1])  # (P + Q) / 2
        panel_chord = np.linalg.norm(te_mid - le_mid, axis=1)
        pc_vtk = numpy_support.numpy_to_vtk(panel_chord)
        pc_vtk.SetName("PanelChord")
        polydata.GetCellData().AddArray(pc_vtk)

        # Bound-vortex leg l = V3 − V2 at 25% chord (root→tip vector)
        vp_np = self.vortex_points.to_numpy()[:N]
        bound_leg = vp_np[:, 2] - vp_np[:, 1]  # V3 − V2
        bl_vtk = numpy_support.numpy_to_vtk(bound_leg)
        bl_vtk.SetName("BoundLeg")
        bl_vtk.SetNumberOfComponents(3)
        polydata.GetCellData().AddArray(bl_vtk)

        # Trailing-edge / leading-edge flags (integer, 0 or 1)
        is_te_np = self.is_TE_panel.to_numpy()[:N].astype(np.int32)
        is_te_vtk = numpy_support.numpy_to_vtk(is_te_np)
        is_te_vtk.SetName("IsTE")
        polydata.GetCellData().AddArray(is_te_vtk)

        is_le_np = self.is_LE_panel.to_numpy()[:N].astype(np.int32)
        is_le_vtk = numpy_support.numpy_to_vtk(is_le_np)
        is_le_vtk.SetName("IsLE")
        polydata.GetCellData().AddArray(is_le_vtk)

        # -- Circulation -------------------------------------------------------

        # Per-panel (horseshoe) circulation Γ  — used directly in K-J
        gamma_np = self.circulation.to_numpy()[:N]
        gamma_vtk = numpy_support.numpy_to_vtk(gamma_np)
        gamma_vtk.SetName("Circulation")
        polydata.GetCellData().AddArray(gamma_vtk)

        # -- Velocity fields ---------------------------------------------------

        # Velocity at bound-vortex midpoints including wake-induced downwash.
        # This is the V used in the actual K-J force kernel:  F = ρΓ(V×l).
        # REPLACES the old "KinematicVelocity" which was identical for all
        # panels and contained no wake information.
        bv_np = self.bound_velocity.to_numpy()[:N]
        bv_vtk = numpy_support.numpy_to_vtk(bv_np)
        bv_vtk.SetName("BoundVelocity")
        bv_vtk.SetNumberOfComponents(3)
        polydata.GetCellData().AddArray(bv_vtk)

        # -- Pressure ----------------------------------------------------------

        # DeltaCp: pressure-jump coefficient across the panel surface.
        #   ΔCp = 2Γ / (V∞ · Δc)
        # where Δc is the panel chord length (TE_mid − LE_mid) and V∞ is the
        # freestream speed taken from the magnitude of the kinematic velocity
        # (body velocity = −V∞ for a translating wing).
        # This IS the aerodynamically meaningful surface pressure coefficient
        # and can be integrated directly:  f_i = q∞ · ΔCp_i · A_i · n̂_i
        #
        # NOTE: the old "PressureCoefficient" field (Cp = 1−|V_induced|²/V∞²)
        # was computed from induced-only velocities at collocation points and
        # was NOT ΔCp; it has been removed to avoid confusion.
        #
        # NOTE: bound_velocity is the INDUCED-ONLY velocity (no freestream).  It
        # must NOT be used to estimate V∞ — use kinematic_velocity instead.
        kin_np = self.kinematic_velocity.to_numpy()[:N]
        kin_mag = np.linalg.norm(kin_np, axis=1)
        V_inf_mag = float(np.median(kin_mag[kin_mag > 1e-8])) if kin_mag.max() > 1e-8 else 1.0
        denom = V_inf_mag * panel_chord
        delta_cp = np.where(denom > 1e-15, 2.0 * gamma_np / denom, 0.0)
        dcp_vtk = numpy_support.numpy_to_vtk(delta_cp)
        dcp_vtk.SetName("DeltaCp")
        polydata.GetCellData().AddArray(dcp_vtk)

        # -- Per-panel forces (as computed by the solver) ----------------------

        # Total panel force F = F_KJ + F_unsteady  (ground-truth for cross-check)
        forces_np = self.forces.to_numpy()[:N]
        forces_vtk = numpy_support.numpy_to_vtk(forces_np)
        forces_vtk.SetName("PanelForce")
        forces_vtk.SetNumberOfComponents(3)
        polydata.GetCellData().AddArray(forces_vtk)

        # Add TimeValue for Paraview synchronization
        time_array = vtkDoubleArray()
        time_array.SetName("TimeValue")
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, flow_time)
        polydata.GetFieldData().AddArray(time_array)

        # Write to file
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(f"{filename}.vtp")
        writer.SetInputData(polydata)
        writer.Write()
