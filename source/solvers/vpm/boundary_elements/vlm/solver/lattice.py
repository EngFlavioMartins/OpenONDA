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
        # This is the horseshoe increment, not the cumulative potential jump.
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
        self.bound_external_velocity = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))
        self.bound_kinematic_velocity = ti.field(dtype=dtype, shape=(max_n_panels, 3))
        self.reference_speed = 1.0

        # Kinematic velocity at collocation_point points (N x 3) - 2D scalar field for better stability
        self.kinematic_velocity = ti.field(dtype=dtype, shape=(max_n_panels, 3))

        # Pressure coefficient (N,) - matching VPM convention
        self.pressure_coefficient = ti.field(dtype=dtype, shape=(max_n_panels,))

        # Panel forces (N x 3)
        self.panel_force = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))
        self.unsteady_panel_force = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))
        self.unsteady_pressure_jump_coefficient = ti.field(dtype=dtype, shape=(max_n_panels,))
        # Pressure moment about the bound midpoint; KJ acts at that midpoint.
        self.panel_moment_correction = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))

        # External velocity field at collocation_point points (N x 3) - matching VPM convention
        self.external_velocity = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels,))
        # Completed near-wake row: two material edge displacements and fluid
        # velocities per TE panel, reused by assembly and particle shedding.
        self.wake_offset = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels, 2))
        self.trailing_edge_velocity = ti.Vector.field(3, dtype=dtype, shape=(max_n_panels, 2))

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
        """Discard wake particles accumulated for the current time step.

        The preallocated wake arrays are retained; only the device counter is
        set to zero.  Call this before a new shedding pass.
        """
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
        """Return the number of valid wake-buffer entries for this step."""
        return self.n_wake_particles[None]

    def reset(self):
        """Clear active panel state and reset the active count to zero.

        Geometry and solution fields across the fixed capacity are cleared or
        reinitialized as appropriate.  Host-side panel metadata is retained;
        callers that reuse the object must upload bodies again before solving.
        """
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
        """Return active collocation points, shape ``(N, 3)`` in metres."""
        return self.collocation_point.to_numpy()[: self.n_panels]

    def get_circulation(self) -> np.ndarray:
        """Return active per-panel circulation, shape ``(N,)`` in m²/s."""
        return self.circulation.to_numpy()[: self.n_panels]

    def get_velocity(self) -> np.ndarray:
        """Return active collocation velocities, shape ``(N, 3)`` in m/s."""
        return self.velocity.to_numpy()[: self.n_panels]

    def get_bound_vortex_velocity(self) -> np.ndarray:
        """Return active bound-leg velocities, shape ``(N, 3)`` in m/s."""
        return self.bound_vortex_velocity.to_numpy()[: self.n_panels]

    def get_kinematic_velocity(self) -> np.ndarray:
        """Return active prescribed panel velocities, shape ``(N, 3)`` in m/s."""
        return self.kinematic_velocity.to_numpy()[: self.n_panels]

    def get_panel_centre(self) -> np.ndarray:
        """Return active panel centroids, shape ``(N, 3)`` in metres."""
        return self.panel_centre.to_numpy()[: self.n_panels]

    def get_external_velocity(self) -> np.ndarray:
        """Return active incident velocities, shape ``(N, 3)`` in m/s."""
        return self.external_velocity.to_numpy()[: self.n_panels]

    def get_pressure_coefficient(self) -> np.ndarray:
        """Return active pressure coefficients, shape ``(N,)`` dimensionless."""
        return self.pressure_coefficient.to_numpy()[: self.n_panels]

    # --- NumPy-based geometry update (avoids Taichi field dimension bugs) -----
    def translate_panels(self, displacement: np.ndarray, start_idx: int = 0, end_idx: int = None):
        """Translate a contiguous panel range on the CPU.

        Parameters
        ----------
        displacement : array_like, shape (3,)
            Translation in metres.
        start_idx : int, default=0
            First panel index to update.
        end_idx : int, optional
            Exclusive final panel index; defaults to ``n_panels``.

        Notes
        -----
        Vertex, vortex-point, collocation, and bound-midpoint positions are
        mutated through NumPy field transfers.  Normals and panel centres are
        not recomputed here; use :meth:`rotate_translate_panels` or a geometry
        rebuild when those derived fields must change.
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
        """Rotate and translate a contiguous panel range on the CPU.

        Parameters
        ----------
        rotation_matrix : array_like, shape (3, 3)
            Rotation applied about ``origin``.
        origin : array_like, shape (3,)
            Rotation centre in metres.
        displacement : array_like, shape (3,)
            Translation applied after rotation, in metres.
        start_idx : int, default=0
            First panel index to update.
        end_idx : int, optional
            Exclusive final panel index; defaults to ``n_panels``.
        update_normal : bool, default=True
            Rotate stored unit normals with the geometry.

        Notes
        -----
        Positions are updated as ``R @ (x - origin) + origin + displacement``.
        The operation mutates the selected Taichi fields and does not
        recompute panel centres, areas, or connectivity.
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
        """Return whether any active panel has non-zero prescribed velocity."""
        if self.kinematic_velocity is None:
            return False
        kinematic_velocity = self.kinematic_velocity.to_numpy()[: self.n_panels]
        return np.any(np.abs(kinematic_velocity) > 1e-10)

    def set_kinematic_velocity(self, kinematic_velocity: np.ndarray):
        """Upload prescribed panel velocities.

        Parameters
        ----------
        kinematic_velocity : array_like, shape (N, 3) or (3,)
            Velocity values in m/s.  A single 3-vector is broadcast to all
            active panels; a longer array is truncated to active capacity and
            a shorter array leaves the remaining values at zero.

        Notes
        -----
        The full fixed-capacity Taichi field is rewritten.  The input is not
        modified.
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
        """Upload the incident velocity at each active panel.

        Parameters
        ----------
        external_velocity : array_like, shape (N, 3) or larger
            Incident fluid velocity in m/s.  At least one row per active panel
            is required; extra rows are ignored.

        Raises
        ------
        ValueError
            If fewer than ``n_panels`` rows or a non-three-component second
            dimension is supplied.
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
        """Return active panel force vectors, shape ``(N, 3)`` in newtons."""
        return self.panel_force.to_numpy()[: self.n_panels]

    def get_aerodynamic_influence_coefficient_matrix(self) -> np.ndarray:
        """Return the active dense influence matrix, shape ``(N, N)``."""
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
        """Atomically write the accepted lattice and its properties as PolyData."""
        from .vtk_export import CELL_FIELDS, write_lattice_vtk

        ti.sync()
        names = ("panel_corner_position", "vortex_point_position", *CELL_FIELDS)
        fields = {name: getattr(self, name).to_numpy()[: self.n_panels] for name in names}
        return write_lattice_vtk(
            fields,
            f"{filename}.vtp",
            reference_speed=self.reference_speed,
            time=time,
        )
