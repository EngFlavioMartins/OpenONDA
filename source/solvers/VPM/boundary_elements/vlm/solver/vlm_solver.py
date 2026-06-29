"""
Vortex-lattice-method solver (VLMSolver): assembles and solves the panel
circulation system and evaluates forces.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import copy
from dataclasses import dataclass
import json
import os
import time
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import taichi as ti

# Import VLM constants from centralized config
from ....config.constants import VLM_SMALL_VELOCITY

EPSILON = VLM_SMALL_VELOCITY

from ..coupling import kinematics as kin_module
from ..coupling.kinematics import RotatingVLM, StaticVLM
from ..geometry.aircraft import Aircraft, Wing
from ..geometry.surface_io import load_surface as _load_surface
from ..kernels.collision import detect_surface_collisions_kernel
from .influence import (
    apply_gamma_smooth,
    compute_AIC_matrix,
    compute_induced_velocities,
    compute_induced_velocities_at_bound,
    compute_panel_forces_coupled,
    compute_panel_forces_impulse_coupled,
    compute_pressure_coefficients,
    compute_RHS_coupled,
)
from .kernels import shed_wake_particles_kernel
from .lattice import VLMLattice
from .linear_solvers import get_linear_solver
from .mesh import (
    generate_vlm_mesh,
    update_trailing_directions_local,
    update_trailing_edge_directions,
)

# Import kinematics for type checking (avoid circular import at runtime)
if TYPE_CHECKING:
    from ..coupling.kinematics import VLMKinematics

@dataclass
class ForceConfig:
    """
    Configuration for aerodynamic force evaluation in VLM-VPM coupling.

    Supports two methodologies:

    **1. Conventional (Kutta-Joukowski):**
       Integrates pressure forces over bound vortex panels using K-J theorem:
       F = ρ Γ × V_local

       Pros:
       - Direct, instantaneous evaluation
       - Well-established in classical VLM
       - No time history required

       Cons:
       - Changes spuriously when far-wake particles are removed
       - Misses added-mass effects
       - Incomplete in unsteady flows

    **2. Experimental (Impulse-based):**
       Computes force from impulse time derivative:
       F = -dI/dt where I = (ρ/2) ∫ x × ω dV

       Pros:
       - Invariant to wake truncation (passes "killer test")
       - Automatically includes added-mass effects
       - Captures unsteady lift components
       - Consistent with momentum conservation

       Cons:
       - Requires time history (2-3 steps)
       - New methodology (less validated in production)
       - Returns zero on first time step

    **Usage:**
          # Use conventional K-J method (default)
          vlm = VLMSolver(force=ForceConfig.kutta_joukowski())

          # Use experimental impulse method
          vlm = VLMSolver(force=ForceConfig.impulse_based(order=2))
    """

    method: Literal["KUTTA_JOUKOWSKI", "IMPULSE"] = "KUTTA_JOUKOWSKI"
    """
      Force evaluation method.

      Options:
            - 'KUTTA_JOUKOWSKI': Classical pressure integration on bound panels (conventional)
            - 'IMPULSE': Force from impulse time derivative F = -dI/dt (experimental)
      """

    impulse_order: int = 2
    """
      Order of time derivative for impulse method (1 or 2).

      Used only when method='IMPULSE'.

      - Order 1: First-order backward difference (O(Δt) error)
      - Order 2: Second-order backward difference (O(Δt²) error)
      - Order 3: Third-order backward difference (O(Δt³) error)

      Order 2 or 3 recommended for smooth force histories.
      """

    impulse_smoothing_window: int | None = None
    """
      Moving average window for impulse force smoothing.

      Used only when method='IMPULSE'.

      If None: No smoothing (use instantaneous derivative)
      If int (e.g., 5): Apply moving average over last N force evaluations

      Recommended: 3-5 for noisy high-vorticity simulations.
      """

    impulse_startup_steps: int = 0
    """
      Number of startup steps during which the impulse signal dI/dt uses
      lower-order (BDF1) differentiation for initial stability.
      """

    impulse_differentiation: Literal["BACKWARD", "CENTRAL"] = "BACKWARD"
    """
      Differentiation scheme for the impulse force (dI/dt).

      - 'BACKWARD': Uses F_n = -(I_n - I_n-1)/dt (BDF1) or BDF2. Centered at t_n-1/2.
      - 'CENTRAL': Uses F_n = -(I_n+1 - I_n-1)/2dt (Centered). High stability, no ringing,
                   but introduces a 1-step lag in reporting.

      Recommended: 'BACKWARD' for real-time control, 'CENTRAL' for smooth validation plots.
      """

    kj_smoothing: bool = False
    """
      Apply 2-step running-average filter to the Kutta-Joukowski bound-circulation
      term in coupled VLM-VPM runs.

      The root cause of the 2Δt oscillation is explicit-coupling lag (the shed
      TE element is not in the solve). The Kelvin-consistent starting-vortex
      augmentation (_augment_starting_vortex) fixes the root cause; kj_smoothing
      is a band-aid filter that masks the symptom and must stay off so validation
      uses unfiltered CL.

      Applying γ_eff = 0.5*(γ_n + γ_{n-1}) before force integration exactly
      cancels the 2Δt mode while leaving the time-mean unchanged:
        - Steady-state CL is unaffected
        - Transient ramp is gently smoothed (one-step moving average of γ)
        - No change to wake shedding, AIC solve, or coupling dynamics

      Disable (False) only for standalone (uncoupled) VLM runs.
      """

    @staticmethod
    def kutta_joukowski() -> "ForceConfig":
        """
        Create conventional Kutta-Joukowski force configuration.

        Uses classical pressure integration over bound vortex panels.
        This is the standard method in VLM theory.

        Returns:
              ForceConfig: K-J force evaluation configuration
        """
        return ForceConfig(method="KUTTA_JOUKOWSKI")

    @staticmethod
    def impulse_based(
        order: int = 2,
        smoothing_window: int | None = None,
        startup_steps: int = 0,
        differentiation: Literal["BACKWARD", "CENTRAL"] = "BACKWARD",
    ) -> "ForceConfig":
        """
        Create experimental impulse-based force configuration.

        Computes force from impulse time derivative F = -dI/dt.
        This method is invariant to wake truncation.

        Args:
            order: Time derivative order (1, 2, or 3). Default: 2 (higher accuracy)
            smoothing_window: Moving average window size. Default: None (no smoothing)
            startup_steps: Number of BDF1-only startup steps. Default: 0
            differentiation: Scheme for dI/dt ('BACKWARD' or 'CENTRAL').

        Returns:
            ForceConfig: Impulse-based force evaluation configuration
        """
        if order not in (1, 2, 3):
            raise ValueError(f"impulse_order must be 1, 2, or 3, got {order}")
        return ForceConfig(
            method="IMPULSE",
            impulse_order=order,
            impulse_smoothing_window=smoothing_window,
            impulse_startup_steps=startup_steps,
            impulse_differentiation=differentiation,
        )

class VLMSolver:
    """
    Vortex Lattice Method solver with VPM coupling support.

    Solves for circulation distribution on thin lifting surfaces using
    horseshoe vortex elements and the zero-normal-flow boundary condition.

    **Initialization Options:**

    :param surface: Surface geometry (Aircraft object, JSON path, or None for multi-surface)
    :param max_panels: Maximum panels to allocate (default: 10000)
    :param dtype: Data type 'f32' or 'f64' (default: 'f32')
    :param linear_solver: Linear system solver (see options below)
    :param relaxation_factor: Renamed to `circulation_relaxation` (default: 1.0)
    :param viscosity: Kinematic viscosity [m²/s] (default: 1.0)
    :param density: Fluid density [kg/m³] (default: 1.0 for air)
    :param kinematics: Optional kinematics for single-surface mode

    **Linear Solver Options:**

    - ``'SCIPY'`` (default): CPU direct solver, fastest for <500 panels
    - ``'BICGSTAB_GPU'``: GPU iterative solver, best for >500 panels (non-symmetric)
    - ``'CG_GPU'``: GPU iterative solver, only for symmetric matrices

    Workflow:
        1. generate_mesh() - Create VLM panels from aircraft geometry
        2. set_motion_profile() - Set kinematics (optional, default: Static)
        3. solve(V_external) - Compute circulation with external velocity field
        4. compute_forces() - Calculate aerodynamic forces and moments

    Example (single surface)::

        >>> solver = VLMSolver(surface="wing.json", linear_solver='BICGSTAB_GPU')
        >>> solver.set_motion_profile(StaticVLM())
        >>> V_ext = np.zeros((n_panels, 3))
        >>> V_ext[:] = [10, 0, 0]
        >>> gamma = solver.solve(V_external=V_ext)
        >>> forces = solver.compute_forces(density=1.0, U_ref_mag=10.0)

    Example (multi-surface rotor)::

        >>> from source.solvers.VPM.boundary_elements.vlm.coupling.kinematics import RotatingVLM
        >>> solver = VLMSolver(linear_solver='BICGSTAB_GPU')
        >>> kinematics = RotatingVLM(omega=10.0, axis=[1,0,0], center=[0,0,0])
        >>> for i in range(3):  # 3 blades
        ...     solver.add_surface("blade.json", kinematics=kinematics,
        ...                        rotation_deg=[i*120, 0, 0])
        >>> # VPM coupling handles the rest
    """

    def __init__(
        self,
        surface=None,
        max_panels: int = 10000,
        dtype: str = "f32",
        linear_solver: str | None = None,
        circulation_relaxation: float = 1.0,
        viscosity: float = 1.0,
        density: float = 1.0,
        sigma_factor: float = 2.5,
        U_inf: np.ndarray | None = None,
        logging_frequency: int = 1,
        force: Optional["ForceConfig"] = None,
        kinematics: Optional["VLMKinematics"] = None,
        sample_surface_forces: bool = False,
    ):
        """
        Initialize VLM solver.

        Args:
            surface: Surface geometry (one of):
                - Aircraft object directly
                - String path to JSON surface file
                - None (use add_surface() later for multi-body)
            max_panels: Maximum panels to allocate across all surfaces.
                Higher values use more GPU memory.
            dtype: Floating point precision:
                - 'f32': Single precision (faster, less memory)
                - 'f64': Double precision (more accurate)
            linear_solver: Algorithm for solving AIC @ gamma = rhs:
                - 'BICGSTAB_GPU' (default): GPU BiCGSTAB iterative solver
                    - Pros: No CPU transfer, scales to large systems
                    - Cons: May need more iterations for ill-conditioned matrices
                - 'SCIPY': CPU direct solver (scipy.linalg.solve)
                    - Pros: Robust, fast for small systems (<500 panels)
                    - Cons: CPU-GPU transfer overhead
                - 'CG_GPU': GPU Conjugate Gradient (for symmetric matrices only)
            circulation_relaxation: Under-relaxation applied to circulation after each AIC solve,
                within a single timestep. 1.0 = no relaxation (standard). Values < 1.0 slow
                convergence but can stabilise ill-conditioned AIC systems on very coarse meshes.
                A previous tutorial value of 0.5 was masking an under-resolved ramp acceleration
                (tau_ramp/dt too small); the correct fix is to reduce the timestep, not this.
            viscosity: Kinematic viscosity of fluid [m²/s].
                Default 1.0 so it won't interfer with results.
            density: Fluid density [kg/m³].
                Default 1.0 so it won't interfer with results.
            sigma_factor: Particle core radius sizing factor.
                Controls the overlap ratio σ/(V·dt) for shed wake particles.
                Higher values provide stronger regularization and prevent
                divergence from near-singular particle interactions.
                Default 2.5 (recommended for rotor wakes in hover).
                Use 1.0 for forward-flight cases with weaker wake interactions.
            U_inf: Freestream velocity vector used for reference normalization.
            logging_frequency: Frequency for VLM force CSV writing in coupled runs.
            kinematics: Body kinematics for single-surface mode.
                For multi-surface, use add_surface() with per-surface kinematics.

        Examples:
            Simple wing::

                >>> solver = VLMSolver(surface="wing.json")

            GPU solver for large rotor::

                >>> solver = VLMSolver(linear_solver='BICGSTAB_GPU', max_panels=5000)
                >>> solver.add_surface("blade.json", kinematics=RotatingVLM(...))
        """
        # Solver configuration
        self.max_panels = max_panels
        self.dtype = dtype
        self.epsilon = EPSILON
        # Adaptive default: SCIPY is faster for small systems (<1000 panels)
        # because BICGSTAB_GPU kernel-launch overhead dominates on small matrices.
        if linear_solver is None:
            linear_solver = "SCIPY" if max_panels < 1000 else "BICGSTAB_GPU"
        self.linear_solver = linear_solver
        self.circulation_relaxation = circulation_relaxation

        # Multi-body storage: Dict[uid -> (Aircraft, kinematics)]
        self.surfaces = {}

        # Combined aircraft for mesh generation (populated by generate_mesh)
        self.aircraft = None
        self.surface = None

        # Lattice will be created when first needed (after Taichi init)
        self.lattice = None
        self._lattice_initialized = False
        self._current_time = None  # Temporary: set during advance operations from parent solver

        # Flight condition / Fluid properties
        self.U_inf = None if U_inf is None else np.array(U_inf, dtype=np.float64)
        self.logging_frequency = max(1, int(logging_frequency))
        self.density = density  # kg/m³ (fluid density)
        self.viscosity = viscosity  # m²/s (kinematic viscosity)
        self.sigma_factor = sigma_factor  # wake particle core sizing
        self.alpha_rad = 0.0
        self.beta_rad = 0.0

        # ---- Transverse shedding threshold ----
        # Minimum |ΔΓ| for emitting a transverse (closure) wake particle.
        # Default 0.0 means always emit, which prevents binary wake-topology
        # changes near steady state.  Set to a finite value (e.g. 1e-3) to
        # reproduce the threshold artefact as a diagnostic comparison.
        self.transverse_shedding_threshold: float = 0.0

        # solution state
        self._mesh_generated = False
        self._AIC_computed = False
        self._solved = False

        # Force evaluation configuration
        self.force = force if force is not None else ForceConfig(method="KUTTA_JOUKOWSKI")

        # Explicit initialization of optional attributes to avoid AttributeError
        self._surface_transforms: dict[str, dict] = {}
        self._surface_group_ids: dict[str, int] = {}
        self._surface_sampling: dict[str, bool] = {}
        self.sample_surface_forces: bool = bool(sample_surface_forces)
        self._mesh_refinement_settings: dict[str, dict] = {}
        self._pending_refinement: dict | None = None

        # Mesh generation settings for idempotency
        self._spanwise_spacing = None
        self._spanwise_spacing_ratio = None
        self._spanwise_spacing_direction = None
        self._spanwise_spacing_region = None

        if surface is not None:
            # Single-surface mode: load immediately
            self.add_surface(surface, kinematics=kinematics)

        print(
            f"   [VLM Solver] Initialized (max_panels={max_panels}, dtype={dtype}, solver={linear_solver})"
        )

    # V_inf removed: use U_inf directly

    def _ensure_lattice_initialized(self) -> None:
        """Ensure lattice is created (lazy initialization after Taichi init).

        VPM is required to be the Taichi master: the VPM Solver must call
        ``initialize_taichi_backend`` (i.e. ``Solver.__init__``) *before* any
        VLM lattice fields are allocated.  This prevents a mismatch between the
        precision that Taichi was initialised with and the dtype requested here.
        """
        if not self._lattice_initialized:
            # Guard: Taichi must already be initialised by the VPM Solver.
            if ti.lang.impl.get_runtime().prog is None:
                raise RuntimeError(
                    "VLMSolver._ensure_lattice_initialized called before Taichi "
                    "is initialised.  Always construct the VPM Solver first so "
                    "that initialize_taichi_backend() runs before any VLM fields "
                    "are created."
                )

            # Warn if the requested VLM dtype does not match the Taichi runtime
            # default floating-point type.  Mismatches cause silent precision
            # conversions (f32 ↔ f64) that can degrade accuracy.
            runtime_fp = ti.lang.impl.get_runtime().default_fp
            vlm_fp = ti.f32 if self.dtype == "f32" else ti.f64
            if runtime_fp != vlm_fp:
                import warnings

                runtime_name = "f32" if runtime_fp == ti.f32 else "f64"
                warnings.warn(
                    f"VLMSolver dtype='{self.dtype}' does not match the Taichi "
                    f"runtime precision '{runtime_name}' set by the VPM Solver.  "
                    f"Overriding VLM dtype to '{runtime_name}' to keep VPM as the "
                    f"precision master and avoid f32/f64 conversion errors.",
                    stacklevel=3,
                )
                vlm_fp = runtime_fp
                self.dtype = runtime_name

            ti_dtype = vlm_fp
            self.lattice = VLMLattice(self.max_panels, ti_dtype)
            self._lattice_initialized = True

    def generate_mesh(
        self,
        spanwise_spacing: str = "uniform",
        spanwise_spacing_ratio: float = 1.0,
        spanwise_spacing_direction: str = "y",
        spanwise_spacing_region: str = "both",
    ) -> None:
        """
        Generate VLM mesh from aircraft geometry.

        Args:
            spanwise_spacing: Panel distribution method ('uniform' or 'geometric')
            spanwise_spacing_ratio: Concentration ratio for geometric spacing (max/min panel size)
            spanwise_spacing_direction: Axis for refinement ('x', 'y', or 'z')
            spanwise_spacing_region: Where to apply refinement:
                'start': Smallest at start (0), largest at end
                'end': Smallest at end (1), largest at start
                'both': Smallest at both ends, largest at center
        """
        self._ensure_lattice_initialized()

        # Avoid redundant generation if settings haven't changed
        if (
            self._mesh_generated
            and self._spanwise_spacing == spanwise_spacing
            and self._spanwise_spacing_ratio == spanwise_spacing_ratio
            and self._spanwise_spacing_direction == spanwise_spacing_direction
            and self._spanwise_spacing_region == spanwise_spacing_region
        ):
            return

        # Store spacing parameters for mesh generation
        self._spanwise_spacing = spanwise_spacing
        self._spanwise_spacing_ratio = spanwise_spacing_ratio
        self._spanwise_spacing_direction = spanwise_spacing_direction
        self._spanwise_spacing_region = spanwise_spacing_region

        print("   [VLM Solver] Generating mesh...")
        t0 = time.time()

        generate_vlm_mesh(
            self.aircraft,
            self.lattice,
            spanwise_spacing=spanwise_spacing,
            spanwise_spacing_ratio=spanwise_spacing_ratio,
            spanwise_spacing_region=spanwise_spacing_region,
        )

        # Apply per-surface transformations AFTER mesh generation
        self._apply_surface_transforms()

        # Populate panel group IDs
        group_ids = self.get_panel_group_ids()

        # Pad group_ids to match max_panels for Taichi field assignment
        if group_ids.size < self.max_panels:
            padded_ids = np.zeros(self.max_panels, dtype=np.int32)
            padded_ids[: group_ids.size] = group_ids
            self.lattice.panel_group_id.from_numpy(padded_ids)
        else:
            self.lattice.panel_group_id.from_numpy(group_ids)

        self._mesh_generated = True
        self._AIC_computed = False
        self._solved = False

        t_elapsed = time.time() - t0
        print(
            f"   [VLM Solver] Mesh generation complete ({self.lattice.num_panels} panels in {t_elapsed:.3f}s)"
        )

        # Print summary
        n_wings = len(self.aircraft.wings)
        total_area = sum(self.lattice.areas.to_numpy()[i] for i in range(self.lattice.num_panels))
        print(f"   [VLM Solver] Wings: {n_wings}, Total area: {total_area:.2f} m²")

    def _build_wing_panel_ranges(self) -> dict[str, tuple[int, int]]:
        """Build mapping from wing UID to panel range indices."""
        panel_idx = 0
        wing_panel_ranges = {}

        for wing_uid, wing in self.aircraft.wings.items():
            n_panels_wing = 0
            for seg in wing.segments.values():
                n_panels_wing += seg.panels_chord * seg.panels_span
                if wing.symmetry > 0:
                    n_panels_wing += seg.panels_chord * seg.panels_span
            wing_panel_ranges[wing_uid] = (panel_idx, panel_idx + n_panels_wing)
            panel_idx += n_panels_wing

        return wing_panel_ranges

    def _transform_panel_points(
        self,
        corners,
        vortex_pts,
        collocation,
        bound_mids,
        newnormals,
        start_idx,
        end_idx,
        R,
        translation,
        rotation_center,
    ) -> None:
        """Apply transformation to a range of panels (vectorized)."""
        # Transform corners and vortex points (shape: [n_panels, 4, 3])
        for j in range(4):
            # Vectorized transformation
            p_corners = corners[start_idx:end_idx, j] - rotation_center
            corners[start_idx:end_idx, j] = (R @ p_corners.T).T + rotation_center + translation

            p_vortex = vortex_pts[start_idx:end_idx, j] - rotation_center
            vortex_pts[start_idx:end_idx, j] = (R @ p_vortex.T).T + rotation_center + translation

        # Transform collocation and bound midpoints
        p_coll = collocation[start_idx:end_idx] - rotation_center
        collocation[start_idx:end_idx] = (R @ p_coll.T).T + rotation_center + translation

        p_bound = bound_mids[start_idx:end_idx] - rotation_center
        bound_mids[start_idx:end_idx] = (R @ p_bound.T).T + rotation_center + translation

        # Rotate normals (no translation)
        newnormals[start_idx:end_idx] = (R @ newnormals[start_idx:end_idx].T).T

    def _write_lattice_arrays(
        self, n_panels, corners, vortex_pts, collocation, bound_mids, newnormals
    ) -> None:
        """Write numpy arrays back to GPU lattice."""
        for i in range(n_panels):
            for j in range(4):
                for k in range(3):
                    self.lattice.corners[i, j][k] = corners[i, j, k]
                    self.lattice.vortex_points[i, j][k] = vortex_pts[i, j, k]
            for k in range(3):
                self.lattice.collocation[i][k] = collocation[i, k]
                self.lattice.bound_midpoints[i][k] = bound_mids[i, k]
                self.lattice.normals[i][k] = newnormals[i, k]

    def _apply_transform_to_surface(
        self,
        surface_name,
        transform,
        wing_panel_ranges,
        corners,
        vortex_pts,
        collocation,
        bound_mids,
        newnormals,
    ) -> None:
        """Apply a single surface transform to matching wings."""
        if transform["rotation_deg"] is None and transform["translation"] is None:
            return

        R = self._build_rotation_matrix(transform["rotation_deg"])
        translation = (
            np.array(transform["translation"])
            if transform["translation"] is not None
            else np.zeros(3)
        )
        rotation_center = (
            np.array(transform["rotation_center"])
            if transform["rotation_center"] is not None
            else np.zeros(3)
        )

        # Find which wings belong to this surface and transform them
        for wing_uid, (start_idx, end_idx) in wing_panel_ranges.items():
            if wing_uid.startswith(surface_name + "_") or wing_uid == surface_name:
                self._transform_panel_points(
                    corners,
                    vortex_pts,
                    collocation,
                    bound_mids,
                    newnormals,
                    start_idx,
                    end_idx,
                    R,
                    translation,
                    rotation_center,
                )

    def _apply_surface_transforms(self) -> None:
        """
        Apply stored transformations to lattice points per-surface.

        This is called after mesh generation to position each surface correctly.
        """
        if not self._surface_transforms:
            return

        wing_panel_ranges = self._build_wing_panel_ranges()

        # Get lattice arrays (must be numpy for modification)
        n_panels = self.lattice.num_panels
        corners = self.lattice.corners.to_numpy()[:n_panels]
        vortex_pts = self.lattice.vortex_points.to_numpy()[:n_panels]
        collocation = self.lattice.collocation.to_numpy()[:n_panels]
        bound_mids = self.lattice.bound_midpoints.to_numpy()[:n_panels]
        newnormals = self.lattice.normals.to_numpy()[:n_panels]

        # Apply each surface transformation
        for surface_name, transform in self._surface_transforms.items():
            self._apply_transform_to_surface(
                surface_name,
                transform,
                wing_panel_ranges,
                corners,
                vortex_pts,
                collocation,
                bound_mids,
                newnormals,
            )

        self._write_lattice_arrays(
            n_panels, corners, vortex_pts, collocation, bound_mids, newnormals
        )

    def _build_rotation_matrix(self, rotation_deg) -> np.ndarray:
        """Build 3x3 rotation matrix from [rx, ry, rz] angles in degrees."""
        R = np.eye(3)
        if rotation_deg is None:
            return R

        rotation_deg = np.array(rotation_deg, dtype=np.float64)

        # Rotation about X
        if abs(rotation_deg[0]) > 1e-10:
            rx = np.radians(rotation_deg[0])
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            R = Rx @ R

        # Rotation about Y
        if abs(rotation_deg[1]) > 1e-10:
            ry = np.radians(rotation_deg[1])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            R = Ry @ R

        # Rotation about Z
        if abs(rotation_deg[2]) > 1e-10:
            rz = np.radians(rotation_deg[2])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            R = Rz @ R

        return R

    def add_surface(
        self,
        surface,
        surface_name: str | None = None,
        kinematics: Optional["VLMKinematics"] = None,
        mesh_refinement_type: str = "uniform",
        mesh_refinement_ratio: float = 1.0,
        mesh_refinement_direction: str = "y",
        mesh_refinement_region: str = "both",
        translation: np.ndarray | None = None,
        rotation_deg: np.ndarray | None = None,
        rotation_center: np.ndarray | None = None,
        group_id: int = 0,
        sample_surface_forces: bool | None = None,
    ) -> str:
        """
        Add a surface to the VLM solver with automatic mesh generation.

        This is the primary method for setting up VLM surfaces. It combines
        surface loading and mesh generation into a single step.

        Args:
            surface: Surface geometry - can be:
                    - Aircraft object directly
                    - String path to JSON surface file
            surface_name: Optional unique name for this surface. If None, uses Aircraft.uid.
            kinematics: Motion profile for this surface (default: StaticVLM)
            mesh_refinement_type: Panel distribution method ('uniform' or 'geometric')
            mesh_refinement_ratio: Concentration ratio for geometric refinement (max/min panel size)
            mesh_refinement_direction: Axis for refinement ('x', 'y', or 'z')
            mesh_refinement_region: Where to apply refinement:
                'start': Smallest panels at start (0), largest at end (1)
                'end': Smallest panels at end (1), largest at start (0)
                'both': Smallest at both ends (0 and 1), largest at center (0.5)
            translation: [x, y, z] translation to apply to surface vertices
            rotation_deg: [rx, ry, rz] rotation angles in degrees (applied in X->Y->Z order)
            rotation_center: Center of rotation (default: origin [0, 0, 0])
            group_id: Integer ID for wake particles shed from this surface (default: 0)

        Returns:
            str: The surface UID/name for reference
        """
        # Load from file if string path
        aircraft = _load_surface(surface) if isinstance(surface, str) else copy.deepcopy(surface)

        # Determine name
        if surface_name is None:
            surface_name = aircraft.uid

        # Default kinematics
        if kinematics is None:
            kinematics = StaticVLM()

        # Store in surfaces dict with mesh refinement settings
        # Note: We now store group_id in a separate dict to avoid breaking existing tuple unpacking
        self.surfaces[surface_name] = (aircraft, kinematics)

        self._surface_group_ids[surface_name] = group_id
        self._surface_sampling[surface_name] = (
            self.sample_surface_forces
            if sample_surface_forces is None
            else bool(sample_surface_forces)
        )

        # Store mesh refinement settings for this surface (used when mesh is generated)
        self._mesh_refinement_settings[surface_name] = {
            "type": mesh_refinement_type,
            "ratio": mesh_refinement_ratio,
            "direction": mesh_refinement_direction,
            "region": mesh_refinement_region,
        }

        # Store transformation parameters (applied AFTER mesh generation)
        self._surface_transforms[surface_name] = {
            "translation": translation,
            "rotation_deg": rotation_deg,
            "rotation_center": rotation_center,
        }

        # Update main aircraft reference (combined or first surface)
        self._update_combined_aircraft()

        # Mark mesh as pending (deferred until VPMSolver initializes Taichi)
        self._mesh_generated = False
        self._pending_refinement = {
            "type": mesh_refinement_type,
            "ratio": mesh_refinement_ratio,
            "direction": mesh_refinement_direction,
            "region": mesh_refinement_region,
        }

        print(
            f"   [VLM Solver] Added surface '{surface_name}' ({aircraft.total_num_panels()} panels, group_id={group_id})"
        )
        return surface_name

    def load_scene(self, layout_file: str):
        """
        Load a complete scene (assembly of surfaces) from a JSON layout file.

        The layout file should define a list of surfaces with their properties:
        {
            "surfaces": [
                {
                    "file": "path/to/surface.json",
                    "name": "surface_name",
                    "translation": [x, y, z],
                    "rotation_deg": [rx, ry, rz],
                    "kinematics": {
                        "type": "RotatingVLM",
                        "rotation_speed": rad_per_s,
                        "rotation_axis": [x, y, z],
                        "rotation_center": [x, y, z]
                    }
                },
                ...
            ]
        }

        This method automatically increments group_id for each added surface.

        Args:
            layout_file: Path to the JSON layout file.
        """
        with open(layout_file) as f:
            data = json.load(f)

        base_dir = os.path.dirname(layout_file)

        group_id_counter = 0

        for surf_data in data.get("surfaces", []):
            # Resolve file path relative to layout file
            surf_file = surf_data.get("file")
            if not os.path.isabs(surf_file):
                surf_file = os.path.join(base_dir, surf_file)

            name = surf_data.get("name")
            translation = np.array(surf_data.get("translation", [0, 0, 0]))
            rotation_deg = np.array(surf_data.get("rotation_deg", [0, 0, 0]))
            rot_center = surf_data.get("rotation_center")
            if rot_center:
                rot_center = np.array(rot_center)

            # Parse Kinematics
            kin_data = surf_data.get("kinematics", {"type": "StaticVLM"})
            kin_type = kin_data.get("type")

            # Use getattr to safely get the class, default to StaticVLM
            try:
                kin_cls = getattr(kin_module, kin_type)
            except AttributeError:
                print(f"Warning: Unknown kinematics type '{kin_type}', defaulting to StaticVLM")
                kin_cls = kin_module.StaticVLM

            # Prepare kwargs for kinematics constructor
            # Filter out 'type' from dict
            kin_kwargs = {k: v for k, v in kin_data.items() if k != "type"}

            # Convert known list arguments to numpy arrays for kinematics
            for k, v in kin_kwargs.items():
                if isinstance(v, list):
                    kin_kwargs[k] = np.array(v)

            kinematics = kin_cls(**kin_kwargs)

            self.add_surface(
                surf_file,
                surface_name=name,
                translation=translation,
                rotation_deg=rotation_deg,
                rotation_center=rot_center,
                kinematics=kinematics,
                group_id=group_id_counter,
            )

            group_id_counter += 1

        print(f"Loaded scene from {layout_file} ({group_id_counter} surfaces)")

    def get_panel_group_ids(self) -> np.ndarray:
        """
        Get array of group IDs for each panel in the lattice.

        Returns:
             np.ndarray: Array of shape (num_panels,) with group ID for each panel.
        """
        if self.lattice is None:
            return np.zeros(0, dtype=np.int32)

        num_panels = self.lattice.num_panels
        group_ids = np.zeros(num_panels, dtype=np.int32)

        if not self._surface_group_ids:
            return group_ids

        wing_ranges = self._build_wing_panel_ranges()

        # Map panels to surfaces
        # This relies on the convention that combined wings are named "{surface}_{wing}"
        # or exactly "{surface}" if single wing.

        for surface_name, gid in self._surface_group_ids.items():
            # Find all wings belonging to this surface
            for wing_uid, (start, end) in wing_ranges.items():
                # Check for match.
                # Wing UID in combined aircraft is constructed as f"{name}_{wing_uid}" in _update_combined_aircraft
                # But careful: if user names surface "wing" and original wing is "wing_0", combined is "wing_wing_0".

                # More robust check:
                # If single surface mode: keys are just original wing UIDs.
                if len(self.surfaces) == 1:
                    # All panels belong to the single surface
                    group_ids[start:end] = gid
                else:
                    # Multi-surface mode: keys are f"{surface_name}_{original_wing_uid}"
                    if wing_uid.startswith(f"{surface_name}_"):
                        group_ids[start:end] = gid

        return group_ids

    def ensure_mesh_generated(self) -> None:
        """
        Ensure mesh is generated. Called by VPMSolver after Taichi is initialized.

        This deferred initialization ensures Taichi fields are created with the
        correct backend (GPU/Vulkan) as configured by VPMSolver.
        """
        if self._mesh_generated:
            return

        if not self.aircraft:
            return

        # Get refinement settings (_pending_refinement is None when no explicit
        # generate_mesh() call was made before Solver() construction)
        refinement = getattr(self, "_pending_refinement", None) or {}

        self.generate_mesh(
            spanwise_spacing=refinement.get("type", "uniform"),
            spanwise_spacing_ratio=refinement.get("ratio", 1.0),
            spanwise_spacing_direction=refinement.get("direction", "y"),
            spanwise_spacing_region=refinement.get("region", "both"),
        )

    def _update_combined_aircraft(self) -> None:
        """Update combined aircraft from all loaded surfaces."""
        if not self.surfaces:
            self.aircraft = None
            self.surface = None
            return

        if len(self.surfaces) == 1:
            # Single surface mode
            name, (aircraft, _) = next(iter(self.surfaces.items()))
            self.aircraft = aircraft
            self.surface = aircraft
        else:
            # Multi-surface: merge into combined aircraft
            combined = Aircraft(uid="combined")
            for name, (aircraft, _) in self.surfaces.items():
                for wing_uid, wing in aircraft.wings.items():
                    # Create unique wing name
                    combined_uid = f"{name}_{wing_uid}"
                    # Copy wing with new uid
                    new_wing = Wing(
                        uid=combined_uid, segments=wing.segments.copy(), symmetry=wing.symmetry
                    )
                    combined.wings[combined_uid] = new_wing
            combined.compute_default_refs()
            self.aircraft = combined
            self.surface = combined

    @property
    def kinematics(self):
        """Get kinematics for single-surface mode."""
        if len(self.surfaces) == 1:
            _, (_, kin) = next(iter(self.surfaces.items()))
            return kin
        return None

    @kinematics.setter
    def kinematics(self, value):
        """Set kinematics for first surface."""
        if self.surfaces:
            first_name = next(iter(self.surfaces.keys()))
            aircraft, _ = self.surfaces[first_name]
            self.surfaces[first_name] = (aircraft, value)

    def _compute_rotor_tip_speed(self, name: str, kin) -> tuple[np.ndarray, float]:
        """Return (velocity_vector, tip_speed) for a RotatingVLM surface, or (zeros, 0)."""
        if not isinstance(kin, RotatingVLM) or not self._mesh_generated:
            return np.zeros(3), 0.0
        omega_mag = abs(kin.omega)
        if omega_mag < 1e-10:
            return np.zeros(3), 0.0
        center = getattr(kin, "center", np.zeros(3))
        axis = getattr(kin, "axis", np.array([0.0, 0.0, 1.0]))
        colloc = self.lattice.get_collocation_points()
        wing_ranges = self._build_wing_panel_ranges()
        prefix = f"{name}_" if len(self.surfaces) > 1 else ""
        best_tip = 0.0
        for w_uid, (ws, we) in wing_ranges.items():
            if not (len(self.surfaces) == 1 or w_uid.startswith(prefix) or w_uid == name):
                continue
            r_vecs = colloc[ws:we] - center
            axial = np.outer(r_vecs @ axis, axis)
            radial = r_vecs - axial
            R_max = np.max(np.linalg.norm(radial, axis=1)) if len(radial) > 0 else 0.0
            best_tip = max(best_tip, omega_mag * R_max)
        return axis * best_tip, best_tip

    def _get_active_kinematic_velocity(self) -> np.ndarray:
        """
        Get a representative kinematic velocity from active surfaces.

        For translating surfaces, returns the translational velocity.
        For rotating surfaces, returns a velocity vector whose magnitude
        equals the maximum tip speed (omega × R_max) across all rotors,
        directed along the rotation axis (for reference scaling only).

        Returns:
            Velocity vector [vx, vy, vz] or zeros if no active kinematics
        """
        t = self._current_time if self._current_time is not None else 0.0
        best_V = np.zeros(3)
        best_mag = 0.0
        for name, (_, kin) in self.surfaces.items():
            if kin is None or isinstance(kin, StaticVLM):
                continue
            V_trans = kin.get_velocity(t)
            V_trans_mag = np.linalg.norm(V_trans)
            if V_trans_mag > best_mag:
                best_V, best_mag = V_trans, V_trans_mag
            V_rot, tip_speed = self._compute_rotor_tip_speed(name, kin)
            if tip_speed > best_mag:
                best_V, best_mag = V_rot, tip_speed
        return best_V

    def _get_max_kinematic_speed(self) -> float:
        """
        Get maximum kinematic speed across all surfaces.

        For rotating surfaces this is the tip speed (omega × R_max).
        For translating surfaces this is the translational speed.

        Returns:
            Maximum kinematic speed [m/s], or 0.0 if no active kinematics
        """
        V = self._get_active_kinematic_velocity()
        return float(np.linalg.norm(V))

    def _get_active_kinematics(self):
        """
        Get the active kinematics object.

        For multi-surface, returns the first non-StaticVLM kinematics.
        """
        if len(self.surfaces) == 1:
            return self.kinematics

        # Multi-surface mode
        for _name, (_, kin) in self.surfaces.items():
            if kin is not None and not isinstance(kin, StaticVLM):
                return kin

        return None

    def update_trailing_directions(self, V_transport: np.ndarray) -> None:
        """
        Update trailing edge directions based on transport velocity.

        The wake trails in the direction of the local relative velocity.

        Args:
           V_transport: Velocity vector/field for wake transport (N, 3) or (3,)
                        Ideally this is V_external - V_skinematic at trailing edge.
        """
        if not self._mesh_generated:
            self.generate_mesh()

        # If scalar vector provided, apply uniformly
        if V_transport.ndim == 1:
            update_trailing_edge_directions(self.lattice, V_transport)
        else:
            # Use local velocity field
            update_trailing_directions_local(self.lattice, V_transport)

        self._AIC_computed = False

    def _find_surface_panel_range(
        self, surface_name: str, wing_ranges: dict, n_panels: int
    ) -> tuple[int, int] | None:
        """Return (start, end) panel indices for a surface, or None if not found."""
        if len(self.surfaces) == 1:
            return 0, n_panels
        prefix = f"{surface_name}_"
        min_start, max_end, found = n_panels, 0, False
        for w_uid, (w_start, w_end) in wing_ranges.items():
            if w_uid.startswith(prefix) or w_uid == surface_name:
                found = True
                min_start = min(min_start, w_start)
                max_end = max(max_end, w_end)
        return (min_start, max_end) if found else None

    def _compute_panel_kin_velocity(
        self, kin, flow_time: float, collocation, start_idx: int, end_idx: int
    ) -> np.ndarray:
        """Return (end-start, 3) kinematic velocity array for a panel slice."""
        omega_vec = kin.get_angular_velocity(flow_time)
        V_trans = kin.get_velocity(flow_time)
        center = getattr(kin, "center", None)
        if center is None:
            center = getattr(kin, "rotation_center", np.zeros(3))
        r_vecs = collocation[start_idx:end_idx] - center
        return V_trans + np.cross(omega_vec, r_vecs)

    def advance_time(self, dt: float, current_time: float) -> None:
        """Advance kinematics state and geometry for all surfaces."""
        if not self._mesh_generated:
            self.generate_mesh()
        self._current_time = float(current_time)
        flow_time = self._current_time
        n_panels = self.lattice.num_panels
        V_kin = np.zeros((n_panels, 3), dtype=np.float64)
        has_kinematics = False
        wing_ranges = self._build_wing_panel_ranges()
        for surface_name, (_aircraft_obj, kin) in self.surfaces.items():
            if kin is None or isinstance(kin, StaticVLM):
                continue
            has_kinematics = True
            panel_range = self._find_surface_panel_range(surface_name, wing_ranges, n_panels)
            if panel_range is None:
                continue
            start_idx, end_idx = panel_range
            kin.update(self, flow_time, dt, panel_range=(start_idx, end_idx))
            collocation = self.lattice.get_collocation_points()
            V_kin[start_idx:end_idx] = self._compute_panel_kin_velocity(
                kin, flow_time, collocation, start_idx, end_idx
            )
        self.lattice.set_kinematic_velocity(V_kin)
        if has_kinematics:
            self._AIC_computed = False
            self._solved = False

    def compute_influence_matrix(self, force: bool = False) -> None:
        """Compute AIC matrix."""
        if self._AIC_computed and not force:
            return

        if not self._mesh_generated:
            self.generate_mesh()

        # print("\nComputing AIC matrix...")
        compute_AIC_matrix(
            self.lattice.collocation,
            self.lattice.vortex_points,
            self.lattice.corners,
            self.lattice.normals,
            self.lattice.is_TE_panel,
            self.lattice.te_index,
            self.lattice.AIC,
            self.lattice.num_panels,
            self.epsilon,
            0,  # coupled_mode = 0 (standalone)
            ti.Vector([0.0, 0.0, 0.0]),  # wake_offset (unused in standalone)
        )
        self._AIC_computed = True

    def check_coupling_stability(
        self, dt: float, background_velocity: list[float] | None = None
    ) -> None:
        """
        Check if time step is stable for VLM-VPM coupling (CFL-like condition).

        Condition: dt >= min_chord / V_characteristic

        Args:
            dt: Time step size [s]
            background_velocity: Background flow velocity [vx, vy, vz] or None
        """
        return

    def _run_linear_solver(self, n_panels: int) -> np.ndarray:
        """Solve AIC@gamma=rhs and return gamma numpy array."""
        solver = get_linear_solver(
            self.linear_solver, max_panels=self.max_panels, use_preconditioner=True
        )
        if solver.is_gpu:
            # 1e-10 is pathologically tight for iterative solvers; 1e-6 is
            # sufficient for VLM engineering accuracy and avoids hundreds of
            # kernel-launch-bound iterations on small systems.
            solver.solve(
                self.lattice.AIC,
                self.lattice.rhs,
                self.lattice.circulation,
                self.lattice.num_panels,
                max_iterations=1000,
                tolerance=1e-6,
            )
        else:
            solver.solve(
                self.lattice.AIC,
                self.lattice.rhs,
                self.lattice.circulation,
                self.lattice.num_panels,
            )
        if self.circulation_relaxation < 1.0:
            self.lattice.apply_relaxation(self.circulation_relaxation)
        return self.lattice.circulation.to_numpy()[:n_panels]

    def _do_debug_sign_check(self, gamma_np: np.ndarray) -> None:
        """One-shot sign convention debug log on first solve."""
        if getattr(self, "_debug_sign_done", False):
            return
        import logging as _logging

        _log_debug = _logging.getLogger("vlm")
        if _log_debug.isEnabledFor(_logging.DEBUG):
            try:
                _n0 = self.lattice.normals.to_numpy()[0]
                _vc0 = self.lattice.kinematic_velocity.to_numpy()[0]
                _ve0 = self.lattice.external_velocity.to_numpy()[0]
                _rhs0 = self.lattice.rhs.to_numpy()[0]
                _aic00 = self.lattice.AIC.to_numpy()[0, 0]
                _g0 = gamma_np[0]
                _log_debug.debug("[SIGN CHECK - step 1]")
                _log_debug.debug(f"  normals[0]   = {_n0}")
                _log_debug.debug(f"  V_kin[0]     = {_vc0}")
                _log_debug.debug(f"  V_ext[0]     = {_ve0}")
                _log_debug.debug(f"  RHS[0]       = {_rhs0:.6f}")
                _log_debug.debug(f"  AIC[0,0]     = {_aic00:.6f}")
                _log_debug.debug(f"  gamma[0]     = {_g0:.6f}")
                _log_debug.debug(f"  RHS/AIC[0,0] = {_rhs0 / _aic00:.6f}  (should equal gamma[0])")
            except Exception as _e:
                _log_debug.debug(f"[SIGN CHECK] failed: {_e}")
        self._debug_sign_done = True

    def solve(
        self,
        V_external: np.ndarray | None = None,
        dt: float | None = None,
        save_old: bool = True,
        coupled: bool = False,
    ) -> np.ndarray:
        """
        Solve VLM system for circulation.

        Args:
            V_external: Total external velocity field at collocation points (N x 3).
                       If None, assumes data is already in lattice.external_velocity (GPU-resident).
            dt: Time step size (optional, used for wake shedding updates if needed)
            save_old: If True (default), saves current circulation to circulation_old
                     before solving. Set to False for re-solves within the same step.
            coupled: If True, uses bound-only AIC for coupling with VPM particles.

        Returns:
            Computed circulation (gamma) array
        """
        _t0 = time.time()
        if not self._mesh_generated:
            self.generate_mesh()
        n_panels = self.lattice.num_panels
        if V_external is not None:
            if V_external.ndim == 1 and V_external.shape[0] == 3:
                V_external = np.tile(V_external, (n_panels, 1))
            if V_external.shape[0] != n_panels:
                raise ValueError(
                    f"V_external shape {V_external.shape} does not match panels {n_panels}"
                )
            self.lattice.set_external_velocity(V_external)
            V_ext_np = V_external
        else:
            V_ext_np = self.lattice.external_velocity.to_numpy()[:n_panels]
        if self.lattice.has_kinematic_velocity():
            V_rel = V_ext_np - self.lattice.get_kinematic_velocity()
        else:
            V_rel = V_ext_np
        _t1 = time.time()
        self.update_trailing_directions(V_rel)
        _t2 = time.time()
        # Near-wake offset: one convection length downstream (U_ref * dt).
        # Closes the gap between the TE and the first free wake particle so the
        # bound solve "sees" its own implicit near-wake panel (canonical UVLM-VPM).
        if coupled and dt is not None:
            _uref = getattr(self, "_last_U_ref", None)
            _wo = (_uref * dt) if _uref is not None else np.zeros(3)
        else:
            _wo = np.zeros(3)
        _wake_offset = ti.Vector([float(_wo[0]), float(_wo[1]), float(_wo[2])])
        compute_AIC_matrix(
            self.lattice.collocation,
            self.lattice.vortex_points,
            self.lattice.corners,
            self.lattice.normals,
            self.lattice.is_TE_panel,
            self.lattice.te_index,
            self.lattice.AIC,
            self.lattice.num_panels,
            self.epsilon,
            coupled_mode=1 if coupled else 0,
            wake_offset=_wake_offset,
        )
        _t3 = time.time()
        if V_external is not None:
            self.lattice.set_external_velocity(V_external)
        compute_RHS_coupled(
            self.lattice.collocation,
            self.lattice.normals,
            self.lattice.external_velocity,
            self.lattice.kinematic_velocity,
            self.lattice.rhs,
            self.lattice.num_panels,
        )
        _t4 = time.time()
        if save_old:
            self.lattice.save_old_circulation()
            self._vlm_step_count = getattr(self, "_vlm_step_count", 0) + 1
        if coupled and save_old:
            self._augment_starting_vortex()
        gamma_np = self._run_linear_solver(n_panels)
        self._do_debug_sign_check(gamma_np)
        _t5 = time.time()
        total_time = _t5 - _t0
        if total_time > 1.0:
            print("-" * 60)
            print("VLM STEP PERFORMANCE")
            print("-" * 60)
            print(f"  Influence Matrix         : {(_t3 - _t2):.3e} s")
            print(f"  Boundary Condition       : {(_t4 - _t3):.3e} s")
            print(f"  Linear Solver            : {(_t5 - _t4):.3e} s")
            print(f"  Total                    : {total_time:.3e} s")
            print("-" * 60)
        self._solved = True
        return gamma_np

    def compute_postprocess(
        self,
        V_external_np: np.ndarray,
        U_ref: np.ndarray,
        density: float,
        dt: float | None = None,
        coupled: bool = False,
    ) -> None:
        """
        Compute derived quantities (velocities, pressures, forces).

        Args:
           V_external_np: External velocity (N, 3).
           U_ref: Reference velocity vector [ux, uy, uz] (m/s).
           density: Fluid density
           dt: Time step size (required for impulse method)
           coupled: Whether in coupled mode (bound only AIC)
        """
        # Ensure external velocity is set (idempotent if same array)
        self.lattice.set_external_velocity(V_external_np)

        U_ref_mag = np.linalg.norm(U_ref)
        if U_ref_mag < 1e-10:
            U_ref_mag = 1.0

        # 0. Update cumulative circulation field for Impulse method
        # Must be done after solve() but before force computation.
        self._compute_cumulative_circulation_cpu()

        # 1. Compute velocities at COLLOCATION points (for Cp)
        compute_induced_velocities(
            self.lattice.num_panels,
            self.lattice.collocation,
            self.lattice.vortex_points,
            self.lattice.circulation,
            self.lattice.velocity,
            self.lattice.external_velocity,
        )

        # 2. Compute pressure coefficients (using collocation velocity)
        compute_pressure_coefficients(
            self.lattice.velocity,
            self.lattice.pressure_coefficient,
            self.lattice.num_panels,
            float(U_ref_mag**2),
        )

        # 3. Compute panel forces
        # smooth_kj=1 in coupled mode cancels the 2Δt oscillation introduced
        # by the explicit VPM-VLM coupling (γ alternates every step because
        # the near-field particle geometry alternates).
        smooth_kj = 1 if (coupled and getattr(self.force, "kj_smoothing", True)) else 0

        # 4. Compute velocities at BOUND VORTEX midpoints (for Forces)
        # When kj_smoothing is active, V_bound must use the same smoothed gamma
        # as the force kernel (0.5*(γ + γ_old)); otherwise the 2Δt oscillation
        # in the raw gamma propagates through V_bound into the KJ forces even
        # though the force kernel smooths gamma itself.
        if smooth_kj:
            apply_gamma_smooth(
                self.lattice.circulation,
                self.lattice.circulation_old,
                self.lattice.circulation_smooth,
                self.lattice.num_panels,
            )
            circ_for_vbound = self.lattice.circulation_smooth
        else:
            circ_for_vbound = self.lattice.circulation
        compute_induced_velocities_at_bound(
            self.lattice.num_panels,
            self.lattice.bound_midpoints,
            self.lattice.vortex_points,
            self.lattice.corners,
            self.lattice.te_index,
            circ_for_vbound,
            self.lattice.bound_velocity,
            self.lattice.external_velocity,
            1 if coupled else 0,
        )

        if self.force.method == "IMPULSE":
            # Impulse method (Unsteady Bernoulli: KJ + Added Mass)
            compute_panel_forces_impulse_coupled(
                self.lattice.bound_velocity,
                self.lattice.vortex_points,
                self.lattice.circulation,
                self.lattice.circulation_old,
                self.lattice.circulation_old2,
                self.lattice.cumulative_circulation,  # Current mu
                self.lattice.cumulative_circulation_old,  # mu_{n-1}
                self.lattice.cumulative_circulation_old2,  # mu_{n-2}
                self.lattice.kinematic_velocity,
                self.lattice.forces,
                self.lattice.areas,
                self.lattice.normals,
                self.lattice.panel_is_mirrored,
                self.lattice.forces_kj,
                self.lattice.forces_unsteady,
                self.lattice.num_panels,
                density,
                dt if dt is not None else 1.0,
                getattr(self, "_vlm_step_count", 0),
                self.force.impulse_startup_steps,
                self.force.impulse_order,
                smooth_kj,
            )
        else:
            # Standard Kutta-Joukowski (Quasi-steady)
            compute_panel_forces_coupled(
                self.lattice.bound_velocity,
                self.lattice.vortex_points,
                self.lattice.circulation,
                self.lattice.circulation_old,
                self.lattice.kinematic_velocity,
                self.lattice.forces,
                self.lattice.num_panels,
                density,
                U_ref_mag,
                smooth_kj,
            )

    def _resolve_uref(self, U_ref: np.ndarray | None, n_panels: int) -> np.ndarray:
        """Return a valid reference velocity, auto-computed if not provided."""
        if U_ref is not None:
            return np.asarray(U_ref, dtype=float)
        if self.U_inf is not None and np.linalg.norm(self.U_inf) > 1e-10:
            return self.U_inf
        V_kinematic = self._get_active_kinematic_velocity()
        V_bg = (
            np.mean(self.lattice.external_velocity.to_numpy()[:n_panels], axis=0)
            if n_panels > 0
            else np.zeros(3)
        )
        U_resolved = V_bg - V_kinematic
        if np.linalg.norm(U_resolved) < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        return U_resolved

    def _decompose_wind_axes(
        self, F_total: np.ndarray, U_ref_mag: float, U_ref: np.ndarray
    ) -> tuple[float, float, float]:
        """Decompose total force into lift, drag, side-force in wind axes."""
        if U_ref_mag > 1e-10:
            V_hat = U_ref / U_ref_mag
            z_hat = np.array([0.0, 0.0, 1.0])
            L_hat = z_hat - np.dot(z_hat, V_hat) * V_hat
            L_hat_norm = np.linalg.norm(L_hat)
            L_hat = L_hat / L_hat_norm if L_hat_norm > 1e-10 else np.array([0.0, 0.0, 1.0])
            C_hat = np.cross(V_hat, L_hat)
            return (
                float(np.dot(F_total, L_hat)),
                float(np.dot(F_total, V_hat)),
                float(np.dot(F_total, C_hat)),
            )
        Fx, _Fy, Fz = F_total
        return -float(Fz), float(Fx), float(_Fy)

    def _compute_force_moments(
        self, forces_np: np.ndarray, c_ref: float | None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], np.ndarray]:
        """Compute moments about reference center and quarter-chord."""
        bound_midpoints_np = self.lattice.bound_midpoints.to_numpy()[: self.lattice.num_panels]
        kinematics = self._get_active_kinematics()
        current_pos = np.zeros(3)
        current_rot = np.eye(3)
        if kinematics is not None and hasattr(kinematics, "current_position"):
            current_pos = kinematics.current_position
            if hasattr(kinematics, "current_orientation"):
                current_rot = kinematics.current_orientation
        r_ref_local = np.array(self.aircraft.refs.get("rcenter", [0.0, 0.0, 0.0]))
        r_ref = current_pos + current_rot @ r_ref_local
        M_total = np.sum(np.cross(bound_midpoints_np - r_ref, forces_np), axis=0)
        if c_ref is None:
            c_ref = float(self.aircraft.refs.get("chord", 1.0))
        r_c4_local = np.array([0.25 * c_ref, 0.0, 0.0])
        r_ref_c4 = current_pos + current_rot @ r_c4_local
        M_c4_total = np.sum(np.cross(bound_midpoints_np - r_ref_c4, forces_np), axis=0)
        return tuple(M_total), tuple(M_c4_total), r_ref  # type: ignore[return-value]

    def _build_force_coefficients(  # noqa: PLR0913
        self,
        L: float,
        D: float,
        C: float,
        Fx: float,
        Fy: float,
        Fz: float,
        M: tuple[float, float, float],
        M_c4: tuple[float, float, float],
        r_ref: np.ndarray,
        density: float,
        U_ref_mag: float,
        S_ref: float | None,
        c_ref: float | None,
        b_ref: float | None,
    ) -> dict[str, float]:
        """Build normalised coefficient dict from raw forces/moments."""
        Mx, My, Mz = M
        Mx_c4, My_c4, Mz_c4 = M_c4
        q = 0.5 * density * U_ref_mag**2
        if S_ref is None:
            S_ref = float(self.aircraft.refs.get("area", 1.0))
        if b_ref is None:
            b_ref = float(self.aircraft.refs.get("span", 1.0))
        if c_ref is None:
            c_ref = float(self.aircraft.refs.get("chord", 1.0))
        norm_f = q * S_ref
        norm_m = q * S_ref * c_ref
        norm_l = q * S_ref * b_ref
        if q > 1e-10 and norm_f > 1e-10:
            CL = L / norm_f
            CD = D / norm_f
            CC = C / norm_f
            CFx = Fx / norm_f
            CFy = Fy / norm_f
            CFz = Fz / norm_f
            Cl, Cm, Cn = Mx / norm_l, My / norm_m, Mz / norm_l
            Cl_c4, Cm_c4, Cn_c4 = Mx_c4 / norm_l, My_c4 / norm_m, Mz_c4 / norm_l
        else:
            CL = CD = CC = CFx = CFy = CFz = 0.0
            Cl = Cm = Cn = Cl_c4 = Cm_c4 = Cn_c4 = 0.0
        return {
            "Fx": Fx,
            "Fy": Fy,
            "Fz": Fz,
            "L": L,
            "D": D,
            "C": C,
            "Mx": Mx,
            "My": My,
            "Mz": Mz,
            "CFx": CFx,
            "CFy": CFy,
            "CFz": CFz,
            "CL": CL,
            "CD": CD,
            "CC": CC,
            "Cl": Cl,
            "Cm": Cm,
            "Cn": Cn,
            "Cl_c4": Cl_c4,
            "Cm_c4": Cm_c4,
            "Cn_c4": Cn_c4,
            "q": q,
            "S_ref": S_ref,
            "r_ref": r_ref,
        }

    def compute_forces(
        self,
        density: float,
        U_ref: np.ndarray | None = None,
        S_ref: float | None = None,
        c_ref: float | None = None,
        b_ref: float | None = None,
    ) -> dict[str, float]:
        """
        Compute integrated aerodynamic forces and moments.

        CL, CD, CC are normalized by the dynamic pressure and reference area:
        CL = L / (0.5 * rho * S_ref * U_ref²)

        Args:
           density: Fluid density (kg/m³)
           U_ref: Reference velocity vector [ux, uy, uz] (for coefficients and L/D axes).
                  If None, auto-computed from U_inf or kinematics.
           S_ref: Reference area (m²). If None, uses aircraft defaults.
           c_ref: Reference chord (m). If None, uses aircraft defaults.
           b_ref: Reference span (m). If None, uses aircraft defaults.

        Returns:
            Dictionary with force components and coefficients
        """
        if not self._solved:
            raise RuntimeError("Must solve system before computing forces")
        forces_np = self.lattice.get_forces()
        F_total = np.sum(forces_np, axis=0)
        Fx, Fy, Fz = F_total
        U_ref = self._resolve_uref(U_ref, self.lattice.num_panels)
        U_ref_mag = np.linalg.norm(U_ref)
        L, D, C = self._decompose_wind_axes(F_total, U_ref_mag, U_ref)
        M, M_c4, r_ref = self._compute_force_moments(forces_np, c_ref)
        return self._build_force_coefficients(
            L, D, C, Fx, Fy, Fz, M, M_c4, r_ref, density, U_ref_mag, S_ref, c_ref, b_ref
        )

    def compute_total_circulation(self) -> np.ndarray:
        """
        Compute total bound circulation vector from VLM panels.

        This is the circulation contribution from the bound vortex elements.
        Each horseshoe vortex contributes Γ * l where l is the bound leg vector.

        Returns:
            np.ndarray: Total circulation vector [Γx, Γy, Γz] [m²/s]

        Note:
            For impulse-consistent force evaluation, this must be combined with
            wake circulation from VPM particles.
        """
        if not self._solved:
            return np.zeros(3)

        n_panels = self.lattice.num_panels
        gamma = self.lattice.circulation.to_numpy()[:n_panels]
        vortex_pts = self.lattice.vortex_points.to_numpy()[:n_panels]

        # Bound leg vector: the spanwise vortex segment at the quarter chord,
        # vortex_pts[:, 2] - vortex_pts[:, 1].  (Using panel corners instead would
        # give the chordwise vector, which is not the bound leg.)
        l_vec = vortex_pts[:, 2] - vortex_pts[:, 1]

        # Total circulation: sum(Γ_i * l_i)
        circulation_total = np.sum(gamma[:, np.newaxis] * l_vec, axis=0)

        return circulation_total

    def compute_bound_impulse(self, density: float) -> np.ndarray:
        r"""
        Compute hydrodynamic impulse of bound circulation (horseshoe system).

        The bound impulse is computed by integrating x cross vorticity over the
        bound segments and trailing filaments up to the trailing edge.
        This is consistent with the fluid-bound impulse convention used in VPM.

        .. math::
            \mathbf{I}_{bound} = \frac{density}{2} \sum_i \int \mathbf{x} \times \boldsymbol{\omega}_i \, dl

        Args:
            density: Fluid density [kg/m³]

        Returns:
            np.ndarray: Bound impulse vector [Ix, Iy, Iz] [kg·m²/s]
        """
        if not self._solved:
            return np.zeros(3)

        n_panels = self.lattice.num_panels
        gamma = self.lattice.circulation.to_numpy()[:n_panels]
        vortex_pts = self.lattice.vortex_points.to_numpy()[:n_panels]
        corners = self.lattice.corners.to_numpy()[:n_panels]

        # 1. Bound Vortex Segment (V2 -> V3 at 1/4 chord)
        v2 = vortex_pts[:, 1]
        v3 = vortex_pts[:, 2]

        # Midpoint and vector
        x_b = 0.5 * (v2 + v3)
        l_b = v3 - v2

        # 2. Trailing Leg Segments (Connect Bound Vortex to TE)
        # We must account for the impulse of the vortex filaments
        # connecting the bound vortex (1/4 chord) to the TE.

        # TE points: S (Left TE) and R (Right TE) in panel orientation:
        # P=0 (front left), Q=1 (front right), R=2 (back right), S=3 (back left)
        # So back edge is R - S.
        s_pts = corners[:, 3]
        r_pts = corners[:, 2]

        # Left Leg: Inflow from TE to Bound (S -> V2)
        x_l = 0.5 * (s_pts + v2)
        l_l = v2 - s_pts

        # Right Leg: Outflow from Bound to TE (V3 -> R)
        x_r = 0.5 * (v3 + r_pts)
        l_r = r_pts - v3

        # 3. Vectorized Impulse Calculation
        # I = 0.5 * density * sum( x_i x (Gamma_i * l_i) )
        g_vec = gamma[:, None]  # Shape (N, 1) for broadcasting

        # Cross products (Shape N, 3)
        term_b = np.cross(x_b, g_vec * l_b)
        term_l = np.cross(x_l, g_vec * l_l)
        term_r = np.cross(x_r, g_vec * l_r)

        # Sum over all panels
        total_impulse = np.sum(term_b + term_l + term_r, axis=0)

        return 0.5 * density * total_impulse

    def _compute_one_surface_forces(
        self,
        surface_name: str,
        forces_np: np.ndarray,
        wing_panel_ranges: dict[str, tuple[int, int]],
        V_hat: np.ndarray,
        norm_f: float,
    ) -> dict[str, float]:
        """Compute wind-axis forces for a single surface."""
        F_surface = np.zeros(3)
        panel_count = 0
        for wing_uid, (start_idx, end_idx) in wing_panel_ranges.items():
            if wing_uid.startswith(surface_name + "_") or wing_uid == surface_name:
                F_surface += np.sum(forces_np[start_idx:end_idx], axis=0)
                panel_count += end_idx - start_idx
        Fx, Fy, Fz = F_surface
        D = np.dot(F_surface, V_hat)
        L_vec = F_surface - D * V_hat
        L = np.linalg.norm(L_vec)
        if np.dot(L_vec, np.array([0, 0, 1])) < 0:
            L = -L
        CL = L / norm_f if norm_f > 1e-10 else 0.0
        CD = D / norm_f if norm_f > 1e-10 else 0.0
        return {
            "L": L,
            "D": D,
            "Fx": Fx,
            "Fy": Fy,
            "Fz": Fz,
            "CL": CL,
            "CD": CD,
            "panel_count": panel_count,
        }

    def compute_per_surface_forces(
        self,
        density: float,
        U_ref: np.ndarray | None = None,
        S_ref: float | None = None,
        c_ref: float | None = None,
        b_ref: float | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute forces for each individual surface.

        All coefficients are normalized by dynamic pressure and reference area:
        CL = L / (0.5 * rho * S_ref * U_ref²)

        Args:
            density: Fluid density (kg/m³)
            U_ref: Reference velocity vector [ux, uy, uz]
            S_ref: Reference area (m²). If None, uses aircraft defaults.
            c_ref: Reference chord (m). If None, uses aircraft defaults.
            b_ref: Reference span (m). If None, uses aircraft defaults.

        Returns:
            Dictionary mapping surface name to force dictionary
        """
        if not self._solved:
            return {}
        forces_np = self.lattice.get_forces()
        U_ref = self._resolve_uref(U_ref, self.lattice.num_panels)
        U_ref_mag = np.linalg.norm(U_ref)
        wing_panel_ranges = self._build_wing_panel_ranges()
        q = 0.5 * density * U_ref_mag**2
        norm_f = q * S_ref if S_ref is not None else q * float(self.aircraft.refs.get("area", 1.0))
        V_hat = U_ref / U_ref_mag if U_ref_mag > 1e-10 else np.array([1.0, 0.0, 0.0])
        return {
            name: self._compute_one_surface_forces(
                name, forces_np, wing_panel_ranges, V_hat, norm_f
            )
            for name in self.surfaces
        }

    def absorb_particles(self, particles, tolerance: float = 0.05) -> int:
        """
        Detect and remove particles colliding with the lifting surface.

        In VLM, the lifting surfaces are thin (zero thickness). Vortex particles
        that pass through the surface can cause numerical singularities and
        unphysical forces. This method detects collisions and removes particles.

        Args:
            particles: The VPM particles object (must have position, _removal_tags fields)
            tolerance: Collision distance threshold [m].
                      Should be roughly equal to the particle core radius or
                      boundary layer thickness. Typical values: 0.05-0.2 * chord

        Returns:
            int: Number of particles removed

        Example:
            >>> # After particle advection, check for collisions
            >>> n_removed = vlm_solver.absorb_particles(vpm.particles, tolerance=0.1)
        """
        if not self._mesh_generated:
            return 0

        n_panels = self.lattice.num_panels
        n_particles = particles.number_of_particles

        if n_panels == 0 or n_particles == 0:
            return 0

        # Reset removal tags
        particles._removal_tags.fill(0)

        # Run GPU Collision Detection
        detect_surface_collisions_kernel(
            particles.position,
            particles._removal_tags,
            self.lattice.corners,
            self.lattice.normals,
            n_particles,
            n_panels,
            float(tolerance),
        )

        # Count tagged particles
        tags = particles._removal_tags.to_numpy()[:n_particles]
        n_hits = int(np.sum(tags))

        if n_hits > 0:
            keep_mask = tags == 0
            n_keep = int(keep_mask.sum())

            if n_keep == 0:
                particles.number_of_particles = 0
                particles.sync_device_counter()
                particles._cached_step = -1
                return n_hits

            new_position = particles.position.to_numpy()[:n_particles][keep_mask]
            new_velocity = particles.velocity.to_numpy()[:n_particles][keep_mask]
            new_circulation = particles.circulation.to_numpy()[:n_particles][keep_mask]
            new_radius = particles.radius.to_numpy()[:n_particles][keep_mask]
            new_volume = particles.volume.to_numpy()[:n_particles][keep_mask]
            new_viscosity = particles.viscosity.to_numpy()[:n_particles][keep_mask]
            new_viscosity_turbulent = particles.viscosity_turbulent.to_numpy()[:n_particles][
                keep_mask
            ]
            new_group_id = particles.group_id.to_numpy()[:n_particles][keep_mask]
            new_zone_id = particles.zone_id.to_numpy()[:n_particles][keep_mask]
            new_velocity_gradient = particles.velocity_gradient.to_numpy()[:n_particles][keep_mask]
            new_strain_rate = particles.strain_rate.to_numpy()[:n_particles][keep_mask]

            particles.replace_from_numpy(
                position=new_position,
                velocity=new_velocity,
                circulation=new_circulation,
                radius=new_radius,
                volume=new_volume,
                viscosity=new_viscosity,
                viscosity_turbulent=new_viscosity_turbulent,
                group_id=new_group_id,
                zone_id=new_zone_id,
                velocity_gradient=new_velocity_gradient,
                strain_rate=new_strain_rate,
            )

            print(f"   (VLM) Absorbed {n_hits} particles impinging on surface.")

        return n_hits

    def log_forces_table(self, density: float, U_ref: np.ndarray | None = None) -> dict[str, float]:
        """
        Log VLM forces in a formatted table matching VPM diagnostics style.

        Prints per-surface forces and total forces with descriptions.

        Args:
            density: Fluid density (kg/m^3)
            U_ref: Reference velocity vector
        """
        print("\n" + "-" * 60)
        print("VLM AERODYNAMIC FORCES")
        print("-" * 60)
        method_name = (
            "Kutta-Joukowski" if self.force.method == "KUTTA_JOUKOWSKI" else "Impulse-Based"
        )
        print(f"  Surface forces computed using {method_name} method:")
        print("    L = Lift force (perpendicular to freestream)")
        print("    D = Drag force (parallel to freestream)")
        print("    CL, CD = Lift and drag coefficients")
        print()

        # Get total forces
        total_forces = self.compute_forces(density, U_ref)

        # Get per-surface forces
        surface_forces = self.compute_per_surface_forces(density, U_ref)

        if len(surface_forces) > 1:
            # Print table header for per-surface
            print(
                f"  {'Surface':<15} {'L [N]':>12} {'D [N]':>12} {'CL':>10} {'CD':>10} {'Panels':>8}"
            )
            print(f"  {'-' * 15} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 8}")

            for surf_name, forces in surface_forces.items():
                print(
                    f"  {surf_name:<15} {forces['L']:>12.3f} {forces['D']:>12.3f} "
                    f"{forces['CL']:>10.3f} {forces['CD']:>10.3f} {forces['panel_count']:>8}"
                )

            print(f"  {'-' * 15} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 8}")

        # Print totals
        L = total_forces.get("L", 0.0)
        D = total_forces.get("D", 0.0)
        CL = total_forces.get("CL", 0.0)
        CD = total_forces.get("CD", 0.0)
        CC = total_forces.get("CC", 0.0)
        L_D = L / D if abs(D) > 1e-10 else float("inf")

        print(f"  {'TOTAL':<15} {L:>12.3f} {D:>12.3f} {CL:>10.3f} {CD:>10.3f}")
        print()
        print(f"  Lift/Drag Ratio (L/D)    : {L_D:.2f}")
        print(f"  Cross-force Coeff (CC)   : {CC:.3f}")

        # Moments
        Cl = total_forces.get("Cl", 0.0)
        Cm = total_forces.get("Cm", 0.0)
        Cn = total_forces.get("Cn", 0.0)
        Cm_c4 = total_forces.get("Cm_c4", 0.0)

        print()
        print("  Moment Coefficients:")
        print(f"    Roll (Cl)    : {Cl:>12.3f}")
        print(f"    Pitch (Cm)   : {Cm:>12.3f}")
        print(f"    Pitch (Cm_c4): {Cm_c4:>12.3f} (About c/4)")
        print(f"    Yaw (Cn)     : {Cn:>12.3f}")

        if "r_ref" in total_forces:
            r = total_forces["r_ref"]
            print(f"    Ref Center   : [{r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}]")

        print("-" * 60, flush=True)

        return total_forces

    def save_results(self, filename: str, flow_time: float = 0.0) -> None:
        """
        Save VLM results to VTK file.

        Args:
            filename: Output filename (without extension)
            flow_time: Simulation time for Paraview TimeValue
        """
        if not self._solved:
            print("Warning: System not solved, saving mesh only")

        self.lattice.save_vtk(filename, flow_time=flow_time)

    def advance(
        self,
        dt: float,
        V_external: np.ndarray,
        density: float = 1.0,
        U_ref: np.ndarray | None = None,
        logging_frequency: int | None = None,
        step: int = 0,
        time: float | None = None,
    ) -> dict[str, np.ndarray] | None:
        """
        Advance VLM-VPM coupled simulation by one time step.

        Args:
            dt: Time step size (s)
            V_external: Total external velocity field at collocation points (N, 3)
            density: Fluid density
            U_ref: Reference velocity vector (defaults to auto-computed)
            logging_frequency: Print forces every N steps (None=use solver default)
            step: Current time step number
            time: Current simulation time (s) - prevents drift if provided

        Returns:
             Dictionary with new wake particles or None
        """
        if not self._mesh_generated:
            self.generate_mesh()

        # 1. Advance kinematics (handles both single and multi-surface)
        # advance_time() internally handles multi-surface kinematics
        self.advance_time(dt, current_time=time)

        # 2. Solve VLM system
        self.solve(V_external, dt)

        # Determine reference values if not provided
        if U_ref is None:
            V_kinematic = self._get_active_kinematic_velocity()
            V_bg = np.mean(V_external, axis=0) if len(V_external) > 0 else np.zeros(3)
            U_ref = V_bg - V_kinematic
            if np.linalg.norm(U_ref) < 1e-10:
                U_ref = np.array([1.0, 0.0, 0.0])

        np.linalg.norm(U_ref)

        # Compute postprocess (velocities, forces) to enable logging
        self.compute_postprocess(V_external, U_ref, density, dt=dt)

        # Automatically compute and cache forces
        self._last_forces = self.compute_forces(density, U_ref)
        self._last_U_ref = U_ref

        # 3. Log forces if requested
        log_freq = self.logging_frequency if logging_frequency is None else int(logging_frequency)
        if log_freq > 0 and step % log_freq == 0:
            try:
                self.log_forces_table(density, U_ref)
            except Exception as e:
                print(f"   (Warning) Could not compute VLM forces: {e}")

        return self._compute_wake_particles(dt, U_ref)

    def _compute_kinematic_offset(self, dt: float) -> np.ndarray:
        """Compute kinematic position offset for wake particles."""
        if self.kinematics is None:
            return np.zeros(3)
        t = self._current_time if self._current_time is not None else 0.0
        V_kinematic = self.kinematics.get_velocity(t)
        return -0.5 * V_kinematic * dt

    def _create_particle_builder(
        self,
        V_mag,
        dt,
        V_inf,
        V_particle_vel,
        pos_offset,
        all_positions,
        all_strengths,
        all_radii,
        all_volumes,
        all_velocities,
    ):
        """Create closure for building wake particles from TE geometry."""

        def add_particles_bulk(p_pos, p_gamma, p_dir, p_span):
            count = len(p_gamma)
            if count == 0:
                return

            # Radius: sigma = 1.5 * max(V_mag * dt, span / 2)
            convection_dist = V_mag * dt
            spanwise_scale = p_span / 2.0
            sigma = 1.5 * np.maximum(convection_dist, spanwise_scale)

            # Volume
            l_te = V_mag * dt
            vol = np.pi * (p_span / 2) ** 2 * l_te

            # Strength: alpha = Gamma * dL * (-direction)
            strength_vec = (p_gamma * l_te)[:, None] * (-p_dir)

            all_positions.append(p_pos + pos_offset)
            all_strengths.append(strength_vec)
            all_radii.append(sigma)
            all_volumes.append(vol)
            all_velocities.append(np.tile(V_particle_vel, (count, 1)))

        return add_particles_bulk

    def _get_te_geometry(self, te_indices, corners, trailing_dirs, U_ref: np.ndarray):
        """Extract and process trailing edge geometry."""
        Ap = corners[te_indices, 3]  # Left TE
        Bp = corners[te_indices, 2]  # Right TE

        infDA = trailing_dirs[te_indices, 0]
        infDB = trailing_dirs[te_indices, 1]

        V_mag = np.linalg.norm(U_ref)
        V_dir = U_ref / V_mag if V_mag > 1e-10 else np.array([1.0, 0.0, 0.0])

        # Normalize directions with safe division
        normA = np.linalg.norm(infDA, axis=1, keepdims=True)
        infDA = np.where(normA > EPSILON, infDA / normA, V_dir)

        normB = np.linalg.norm(infDB, axis=1, keepdims=True)
        infDB = np.where(normB > EPSILON, infDB / normB, V_dir)

        span_len = np.linalg.norm(Bp - Ap, axis=1)

        return Ap, Bp, infDA, infDB, span_len

    def _fill_segment_cumulative(
        self,
        gamma: np.ndarray,
        gamma_cumulative: np.ndarray,
        panel_idx: int,
        nc: int,
        ns: int,
    ) -> int:
        """Fill gamma_cumulative for one segment (original or mirror). Returns next panel_idx."""
        n_seg = nc * ns
        if panel_idx + n_seg <= len(gamma_cumulative):
            seg_gamma = gamma[panel_idx : panel_idx + n_seg]
            cumsum = np.cumsum(seg_gamma.reshape((ns, nc)), axis=1)
            gamma_cumulative[panel_idx : panel_idx + n_seg] = cumsum.ravel()
        return panel_idx + n_seg

    def _compute_cumulative_circulation_cpu(self) -> None:
        """
        Compute cumulative circulation for all panels using segment information.

        This replaces the Taichi kernel version which relies on neighbor_indices
        that may be incorrectly set for tapered/swept wings.

        For each spanwise station, the cumulative circulation is the sum of all
        chordwise panel circulations. This is what gets shed as trailing vortices.
        """
        n_panels = self.lattice.num_panels
        gamma = self.lattice.circulation.to_numpy()[:n_panels]
        gamma_cumulative = np.zeros(n_panels, dtype=np.float64)

        panel_idx = 0
        for _wing_uid, wing in self.aircraft.wings.items():
            for _seg_uid, seg in wing.segments.items():
                nc = seg.panels_chord
                ns = seg.panels_span
                panel_idx = self._fill_segment_cumulative(
                    gamma, gamma_cumulative, panel_idx, nc, ns
                )
                if wing.symmetry > 0:
                    panel_idx = self._fill_segment_cumulative(
                        gamma, gamma_cumulative, panel_idx, nc, ns
                    )

        # Upload to Taichi field for kernel access
        dtype_np = np.float32 if self.lattice.dtype == ti.f32 else np.float64
        gamma_full = np.zeros(self.lattice.max_panels, dtype=dtype_np)
        gamma_full[:n_panels] = gamma_cumulative
        self.lattice.cumulative_circulation.from_numpy(gamma_full)

    def _compute_segment_circulation(self, gamma, panel_idx, nc, ns) -> np.ndarray:
        """Compute cumulative circulation for segment."""
        seg_gamma = gamma[panel_idx : panel_idx + ns * nc]
        if len(seg_gamma) == ns * nc:
            return np.sum(seg_gamma.reshape((ns, nc)), axis=1)
        return np.zeros(ns)

    def _shed_te_particles(
        self, Ap, Bp, infDA, infDB, span_len, Gamma_cumulative, add_particles, ns
    ):
        """Shed wake particles from trailing edge."""
        # Left leg
        delta_gamma_A = np.zeros(ns)
        delta_gamma_A[0] = Gamma_cumulative[0]
        delta_gamma_A[1:] = Gamma_cumulative[1:] - Gamma_cumulative[:-1]

        mask_A = np.abs(delta_gamma_A) > EPSILON
        if np.any(mask_A):
            add_particles(Ap[mask_A], delta_gamma_A[mask_A], infDA[mask_A], span_len[mask_A])

        # Right leg (last panel only)
        last_gamma = -Gamma_cumulative[-1]
        if np.abs(last_gamma) > EPSILON:
            add_particles(
                Bp[ns - 1 : ns], np.array([last_gamma]), infDB[ns - 1 : ns], span_len[ns - 1 : ns]
            )

    def _process_segment_wake(
        self,
        segment,
        panel_idx,
        n_panels,
        gamma,
        corners,
        trailing_dirs,
        U_ref: np.ndarray,
        add_particles,
        nc,
        ns,
    ):
        """Process wake shedding for one segment."""
        # Get trailing edge indices
        j_indices = np.arange(ns)
        te_indices = panel_idx + j_indices * nc + (nc - 1)

        mask = te_indices < n_panels
        if not np.any(mask):
            return panel_idx + nc * ns

        te_indices = te_indices[mask]

        # Extract geometry
        Ap, Bp, infDA, infDB, span_len = self._get_te_geometry(
            te_indices, corners, trailing_dirs, U_ref
        )

        # Compute circulation
        Gamma_cumulative = self._compute_segment_circulation(gamma, panel_idx, nc, ns)

        # Shed particles
        self._shed_te_particles(Ap, Bp, infDA, infDB, span_len, Gamma_cumulative, add_particles, ns)

        return panel_idx + nc * ns

    def _compute_wake_particles(
        self,
        dt: float,
        U_ref: np.ndarray,
        V_particle_vel: np.ndarray = None,
        reset_buffer: bool = True,
    ) -> dict[str, np.ndarray] | None:
        """
        Compute wake particles using GPU kernel.

        Uses CUMULATIVE circulation (sum of all chordwise panels) for trailing vortex
        strength, which is the physical bound circulation at each spanwise station.

        For hover/rotating surfaces: Uses per-panel kinematic velocity stored in
        lattice.kinematic_velocity for convection, ensuring wake particles are
        convected in the physically correct direction at each TE panel.

        Args:
            dt: Time step size (s)
            U_ref: Reference velocity vector (m/s)
            V_particle_vel: Global initial convection velocity (m/s)
            reset_buffer: If True (default), resets the wake buffer count to zero
                         before shedding. Set to False if LEV particles were
                         already shed into the buffer this step.
        """
        n_panels = self.lattice.num_panels

        if n_panels == 0:
            return None

        V_mag = np.linalg.norm(U_ref)
        V_dir = U_ref / V_mag if V_mag > 1e-10 else np.array([1.0, 0.0, 0.0])

        # In hover mode (no freestream), U_ref_mag comes from tip speed.
        # The shedding kernel needs it for sigma/l_te sizing.
        # Use per-panel kinematic velocity magnitude instead of a single global value.
        if V_mag < 1e-10:
            V_mag = self._get_max_kinematic_speed()
            if V_mag < 1e-10:
                # Truly no motion — nothing to shed
                return None

        # For particle initial velocity: use per-panel kinematic velocity if available,
        # otherwise use the provided V_particle_vel or freestream
        V_particle = V_particle_vel if V_particle_vel is not None else V_dir * V_mag

        # Reset wake buffer if requested
        if reset_buffer:
            self.lattice.reset_wake_buffer()

        # Compute cumulative circulation BEFORE shedding using CPU method
        self._compute_cumulative_circulation_cpu()

        shedding_threshold = float(self.transverse_shedding_threshold)

        # Check if we have per-panel kinematic velocity (hover/rotating mode)
        # If so, use local kinematic velocity for convection at each TE panel
        use_local_convection = self.lattice.has_kinematic_velocity()

        if use_local_convection:
            # Per-panel convection: pass external_velocity and normals so the kernel
            # can compute V_conv = V_external - V_kinematic (includes downwash)
            V_conv_ref = V_dir * V_mag
            shed_wake_particles_kernel(
                self.lattice.num_panels,
                dt,
                ti.Vector([V_conv_ref[0], V_conv_ref[1], V_conv_ref[2]]),
                ti.Vector([V_particle[0], V_particle[1], V_particle[2]]),
                self.sigma_factor,
                float(shedding_threshold),
                self.lattice.cumulative_circulation,
                self.lattice.cumulative_circulation_old,
                self.lattice.corners,
                self.lattice.neighbor_indices,
                self.lattice.is_TE_panel,
                self.lattice.panel_is_mirrored,
                self.lattice.panel_group_id,
                self.lattice.kinematic_velocity,
                self.lattice.external_velocity,
                self.lattice.normals,
                1,  # use_local_velocity = True
                self.lattice.wake_positions,
                self.lattice.wake_velocities,
                self.lattice.wake_strengths,
                self.lattice.wake_radii,
                self.lattice.wake_volumes,
                self.lattice.wake_group_ids,
                self.lattice.num_wake_particles,
            )
        else:
            # Global convection (forward flight mode): single V_convection for all panels
            V_conv_ref = V_dir * V_mag
            shed_wake_particles_kernel(
                self.lattice.num_panels,
                dt,
                ti.Vector([V_conv_ref[0], V_conv_ref[1], V_conv_ref[2]]),
                ti.Vector([V_particle[0], V_particle[1], V_particle[2]]),
                self.sigma_factor,
                float(shedding_threshold),
                self.lattice.cumulative_circulation,
                self.lattice.cumulative_circulation_old,
                self.lattice.corners,
                self.lattice.neighbor_indices,
                self.lattice.is_TE_panel,
                self.lattice.panel_is_mirrored,
                self.lattice.panel_group_id,
                self.lattice.kinematic_velocity,  # Unused when use_local=0
                self.lattice.external_velocity,  # Unused when use_local=0
                self.lattice.normals,  # Unused when use_local=0
                0,  # use_local_velocity = False
                self.lattice.wake_positions,
                self.lattice.wake_velocities,
                self.lattice.wake_strengths,
                self.lattice.wake_radii,
                self.lattice.wake_volumes,
                self.lattice.wake_group_ids,
                self.lattice.num_wake_particles,
            )

        # Check for buffer overflow and clamp to buffer size
        n_shed = self.lattice.num_wake_particles[None]
        n_buffer = self.lattice.wake_positions.shape[0]
        if n_shed > n_buffer:
            print(
                f"[WARNING] VLM Wake Buffer Overflow: {n_shed} particles generated > {n_buffer} buffer size. "
                f"{n_shed - n_buffer} particles were dropped."
            )
            # Clamp to avoid reading uninitialised memory downstream
            n_shed = n_buffer
            self.lattice.num_wake_particles[None] = n_buffer

        # We return a specific marker to indicate GPU data is ready
        return {"_gpu_transfer_ready": True}

    # Near-wake correction  (bypass VPM regularisation for shed particles)
    @staticmethod
    def _near_wake_biot_savart(
        targets: np.ndarray,
        sources: np.ndarray,
        strengths: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        """
        Compute velocity at *targets* due to vortex particles at *sources*
        using a lightly desingularised Biot-Savart law (algebraic core):

            V(x) = -1/(4π) Σ_j  (r_j × α_j) / (|r_j|² + ε²)^{3/2}

        This avoids the aggressive Winckelmans / Gaussian regularisation of
        the VPM kernel and provides near-exact velocities at distances
        r >> ε.

        Parameters
        ----------
        targets : (M, 3) collocation positions
        sources : (N, 3) particle positions
        strengths : (N, 3) particle strength vectors  α = ω × Volume
        epsilon : desingularisation radius [m]

        Returns
        -------
        (M, 3) induced velocity at each target
        """
        # r[i, j, :] = targets[i] - sources[j]          shape (M, N, 3)
        r = targets[:, None, :] - sources[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", r, r)  # (M, N)
        denom = (r2 + epsilon * epsilon) ** 1.5  # (M, N)

        # cross product  r × α  for every (target, source) pair
        cross = np.cross(r, strengths[None, :, :])  # (M, N, 3)

        # sum over sources, divide by 4π
        inv_4pi = 1.0 / (4.0 * np.pi)
        return -inv_4pi * np.einsum("ijk,ij->ik", cross, 1.0 / denom)

    # Implicit starting-vortex AIC augmentation
    @staticmethod
    def _fill_te_strip_segment(
        te_indices: list[int],
        strip_of: np.ndarray,
        panel_offset: int,
        nc: int,
        ns: int,
    ) -> int:
        """Register TE strip entries for one segment. Returns next panel_offset."""
        for j in range(ns):
            s = len(te_indices)
            te_indices.append(panel_offset + j * nc + nc - 1)
            strip_of[panel_offset + j * nc : panel_offset + (j + 1) * nc] = s
        return panel_offset + nc * ns

    def _build_te_strip_map(self) -> tuple[list[int], np.ndarray]:
        """Return (te_indices, strip_of) mapping each panel to its TE strip."""
        n = self.lattice.num_panels
        te_indices: list[int] = []
        strip_of = np.zeros(n, dtype=np.intp)
        panel_offset = 0
        for _wing_uid, wing in self.aircraft.wings.items():
            for _seg_uid, seg in wing.segments.items():
                nc = seg.panels_chord
                ns = seg.panels_span
                panel_offset = self._fill_te_strip_segment(
                    te_indices, strip_of, panel_offset, nc, ns
                )
                if wing.symmetry > 0 and panel_offset + nc * ns <= n:
                    panel_offset = self._fill_te_strip_segment(
                        te_indices, strip_of, panel_offset, nc, ns
                    )
        return te_indices, strip_of

    def _compute_sv_influence(
        self,
        te_indices: list[int],
        strip_of: np.ndarray,
        epsilon_sv: float,
    ) -> np.ndarray:
        """Compute K[i,s] = V_segment(coll_i, TeL_s→TeR_s) · n_i (N×N_TE)."""
        n = self.lattice.num_panels
        corners_np = self.lattice.corners.to_numpy()[:n]
        TeL_arr = np.array([corners_np[idx, 3] for idx in te_indices])
        TeR_arr = np.array([corners_np[idx, 2] for idx in te_indices])
        coll = self.lattice.collocation.to_numpy()[:n]
        nrml = self.lattice.normals.to_numpy()[:n]
        r_PA = coll[:, None, :] - TeL_arr[None, :, :]
        r_PB = coll[:, None, :] - TeR_arr[None, :, :]
        r_AB = TeR_arr[None, :, :] - TeL_arr[None, :, :]
        cross = np.cross(r_PA, r_PB)
        cross_norm2_safe = np.einsum("ijk,ijk->ij", cross, cross) + epsilon_sv**2
        r_PA_mag_safe = np.maximum(np.sqrt(np.einsum("ijk,ijk->ij", r_PA, r_PA)), epsilon_sv)
        r_PB_mag_safe = np.maximum(np.sqrt(np.einsum("ijk,ijk->ij", r_PB, r_PB)), epsilon_sv)
        factor = np.einsum(
            "ijk,ijk->ij", r_AB, r_PA / r_PA_mag_safe[:, :, None] - r_PB / r_PB_mag_safe[:, :, None]
        )
        coeff = factor / (4.0 * np.pi * cross_norm2_safe)
        K = np.einsum("ijk,ik->ij", cross * coeff[:, :, None], nrml)
        return K  # (N, N_TE) — unused n_te kept for type documentation

    def _apply_sv_augmentation(
        self,
        K: np.ndarray,
        strip_of: np.ndarray,
        te_indices: list[int],
    ) -> None:
        """Apply starting-vortex augmentation to AIC and RHS in-place.

        _compute_sv_influence returns K = +V_seg·n (upwash for TeL→TeR).
        The Kelvin constraint gives: (A − K_strip)·γ = RHS − K·Γ_old,
        so both AIC and RHS must be DECREMENTED by K.
        """
        n = self.lattice.num_panels
        aic_np = self.lattice.AIC.to_numpy()
        aic_np[:n, :n] -= K[:, strip_of]
        self.lattice.AIC.from_numpy(aic_np)
        cum_circ = self.lattice.cumulative_circulation.to_numpy()[:n]
        cum_circ_te = np.array([cum_circ[idx] for idx in te_indices])
        rhs_np = self.lattice.rhs.to_numpy()
        rhs_np[:n] -= K @ cum_circ_te
        self.lattice.rhs.from_numpy(rhs_np)

    def _augment_starting_vortex(self, epsilon_sv: float = 1e-3) -> None:
        """
        Augment the AIC matrix and RHS to implicitly account for the
        starting vortex shed at the trailing edge when circulation changes.

        Physics
        -------
        At every time step the VLM circulation changes by
        ΔΓ = Γ_cum_new − Γ_cum_old (cumulative / doublet circulation).
        A starting vortex with circulation −ΔΓ is shed as a spanwise
        filament from TeL to TeR at the trailing edge.

        Using the **finite vortex segment** Biot-Savart law (not the
        particle approximation, which overestimates by ~4× at close
        range), the influence coefficient is:

            K[i,s] = −V_segment(coll_i, TeL_s→TeR_s, Γ=1) · n_i

        We split ΔΓ into its unknown part (Γ_new, proportional to the
        panel unknowns γ_k) and its known part (Γ_old):

            AIC_mod[i,k] = AIC[i,k] + K[i, strip(k)]
            RHS_mod[i]   = RHS[i]   + Σ_s  K[i,s] × Γ_cum_old[s]

        The implicit treatment is stable because the starting vortex
        becomes part of the system matrix rather than the forcing.

        Parameters
        ----------
        epsilon_sv : float
            Cutoff radius [m] for the segment desingularisation.
        """
        te_indices, strip_of = self._build_te_strip_map()
        if not te_indices:
            return
        K = self._compute_sv_influence(te_indices, strip_of, epsilon_sv)
        self._apply_sv_augmentation(K, strip_of, te_indices)

    def _resolve_coupling_uref(self, config) -> np.ndarray:
        """Resolve the reference velocity used by coupled advance."""
        if hasattr(self, "U_inf") and self.U_inf is not None and np.linalg.norm(self.U_inf) > 1e-10:
            return self.U_inf
        if hasattr(config, "background_velocity") and config.background_velocity is not None:
            V_bg = np.array(config.background_velocity)
            if np.linalg.norm(V_bg) > 1e-10:
                return V_bg
        V_kinematic = self._get_active_kinematic_velocity()
        kin_mag = np.linalg.norm(V_kinematic)
        return -V_kinematic if kin_mag > 1e-10 else np.array([1.0, 0.0, 0.0])

    def _determine_include_freestream(self, config) -> bool:
        """Return False when kinematics are active but no background flow is set."""
        try:
            active_kin = self._get_active_kinematics()
            if active_kin is not None:
                V_bg = getattr(config, "background_velocity", np.zeros(3))
                if np.linalg.norm(V_bg) < 1e-10:
                    return False
        except Exception:
            pass
        return True

    def advance_coupled(
        self,
        particles,
        physics,
        config,
        dt: float,
        time_step: int,
        time: float | None = None,
    ) -> dict[str, np.ndarray] | None:
        """
        Advance VLM-VPM coupled simulation by one time step.

        Operation ordering — shed AFTER solve:

          1. Advance kinematics (move geometry)
          2. Compute U_ref for shedding / normalization
          3. Compute VPM-induced velocity at collocation
             (includes particles from previous steps, advected downstream)
          4. Solve VLM (coupled AIC — bound horseshoe + near-wake panel)
          5. Shed TE near-wake row (uses CLEAN post-solve cumulative Γ)
          6. Post-process forces (Kutta-Joukowski / impulse)
          7. Transfer the aged near-wake row to the free VPM wake
          8. Absorb colliding particles

        Why shed after solve:
        The shed vorticity must be spatially separated from the trailing-edge
        collocation point before it contributes to the VPM-induced velocity
        field.  By shedding AFTER the solve, the row shed at step N is
        evaluated at step N+1 when it sits ~V·dt downstream of the TE.
        """
        if not self._mesh_generated:
            self.generate_mesh()

        n_panels = self.lattice.num_panels

        # ──────────────────────────────────────────────────────────────
        # 1. Advance kinematics (move geometry to new positions)
        # ──────────────────────────────────────────────────────────────
        self.advance_time(dt, current_time=time)

        # ──────────────────────────────────────────────────────────────
        # 2. Determine reference / convection velocities early
        # ──────────────────────────────────────────────────────────────
        U_ref = self._resolve_coupling_uref(config)
        self._last_U_ref = U_ref
        include_fs = self._determine_include_freestream(config)

        # Convection velocity for particle initial velocity (use previous
        # step's external field; not yet updated for this step, but the
        # VPM solver will recompute correct velocities during advection).
        V_ext_prev = self.lattice.external_velocity.to_numpy()[:n_panels]
        V_bg_unsteady = np.mean(V_ext_prev, axis=0)
        bg_mag = np.linalg.norm(V_bg_unsteady)
        V_shed = V_bg_unsteady if bg_mag > 1e-3 else None

        # ──────────────────────────────────────────────────────────────
        # 3. Compute VPM-induced velocity at collocation points.
        #    Particles from previous steps are already convected downstream,
        #    providing spatial separation for the explicit coupling.
        # ──────────────────────────────────────────────────────────────
        physics.compute_target_velocities(
            particles,
            self.lattice.collocation,
            self.lattice.external_velocity,
            include_freestream=include_fs,
        )

        # ──────────────────────────────────────────────────────────────
        # 4. Solve VLM system (coupled AIC — bound horseshoe + near-wake)
        # ──────────────────────────────────────────────────────────────
        self.solve(V_external=None, dt=dt, coupled=True)

        # ──────────────────────────────────────────────────────────────
        # 5. Shed the TE near-wake row from the clean post-solve cumulative Γ.
        # ──────────────────────────────────────────────────────────────
        self.lattice.reset_wake_buffer()
        result = self._compute_wake_particles(dt, U_ref, V_particle_vel=V_shed, reset_buffer=False)

        # ──────────────────────────────────────────────────────────────
        # 6. Post-process forces (Kutta-Joukowski / impulse) from the solved Γ.
        # ──────────────────────────────────────────────────────────────
        V_ext_np = self.lattice.external_velocity.to_numpy()[:n_panels]
        self.compute_postprocess(V_ext_np, U_ref, self.density, dt=dt, coupled=True)
        self._last_forces = self.compute_forces(self.density, self._last_U_ref)

        # ──────────────────────────────────────────────────────────────
        # 7. Transfer the shed wake particles to the free VPM wake.
        # ──────────────────────────────────────────────────────────────
        if result and result.get("_gpu_transfer_ready"):
            num_shed = self.lattice.num_wake_particles[None]
            if num_shed > 0:
                particles.add_vortex_particles_from_fields_grouped(
                    num_shed,
                    self.lattice.wake_positions,
                    self.lattice.wake_velocities,
                    self.lattice.wake_strengths,
                    self.lattice.wake_radii,
                    self.lattice.wake_volumes,
                    self.lattice.wake_group_ids,
                    viscosity=self.viscosity,
                )

        # ──────────────────────────────────────────────────────────────
        # 9. Absorb particles that collide with lifting surfaces
        # ──────────────────────────────────────────────────────────────
        # tolerance = perpendicular distance from the panel plane [m].
        # For zero-thickness lifting surfaces (flat plates, thin airfoils),
        # absorption must be disabled (tolerance=0).  The collision kernel
        # checks `dist_perp < tolerance`, so tolerance=0 means no particle
        # can ever satisfy the condition.
        #
        # Background: VLM panels are infinitely thin.  Any positive tolerance
        # will erroneously remove wake particles that simply pass close to
        # the plate, since they can be arbitrarily close to the zero-thickness
        # plane.  With tolerance=0.03, ~32 particles were intermittently
        # removed at specific convective times (τ≈1 and τ≈5), creating
        # step-discontinuities in particle count that the BDF impulse-force
        # derivative amplifies into CL spikes.
        #
        # For thick bodies (3D geometry with enclosed volumes), set
        # _absorb_tolerance to a positive value (e.g. core_radius).
        absorb_tol = getattr(self, "_absorb_tolerance", 0.0)
        self.absorb_particles(particles, tolerance=absorb_tol)

        return None
