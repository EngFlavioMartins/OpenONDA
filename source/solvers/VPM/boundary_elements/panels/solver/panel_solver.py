"""
Upgraded PanelSolver orchestration.
==================
Main solver class managing PanelLattice, linear solvers, and force computation.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
import json
import logging
import os
from typing import Any, Literal

import numpy as np
import taichi as ti

from ..coupling import kinematics as kin_module
from ..coupling.separation import SeparationModel
from ..coupling.shedding import CouplingConfig, VortexShedder
from ..kernels.induced_velocity import compute_induced_velocity_kernel
from .influence import (
    build_AIC_matrix,
    build_AIC_matrix_dirichlet,
    compute_forces_bernoulli,
    compute_forces_impulse,
    compute_forces_kutta_joukowski,
    compute_pressure_bernoulli,
    compute_RHS_dirichlet_with_sources,
    compute_RHS_neumann_with_sources,
    compute_surface_velocities,
    compute_surface_velocities_with_sources,
)
from .kernels import panel_update_rotation_kernel, panel_update_translation_kernel
from .lattice import PanelLattice
from .linear_solvers import PanelBiCGSTABSolver, PanelScipySolver
from .mesh import add_body_from_mesh_stl
from .vtk_export import panel_mesh_to_vtp

logger = logging.getLogger("vpm")


@dataclass
class ForceConfig:
    method: Literal["BERNOULLI", "KUTTA_JOUKOWSKI", "IMPULSE"] = "BERNOULLI"
    impulse_order: Literal["BDF1", "BDF2"] = "BDF2"
    smoothing_window: int = 3

    @classmethod
    def bernoulli(cls):
        return cls(method="BERNOULLI")

    @classmethod
    def kutta_joukowski(cls):
        return cls(method="KUTTA_JOUKOWSKI")

    @classmethod
    def impulse_based(
        cls,
        order: Literal["BDF1", "BDF2"] = "BDF2",
        window: int = 3,
    ):
        return cls(method="IMPULSE", impulse_order=order, smoothing_window=window)


class PanelSolver:
    def __init__(
        self,
        max_panels: int = 10000,
        float_dtype: str = "f32",
        linear_solver: Literal["SCIPY", "BICGSTAB_GPU"] = "SCIPY",
        force_config: ForceConfig | None = None,
        coupling_config: CouplingConfig | None = None,
        LESP_crit: float = 0.11,
        separation_enabled: bool = False,
        bc_type: Literal["DIRICHLET", "NEUMANN"] = "DIRICHLET",
        density: float = 1.225,
        U_inf: np.ndarray | None = None,
        logging_frequency: int = 1,
    ):
        self.max_panels = max_panels
        self.float_dtype = float_dtype
        self.linear_solver_name = linear_solver
        self._coupling_config = coupling_config or CouplingConfig(
            lesp_crit=LESP_crit,
            separation_enabled=separation_enabled,
        )
        self.force_config = force_config or ForceConfig.bernoulli()
        self.lesp_crit = self._coupling_config.lesp_crit
        self.bc_type = bc_type
        self.density = density
        self.U_inf = None if U_inf is None else np.array(U_inf, dtype=np.float64)
        self.logging_frequency = max(1, int(logging_frequency))
        self.step = 0
        self._current_time = 0.0
        self._solved = False
        self._last_forces: dict[str, float] = {}
        self._last_U_ref: np.ndarray | None = None

        # Lazy initialization state
        self.lattice: PanelLattice | None = None
        self.shadow_lattice: PanelLattice | None = None  # For double-buffering if needed
        self.shedder: VortexShedder | None = None
        self.separation: SeparationModel | None = None
        self.solver_strategy = None

        # Fields (initialized lazily)
        self.AIC = None
        self.rhs = None
        self.panel_forces = None
        self.V_surface = None

        self.results = {"forces": [], "moments": [], "times": [], "diagnostics": []}
        self.is_initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of GPU fields and sub-solvers after Taichi init."""
        if self.lattice is not None:
            return

        # 1. Create Lattice
        self.lattice = PanelLattice(self.max_panels, self.float_dtype)
        ti_dtype = self.lattice.ti_dtype

        # 2. Create helper objects
        self.shedder = VortexShedder(
            LESP_crit=self.lesp_crit,
            max_panels=self.max_panels,
            float_dtype=self.float_dtype,
        )
        self.separation = SeparationModel(
            max_panels=self.max_panels,
            enabled=self._coupling_config.separation_enabled,
            float_dtype=self.float_dtype,
        )

        # 3. Create GPU fields
        self.AIC = ti.field(ti_dtype, shape=(self.max_panels, self.max_panels))
        self.rhs = ti.field(ti_dtype, shape=self.max_panels)
        self.panel_forces = ti.Vector.field(3, dtype=ti_dtype, shape=self.max_panels)
        self.V_surface = ti.Vector.field(3, dtype=ti_dtype, shape=self.max_panels)

        # 4. Strategy pattern for linear solver
        if self.linear_solver_name == "SCIPY":
            self.solver_strategy = PanelScipySolver()
        else:
            self.solver_strategy = PanelBiCGSTABSolver(self.max_panels, ti_dtype)

        print(
            f"   [Panel Solver] Initialized (max_panels={self.max_panels}, dtype={self.float_dtype}, solver={self.linear_solver_name})"
        )

    def add_surface(
        self,
        uid: str,
        stl_path: str,
        motion: Any = None,
        group_id: int = 0,
    ) -> None:
        self._ensure_initialized()
        add_body_from_mesh_stl(self.lattice, uid, stl_path, motion=motion, group_id=group_id)

    def load_scene(self, layout_file: str):
        """
        Load a complete scene (assembly of bodies) from a JSON layout file.

        The layout file should define a list of bodies with their properties::

            {
                "bodies": [
                    {
                        "uid": "wing",
                        "stl_path": "path/to/wing.stl",
                        "translation": [x, y, z],
                        "rotation_deg": [rx, ry, rz],
                        "rotation_center": [cx, cy, cz],
                        "group_id": 0,
                        "kinematics": {
                            "type": "TranslatingPanel",
                            "velocity": [10.0, 0.0, 0.0]
                        }
                    }
                ]
            }

        Kinematics type strings match the class names in ``panels.coupling.kinematics``:
        ``StaticPanel``, ``TranslatingPanel``, ``RotatingPanel``, ``PitchingPanel``,
        ``HeavingPanel``, ``ManeuverPanel``, ``CompositePanel``, ``Static``, ``Plunging``,
        ``RampedRotation``.
        """
        self._ensure_initialized()
        with open(layout_file) as f:
            data = json.load(f)

        base_dir = os.path.dirname(layout_file)

        for body_data in data.get("bodies", []):
            uid = body_data.get("uid", "body")
            stl_file = body_data.get("stl_path", body_data.get("file", ""))
            if not os.path.isabs(stl_file):
                stl_file = os.path.join(base_dir, stl_file)

            group_id = body_data.get("group_id", 0)

            kin_data = body_data.get("kinematics", {"type": "StaticPanel"})
            kin_type = kin_data.get("type")
            try:
                kin_cls = getattr(kin_module, kin_type)
            except AttributeError:
                print(f"Warning: Unknown kinematics type '{kin_type}', defaulting to StaticPanel")
                kin_cls = kin_module.StaticPanel

            kin_kwargs = {k: v for k, v in kin_data.items() if k != "type"}
            for k, v in kin_kwargs.items():
                if isinstance(v, list):
                    kin_kwargs[k] = np.array(v)

            motion = kin_cls(**kin_kwargs) if kin_kwargs else kin_cls()

            self.add_surface(uid=uid, stl_path=stl_file, motion=motion, group_id=group_id)

    def save_results(self, filename: str, flow_time: float = 0.0) -> None:
        """
        Save panel results to VTK file.

        Args:
            filename: Output filename (without extension)
            flow_time: Simulation time for Paraview TimeValue
        """
        self._ensure_initialized()
        n = self.lattice.num_panels
        if n == 0:
            return

        vertices = self.lattice.vertices.to_numpy()[:n]
        centers = self.lattice.centers.to_numpy()[:n]
        normals = self.lattice.normals.to_numpy()[:n]
        strengths = self.lattice.strengths.to_numpy()[:n]
        areas = self.lattice.areas.to_numpy()[:n]
        cp = self.lattice.Cp.to_numpy()[:n]
        lesp = self.lattice.lesp.to_numpy()[:n]
        is_te = self.lattice.is_TE_panel.to_numpy()[:n]
        is_le = self.lattice.is_LE_panel.to_numpy()[:n]
        group_ids = self.lattice.panel_group_id.to_numpy()[:n]
        panel_forces = (
            self.panel_forces.to_numpy()[:n] if self.panel_forces is not None else np.zeros((n, 3))
        )

        panel_mesh_to_vtp(
            vertices=vertices,
            centers=centers,
            normals=normals,
            strengths=strengths,
            areas=areas,
            cp=cp,
            panel_forces=panel_forces,
            lesp=lesp,
            is_te=is_te,
            is_le=is_le,
            group_ids=group_ids,
            flow_time=flow_time,
            filepath=f"{filename}.vtp",
        )

    def initialize(self, force: bool = False) -> None:
        self._ensure_initialized()
        # Fallback to ensuring mesh is generated if we have bodies but no panels (if applicable)
        if force or not self.is_initialized:
            n = self.lattice.num_panels
            if n > 0:
                if self.bc_type == "DIRICHLET":
                    build_AIC_matrix_dirichlet(
                        self.lattice.vertices,
                        self.lattice.centers,
                        self.lattice.normals,
                        self.AIC,
                        n,
                    )
                else:
                    build_AIC_matrix(
                        self.lattice.vertices,
                        self.lattice.centers,
                        self.lattice.normals,
                        self.AIC,
                        n,
                    )
                self.is_initialized = True

    def _resolve_wake_field(self) -> ti.Vector.field:
        n = self.lattice.num_panels
        if hasattr(self, "_zero_wake") and self._zero_wake.shape[0] == n:
            return self._zero_wake
        self._zero_wake = ti.Vector.field(3, dtype=self.lattice.ti_dtype, shape=n)
        self._zero_wake.fill(0.0)
        return self._zero_wake

    def solve(self, V_inf: np.ndarray, V_wake_field: Any, time: float) -> None:
        self._ensure_initialized()
        n = self.lattice.num_panels
        self.initialize()

        if V_wake_field is None:
            V_wake_field = self._resolve_wake_field()

        ti_v_inf = ti.Vector(V_inf.tolist())

        if self.bc_type == "DIRICHLET":
            compute_RHS_dirichlet_with_sources(
                self.lattice.vertices,
                self.lattice.centers,
                self.lattice.normals,
                ti_v_inf,
                V_wake_field,
                self.rhs,
                n,
            )
        else:
            # Neumann BC with source-doublet formulation (Hess-Smith)
            compute_RHS_neumann_with_sources(
                self.lattice.vertices,
                self.lattice.centers,
                self.lattice.normals,
                ti_v_inf,
                V_wake_field,
                self.rhs,
                n,
            )

        success = self.solver_strategy.solve(self.AIC, self.rhs, self.lattice.strengths, n)
        if not success:
            logger.error("Panel linear solver failed to converge.")
        self._solved = success
        self.results["diagnostics"].append(
            {
                "step": int(self.step),
                "time": float(time),
                "num_panels": int(n),
                "linear_solver": type(self.solver_strategy).__name__,
                "linear_solver_success": bool(success),
                "force_method": self.force_config.method,
                "bc_type": self.bc_type,
            }
        )

    def ensure_mesh_generated(self) -> None:
        self._ensure_initialized()
        if self.lattice.num_panels <= 0:
            raise RuntimeError(
                "PanelSolver has no panels loaded. Add a surface before solve/advance."
            )

    def apply_translation_update(
        self, displacement: np.ndarray, linear_velocity: np.ndarray, body_range
    ) -> None:
        self._ensure_initialized()
        panel_update_translation_kernel(self.lattice, body_range, displacement, linear_velocity)

    def apply_rotation_update(
        self,
        rotation_matrix: np.ndarray,
        angular_velocity: np.ndarray,
        rotation_center: np.ndarray,
        body_range,
    ) -> None:
        self._ensure_initialized()
        panel_update_rotation_kernel(
            self.lattice, body_range, rotation_matrix, angular_velocity, rotation_center
        )

    def compute_forces(
        self, V_inf: np.ndarray, V_wake_field: Any, dt: float, rho: float
    ) -> dict[int, np.ndarray]:
        """
        Compute integrated force vector per body group using Bernoulli or Impulse.
        """
        n = self.lattice.num_panels
        if n == 0:
            return {}

        v_inf_mag = np.linalg.norm(V_inf)
        ti_v_inf = ti.Vector(V_inf.tolist())

        if V_wake_field is None:
            V_wake_field = self._resolve_wake_field()

        if self.force_config.method == "BERNOULLI":
            compute_surface_velocities(
                self.lattice.vertices,
                self.lattice.centers,
                self.lattice.strengths,
                ti_v_inf,
                V_wake_field,
                self.V_surface,
                n,
            )
            compute_pressure_bernoulli(self.V_surface, float(v_inf_mag), self.lattice.Cp, n)
            compute_forces_bernoulli(
                self.V_surface,
                float(v_inf_mag),
                self.lattice.areas,
                self.lattice.normals,
                rho,
                self.panel_forces,
                n,
            )
        elif self.force_config.method == "KUTTA_JOUKOWSKI":
            compute_surface_velocities(
                self.lattice.vertices,
                self.lattice.centers,
                self.lattice.strengths,
                ti_v_inf,
                V_wake_field,
                self.V_surface,
                n,
            )
            compute_forces_kutta_joukowski(
                self.lattice.strengths,
                self.V_surface,
                self.lattice.vertices,
                self.lattice.areas,
                self.lattice.normals,
                rho,
                self.panel_forces,
                n,
            )
        elif self.force_config.method == "IMPULSE":
            impulse_step = self.step
            if self.force_config.impulse_order == "BDF1":
                impulse_step = min(self.step, 1)
            compute_forces_impulse(
                self.lattice.strengths,
                self.lattice.strengths_old,
                self.lattice.strengths_old2,
                self.lattice.areas,
                self.lattice.normals,
                rho,
                dt,
                impulse_step,
                self.panel_forces,
                n,
            )

        # Summarize forces by group_id
        forces_np = self.panel_forces.to_numpy()[:n]
        total_forces: dict[int, np.ndarray] = {}

        for body in self.lattice.bodies:
            gid = body.group_id
            body_force = np.sum(forces_np[body.start_idx : body.start_idx + body.count], axis=0)
            if gid not in total_forces:
                total_forces[gid] = np.zeros(3)
            total_forces[gid] += body_force

        self.results["forces"].append(total_forces)

        for gid, f in total_forces.items():
            logger.info(f"Step {self.step}: Group {gid} Force = {f}")
        return total_forces

    def compute_loads(
        self, V_inf: np.ndarray, V_wake_field: Any, dt: float, rho: float
    ) -> dict[int, np.ndarray]:
        return self.compute_forces(V_inf, V_wake_field, dt, rho)

    def compute_induced_velocity(self, points: np.ndarray) -> np.ndarray:
        self._ensure_initialized()
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        n_panels = self.lattice.num_panels
        if n_panels == 0:
            return np.zeros_like(points)

        vertices = self.lattice.vertices.to_numpy()[:n_panels]
        strengths = self.lattice.strengths.to_numpy()[:n_panels]
        velocity = np.zeros_like(points, dtype=float)

        if n_panels >= 1000:
            compute_induced_velocity_kernel(vertices, strengths, points, velocity)
        else:

            def _segment_velocity(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
                r1 = p - a
                r2 = p - b
                r1xr2 = np.cross(r1, r2)
                d1 = np.linalg.norm(r1)
                d2 = np.linalg.norm(r2)
                denom = d1 * d2 + np.dot(r1, r2) + 1e-12
                coeff = (
                    (1.0 / (4.0 * np.pi))
                    * (1.0 / (d1 + 1e-12) + 1.0 / (d2 + 1e-12))
                    * (1.0 / denom)
                )
                return coeff * r1xr2

            for j in range(n_panels):
                v0, v1, v2 = vertices[j, 0], vertices[j, 1], vertices[j, 2]
                gamma = strengths[j]
                for i in range(points.shape[0]):
                    p = points[i]
                    velocity[i] += gamma * (
                        _segment_velocity(p, v0, v1)
                        + _segment_velocity(p, v1, v2)
                        + _segment_velocity(p, v2, v0)
                    )

        return velocity

    def advance(
        self,
        V_inf: np.ndarray | None = None,
        V_wake_field: Any = None,
        t: float | None = None,
        dt: float | None = None,
        particles: Any = None,
        physics: Any = None,
        config: Any = None,
        time: float | None = None,
        time_step: int | None = None,
        logging_frequency: int | None = None,
        density: float | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray] | None:
        """
        Advance panel simulation by one time step.

        Supports two calling conventions:

        1. Legacy (positional): advance(V_inf, V_wake_field, t, dt)
        2. VPM-coupled (keyword): advance(particles=..., physics=..., V_inf=..., dt=..., time=..., ...)

        Returns newly shed wake particles to be added to VPM system.
        """
        # Resolve arguments from either calling convention
        if V_inf is None and config is not None:
            V_inf = np.array(getattr(config, "background_velocity", [1.0, 0.0, 0.0]))
        if V_inf is None:
            V_inf = self.U_inf if self.U_inf is not None else np.array([1.0, 0.0, 0.0])
        if dt is None:
            dt = 0.01
        if t is None and time is not None:
            t = time
        if t is None:
            t = self._current_time
        if time_step is not None:
            self.step = time_step
        if density is not None:
            self.density = density
        if logging_frequency is not None:
            self.logging_frequency = logging_frequency

        self.ensure_mesh_generated()

        # 1. Save history for BDF2
        self.lattice.save_old_strengths()

        # 2. Update geometry via kinematics (if any)
        for body in self.lattice.bodies:
            kinematics = getattr(body, "kinematics", None)
            if kinematics is None:
                kinematics = getattr(body, "motion", None)
            if kinematics is not None:
                kinematics.update(self, t, dt, (body.start_idx, body.start_idx + body.count))

        # 3. Compute VPM-induced velocity at collocation points (if coupled)
        V_wake = V_wake_field
        if particles is not None and physics is not None:
            # Use physics to compute induced velocity from particles at panel centers
            physics.compute_target_velocities(
                particles,
                self.lattice.centers,
                self.lattice.velocities,  # Use as temporary storage for external velocity
                include_freestream=False,  # Freestream added separately
            )
            V_wake = self.lattice.velocities

        # 4. Solve potential flow
        self.solve(V_inf, V_wake, t)

        # 5. Compute post-process (forces, Cp)
        self.compute_postprocess(V_inf, V_inf, self.density, dt=dt, coupled=(particles is not None))

        # 6. Compute loads (legacy method, also stores in results)
        self.compute_loads(V_inf, V_wake, dt, self.density)

        # 7. Log forces if requested
        log_freq = self.logging_frequency
        if log_freq > 0 and self.step % log_freq == 0:
            try:
                self.log_forces_table(self.density, V_inf)
            except Exception as e:
                print(f"   (Warning) Could not compute panel forces: {e}")

        # 8. Shed vorticity into decoupled wake buffer (TE/LE)
        if self.shedder is not None:
            self.shedder.shed(self.lattice, V_inf, dt, t, self.step)

        # 9. Optional: General surface separation (APG onset)
        if self.separation is not None:
            self.separation.detect_and_shed(self.lattice, dt)

        # 10. Extract buffered particles for external VPM injection
        new_particles = self._extract_wake_particles()

        self.results["times"].append(float(t))
        self._current_time = float(t)
        self._last_V_inf = V_inf
        self._last_V_wake = V_wake

        self.step += 1
        return new_particles

    def advance_time(self, dt: float, current_time: float) -> None:
        """Advance kinematics state and geometry for all surfaces (VLM-compatible API)."""
        self.ensure_mesh_generated()
        self._current_time = float(current_time)

        V_inf = self._last_V_inf if hasattr(self, "_last_V_inf") else np.array([1.0, 0.0, 0.0])
        V_wake = self._last_V_wake if hasattr(self, "_last_V_wake") else None

        self.lattice.save_old_strengths()
        for body in self.lattice.bodies:
            kinematics = getattr(body, "kinematics", None) or getattr(body, "motion", None)
            if kinematics is not None:
                kinematics.update(
                    self, current_time, dt, (body.start_idx, body.start_idx + body.count)
                )
        self.solve(V_inf, V_wake, current_time)

    def reset_wake_buffer(self) -> None:
        """Reset wake panel buffer for new time step."""
        if self.lattice is not None:
            self.lattice.num_wake_panels_to_shed[None] = 0

    def _extract_wake_particles(self) -> dict[str, np.ndarray] | None:
        """Extracts newly shed particles from GPU buffer to NumPy for the VPM coupler."""
        n_shed = self.lattice.num_wake_panels_to_shed[None]
        if n_shed == 0:
            return None

        return {
            "points": self.lattice.wake_positions.to_numpy()[:n_shed],
            "strengths": self.lattice.wake_strengths.to_numpy()[:n_shed],
            "radii": self.lattice.wake_radii.to_numpy()[:n_shed],
            "volumes": self.lattice.wake_volumes.to_numpy()[:n_shed],
        }

    def compute_induced_velocity_direct(self, particles) -> None:
        """
        Compute velocity induced by panels on VPM particles and add to particles.velocity.

        This method is called by the core VPM solver to couple the panel method
        with the particle method. It extracts particle positions, computes the
        induced velocity using the existing compute_induced_velocity method,
        and adds the result to particles.velocity.

        Args:
            particles: VPM particles object with position and velocity fields
        """
        if self.lattice is None or self.lattice.num_panels == 0:
            return

        n_particles = particles.number_of_particles
        if n_particles == 0:
            return

        np_dtype = getattr(particles, "_np_float_dtype", np.float32)

        # Synchronize any pending Taichi kernel writes before reading
        # particle data (velocity_self may have just written to these fields).
        ti.sync()

        # Extract particle positions in the container's native dtype
        positions = particles.position.to_numpy()[:n_particles].astype(np_dtype)

        # Compute induced velocity at particle positions
        v_induced = self.compute_induced_velocity(positions)

        # Add to particle velocities.
        # v_induced is f64 from numpy operations; v_current is f32.
        # We cast the sum to f32 for the taichi field.
        v_current = particles.velocity.to_numpy()[:n_particles]
        v_new = (v_current + v_induced).astype(np_dtype)

        # Write back using the particle container's copy kernel which
        # handles dtype conversion safely (avoids from_numpy issues on
        # fields previously written by Taichi kernels).
        particles._copy_to_taichi_vectors(v_new, particles.velocity, 0, n_particles)

    def absorb_particles(self, particles, tolerance: float = 0.05) -> int:
        """
        Detect and remove particles colliding with the panel surface.

        For thick bodies (sphere, cube, etc.), particles that pass through
        the surface can cause numerical issues. This method detects particles
        whose perpendicular distance to any panel is less than tolerance and
        whose projection lies inside the panel triangle, and removes them.

        Args:
            particles: VPM particles object
            tolerance: Collision distance threshold [m]

        Returns:
            Number of particles removed
        """
        if self.lattice is None or self.lattice.num_panels == 0:
            return 0

        n_particles = particles.number_of_particles
        n_panels = self.lattice.num_panels

        if n_particles == 0 or n_panels == 0:
            return 0

        # Extract data
        positions = particles.position.to_numpy()[:n_particles]
        centers = self.lattice.centers.to_numpy()[:n_panels]
        normals = self.lattice.normals.to_numpy()[:n_panels]
        vertices = self.lattice.vertices.to_numpy()[:n_panels]

        # Simple collision detection: check if particle is inside the body
        # by checking if it's on the negative side of all panel normals
        # (assuming outward-pointing normals for a closed body)
        keep_mask = np.ones(n_particles, dtype=bool)

        for i in range(n_particles):
            p = positions[i]
            # Check distance to each panel center
            for j in range(n_panels):
                # Vector from panel center to particle
                r = p - centers[j]
                # Perpendicular distance to panel plane
                d_perp = abs(np.dot(r, normals[j]))
                if d_perp < tolerance:
                    # Check if projection is inside triangle (simplified check)
                    # Project onto panel plane
                    p_proj = p - d_perp * normals[j] * np.sign(np.dot(r, normals[j]))
                    # Check if inside triangle using barycentric coordinates
                    v0 = vertices[j, 1] - vertices[j, 0]
                    v1 = vertices[j, 2] - vertices[j, 0]
                    v2 = p_proj - vertices[j, 0]
                    d00 = np.dot(v0, v0)
                    d01 = np.dot(v0, v1)
                    d11 = np.dot(v1, v1)
                    d20 = np.dot(v2, v0)
                    d21 = np.dot(v2, v1)
                    denom = d00 * d11 - d01 * d01
                    if abs(denom) > 1e-12:
                        v = (d11 * d20 - d01 * d21) / denom
                        w = (d00 * d21 - d01 * d20) / denom
                        u = 1.0 - v - w
                        if u >= -0.1 and v >= -0.1 and w >= -0.1:
                            keep_mask[i] = False
                            break

        n_removed = int(np.sum(~keep_mask))
        if n_removed == 0:
            return 0

        # Compact particles
        n_keep = int(np.sum(keep_mask))
        if n_keep == 0:
            particles.number_of_particles = 0
            return n_removed

        # Extract kept particles
        new_position = positions[keep_mask]
        new_velocity = particles.velocity.to_numpy()[:n_particles][keep_mask]
        new_circulation = particles.circulation.to_numpy()[:n_particles][keep_mask]
        new_vorticity = particles.vorticity.to_numpy()[:n_particles][keep_mask]
        new_radius = particles.radius.to_numpy()[:n_particles][keep_mask]
        new_volume = particles.volume.to_numpy()[:n_particles][keep_mask]
        new_viscosity = particles.viscosity.to_numpy()[:n_particles][keep_mask]
        new_viscosity_turbulent = particles.viscosity_turbulent.to_numpy()[:n_particles][keep_mask]
        new_viscosity_effective = particles.viscosity_effective.to_numpy()[:n_particles][keep_mask]
        new_group_id = particles.group_id.to_numpy()[:n_particles][keep_mask]
        new_velocity_gradient = particles.velocity_gradient.to_numpy()[:n_particles][keep_mask]
        new_strain_rate = particles.strain_rate.to_numpy()[:n_particles][keep_mask]

        # Write back to particles
        particles._copy_to_taichi_vectors(new_position, particles.position, 0, n_keep)
        particles._copy_to_taichi_vectors(new_velocity, particles.velocity, 0, n_keep)
        particles._copy_to_taichi_vectors(new_circulation, particles.circulation, 0, n_keep)
        particles._copy_to_taichi_vectors(new_vorticity, particles.vorticity, 0, n_keep)
        particles._copy_to_taichi_scalars(new_radius, particles.radius, 0, n_keep)
        particles._copy_to_taichi_scalars(new_volume, particles.volume, 0, n_keep)
        particles._copy_to_taichi_scalars(new_viscosity, particles.viscosity, 0, n_keep)
        particles._copy_to_taichi_scalars(
            new_viscosity_turbulent, particles.viscosity_turbulent, 0, n_keep
        )
        particles._copy_to_taichi_scalars(
            new_viscosity_effective, particles.viscosity_effective, 0, n_keep
        )
        particles._copy_to_taichi_scalars(new_group_id, particles.group_id, 0, n_keep)
        particles._copy_to_taichi_matrices(
            new_velocity_gradient, particles.velocity_gradient, 0, n_keep
        )
        particles._copy_to_taichi_matrices(new_strain_rate, particles.strain_rate, 0, n_keep)

        particles.number_of_particles = n_keep
        particles.sync_device_counter()

        print(f"   (Panel) Absorbed {n_removed} particles impinging on surface.")
        return n_removed

    def compute_postprocess(
        self,
        V_external_np: np.ndarray,
        U_ref: np.ndarray,
        density: float,
        dt: float | None = None,
        coupled: bool = False,
    ) -> None:
        """
        Compute derived quantities (velocities, pressures, forces) after solve.

        Args:
            V_external_np: External velocity (N, 3) or (3,)
            U_ref: Reference velocity vector [ux, uy, uz] (m/s)
            density: Fluid density
            dt: Time step size (required for impulse method)
            coupled: Whether in coupled mode (unused for panel method)
        """
        if self.lattice is None or self.lattice.num_panels == 0:
            return

        n = self.lattice.num_panels
        U_ref_mag = np.linalg.norm(U_ref)
        if U_ref_mag < 1e-10:
            U_ref_mag = 1.0

        # Resolve V_inf
        if V_external_np.ndim == 1 and V_external_np.shape[0] == 3:
            V_inf = V_external_np
        else:
            V_inf = np.mean(V_external_np, axis=0)

        ti_v_inf = ti.Vector(V_inf.tolist())
        V_wake = self._resolve_wake_field()

        # Compute source strengths and upload to lattice
        # For Dirichlet BC: σ_j = -V_∞ · n_j
        normals_np = self.lattice.normals.to_numpy()[:n]
        source_strengths_np = -np.dot(normals_np, V_inf)
        # Cast to lattice dtype to avoid f32<-f64 precision warnings
        ti_dtype = np.float32 if self.float_dtype == "f32" else np.float64
        source_full = np.zeros(self.lattice.max_panels, dtype=ti_dtype)
        source_full[:n] = source_strengths_np.astype(ti_dtype)
        self.lattice.source_strengths.from_numpy(source_full)

        # Compute surface velocities
        # Use source-doublet formulation for both DIRICHLET and NEUMANN BC
        compute_surface_velocities_with_sources(
            self.lattice.vertices,
            self.lattice.centers,
            self.lattice.normals,
            self.lattice.strengths,
            self.lattice.source_strengths,
            ti_v_inf,
            V_wake,
            self.V_surface,
            n,
        )

        # Compute pressure coefficients
        compute_pressure_bernoulli(self.V_surface, float(U_ref_mag), self.lattice.Cp, n)

        # Compute forces
        compute_forces_bernoulli(
            self.V_surface,
            float(U_ref_mag),
            self.lattice.areas,
            self.lattice.normals,
            density,
            self.panel_forces,
            n,
        )

    def compute_forces_coefficients(
        self,
        density: float,
        U_ref: np.ndarray | None = None,
        S_ref: float | None = None,
        c_ref: float | None = None,
        b_ref: float | None = None,
    ) -> dict[str, float]:
        """
        Compute integrated aerodynamic forces and moments as coefficients.

        Returns a dictionary with keys: CL, CD, CC, Fx, Fy, Fz, Mx, My, Mz,
        Cl, Cm, Cn, q, S_ref, etc.

        Args:
            density: Fluid density (kg/m³)
            U_ref: Reference velocity vector [ux, uy, uz] (for coefficients and L/D axes).
                   If None, uses self.U_inf or auto-computed.
            S_ref: Reference area (m²). If None, uses sum of panel areas.
            c_ref: Reference chord (m). If None, uses sqrt(S_ref).
            b_ref: Reference span (m). If None, uses sqrt(S_ref).

        Returns:
            Dictionary with force components and coefficients
        """
        if self.lattice is None or self.lattice.num_panels == 0:
            return {}

        if not self._solved:
            raise RuntimeError("Must solve system before computing forces")

        # Resolve reference values
        if U_ref is None:
            U_ref = self.U_inf if self.U_inf is not None else np.array([1.0, 0.0, 0.0])
        U_ref_mag = np.linalg.norm(U_ref)
        if U_ref_mag < 1e-10:
            U_ref_mag = 1.0

        n = self.lattice.num_panels
        forces_np = self.panel_forces.to_numpy()[:n]
        F_total = np.sum(forces_np, axis=0)
        Fx, Fy, Fz = F_total

        # Decompose into lift, drag, side-force
        V_hat = U_ref / U_ref_mag
        z_hat = np.array([0.0, 0.0, 1.0])
        L_hat = z_hat - np.dot(z_hat, V_hat) * V_hat
        L_hat_norm = np.linalg.norm(L_hat)
        L_hat = L_hat / L_hat_norm if L_hat_norm > 1e-10 else z_hat
        C_hat = np.cross(V_hat, L_hat)

        L = float(np.dot(F_total, L_hat))
        D = float(np.dot(F_total, V_hat))
        C = float(np.dot(F_total, C_hat))

        # Compute moments about reference center
        centers_np = self.lattice.centers.to_numpy()[:n]
        r_ref = np.mean(centers_np, axis=0)
        M_total = np.sum(np.cross(centers_np - r_ref, forces_np), axis=0)
        Mx, My, Mz = M_total

        # Reference values
        if S_ref is None:
            areas_np = self.lattice.areas.to_numpy()[:n]
            S_ref = float(np.sum(areas_np))
        if c_ref is None:
            c_ref = float(np.sqrt(S_ref))
        if b_ref is None:
            b_ref = float(np.sqrt(S_ref))

        # Non-dimensionalize
        q = 0.5 * density * U_ref_mag**2
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
            Cl = Mx / norm_l
            Cm = My / norm_m
            Cn = Mz / norm_l
        else:
            CL = CD = CC = CFx = CFy = CFz = 0.0
            Cl = Cm = Cn = 0.0

        result = {
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
            "q": q,
            "S_ref": S_ref,
            "c_ref": c_ref,
            "b_ref": b_ref,
            "r_ref": r_ref,
        }

        self._last_forces = result
        self._last_U_ref = U_ref
        return result

    def log_forces_table(self, density: float, U_ref: np.ndarray | None = None) -> dict[str, float]:
        """
        Log panel forces in a formatted table matching VLM diagnostics style.

        Prints per-surface forces and total forces with descriptions.

        Args:
            density: Fluid density (kg/m^3)
            U_ref: Reference velocity vector

        Returns:
            Dictionary of force coefficients
        """
        print("\n" + "-" * 60)
        print("PANEL AERODYNAMIC FORCES")
        print("-" * 60)
        method_name = self.force_config.method
        print(f"  Force computation method: {method_name}")
        print("    L = Lift force (perpendicular to freestream)")
        print("    D = Drag force (parallel to freestream)")
        print("    CL, CD = Lift and drag coefficients")
        print()

        # Get total forces
        total_forces = self.compute_forces_coefficients(density, U_ref)

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

        print()
        print("  Moment Coefficients:")
        print(f"    Roll (Cl)    : {Cl:>12.3f}")
        print(f"    Pitch (Cm)   : {Cm:>12.3f}")
        print(f"    Yaw (Cn)     : {Cn:>12.3f}")

        if "r_ref" in total_forces:
            r = total_forces["r_ref"]
            print(f"    Ref Center   : [{r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}]")

        print("-" * 60, flush=True)

        return total_forces

    def compute_per_surface_forces(
        self,
        density: float,
        U_ref: np.ndarray | None = None,
        S_ref: float | None = None,
        c_ref: float | None = None,
        b_ref: float | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute forces for each individual surface (body group).

        Args:
            density: Fluid density (kg/m³)
            U_ref: Reference velocity vector [ux, uy, uz]
            S_ref: Reference area (m²). If None, uses sum of panel areas.
            c_ref: Reference chord (m). If None, uses sqrt(S_ref).
            b_ref: Reference span (m). If None, uses sqrt(S_ref).

        Returns:
            Dictionary mapping surface name to force dictionary
        """
        if self.lattice is None or self.lattice.num_panels == 0:
            return {}

        if not self._solved:
            return {}

        # Resolve reference values
        if U_ref is None:
            U_ref = self.U_inf if self.U_inf is not None else np.array([1.0, 0.0, 0.0])
        U_ref_mag = np.linalg.norm(U_ref)
        if U_ref_mag < 1e-10:
            U_ref_mag = 1.0

        n = self.lattice.num_panels
        forces_np = self.panel_forces.to_numpy()[:n]
        areas_np = self.lattice.areas.to_numpy()[:n]

        if S_ref is None:
            S_ref = float(np.sum(areas_np))
        if c_ref is None:
            c_ref = float(np.sqrt(S_ref))
        if b_ref is None:
            b_ref = float(np.sqrt(S_ref))

        q = 0.5 * density * U_ref_mag**2
        norm_f = q * S_ref
        V_hat = U_ref / U_ref_mag if U_ref_mag > 1e-10 else np.array([1.0, 0.0, 0.0])

        result = {}
        for body in self.lattice.bodies:
            uid = body.uid
            body_forces = forces_np[body.start_idx : body.start_idx + body.count]
            F_surface = np.sum(body_forces, axis=0)
            Fx, Fy, Fz = F_surface
            D = np.dot(F_surface, V_hat)
            L_vec = F_surface - D * V_hat
            L = np.linalg.norm(L_vec)
            if np.dot(L_vec, np.array([0, 0, 1])) < 0:
                L = -L
            CL = L / norm_f if norm_f > 1e-10 else 0.0
            CD = D / norm_f if norm_f > 1e-10 else 0.0
            result[uid] = {
                "L": L,
                "D": D,
                "Fx": Fx,
                "Fy": Fy,
                "Fz": Fz,
                "CL": CL,
                "CD": CD,
                "panel_count": body.count,
            }

        return result
