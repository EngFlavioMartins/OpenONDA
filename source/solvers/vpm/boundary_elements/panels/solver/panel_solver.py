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
from ..kernels.induced_velocity import (
    compute_induced_velocity_kernel,
    compute_source_induced_velocity_kernel,
)
from .influence import (
    build_dirichlet_aerodynamic_influence_coefficient_matrix,
    build_source_aerodynamic_influence_coefficient_matrix,
    compute_dirichlet_right_hand_side_with_sources,
    compute_forces_bernoulli,
    compute_forces_kutta_joukowski,
    compute_pressure_bernoulli,
    compute_right_hand_side,
    compute_surface_velocity_with_sources,
)
from .kernels import panel_update_rotation_kernel, panel_update_translation_kernel
from .lattice import PanelLattice
from .linear_solvers import PanelBiCGSTABSolver, PanelScipySolver
from .mesh import add_body_from_mesh_stl
from .vtk_export import panel_mesh_to_vtp

logger = logging.getLogger("vpm")


@dataclass
class ForceConfig:
    """Configuration for aerodynamic force evaluation on panel methods.

    Supports two methodologies with different trade-offs in accuracy,
    unsteadiness handling, and wake-truncation sensitivity.

    **1. Bernoulli (default):**
       Integrates surface pressure from the unsteady Bernoulli equation:

           pressure_coefficient = 1 - (surface_velocity / freestream_velocity)² - (2 / freestream_velocity²) ∂φ/∂t

       Pros:
       - Accurate for inviscid, irrotational flow
       - Handles unsteady added-mass naturally
       - Well-established in panel-method literature

       Cons:
       - Requires velocity-potential time history
       - Sensitive to panel discretisation quality

    **2. Kutta-Joukowski:**
       Applies K-J theorem per panel:  F = ρ Γ × V_local

       Pros:
       - Direct evaluation, no pressure integration
       - Fast and robust for thin lifting surfaces

       Cons:
       - Less accurate for thick bodies
       - Misses unsteady added-mass effects

    Examples
    --------
    .. code-block:: python

        # Bernoulli (default)
        force = ForceConfig.bernoulli()

        # Kutta-Joukowski
        force = ForceConfig.kutta_joukowski()
    """

    method: Literal["BERNOULLI", "KUTTA_JOUKOWSKI"] = "BERNOULLI"

    @classmethod
    def bernoulli(cls):
        return cls(method="BERNOULLI")

    @classmethod
    def kutta_joukowski(cls):
        return cls(method="KUTTA_JOUKOWSKI")


class PanelSolver:
    def __init__(
        self,
        max_n_panels: int = 10000,
        float_dtype: str = "f32",
        linear_solver: Literal["SCIPY", "BICGSTAB_GPU"] = "SCIPY",
        force_config: ForceConfig | None = None,
        boundary_condition_type: Literal["DIRICHLET", "NEUMANN"] = "DIRICHLET",
        density: float = 1.225,
        freestream_velocity: np.ndarray | None = None,
        logging_interval_steps: int = 1,
        coupling_scope: Literal["full", "vpm_boundary_condition", "normal", "pressure"] = "full",
    ):
        self.max_n_panels = max_n_panels
        self.float_dtype = float_dtype
        self.linear_solver_name = linear_solver
        self.force_config = force_config or ForceConfig.bernoulli()
        self.boundary_condition_type = boundary_condition_type
        self.density = density
        self.freestream_velocity = (
            None if freestream_velocity is None else np.array(freestream_velocity, dtype=np.float64)
        )
        self.logging_interval_steps = max(1, int(logging_interval_steps))
        if coupling_scope not in ("full", "vpm_boundary_condition", "normal", "pressure"):
            raise ValueError(
                "coupling_scope must be 'full', 'vpm_boundary_condition', 'normal', or 'pressure'"
            )
        self.coupling_scope = coupling_scope
        self.step = 0
        self._current_time = 0.0
        self._solved = False
        self._last_forces: dict[str, float] = {}
        self._last_reference_velocity: np.ndarray | None = None

        # Lazy initialization state
        self.lattice: PanelLattice | None = None
        self.solver_strategy = None

        # Fields (initialized lazily)
        self.aerodynamic_influence_coefficient = None
        self.right_hand_side = None
        self.panel_force = None
        self.surface_velocity = None

        self.results = {
            "force_history": [],
            "moment_history": [],
            "time_history": [],
            "diagnostic_history": [],
        }
        self.is_initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of GPU fields and sub-solvers after Taichi init."""
        if self.lattice is not None:
            return

        # 1. Create Lattice
        self.lattice = PanelLattice(self.max_n_panels, self.float_dtype)
        ti_dtype = self.lattice.ti_dtype

        # 2. Create GPU fields
        self.aerodynamic_influence_coefficient = ti.field(
            ti_dtype, shape=(self.max_n_panels, self.max_n_panels)
        )
        self.right_hand_side = ti.field(ti_dtype, shape=self.max_n_panels)
        self.panel_force = ti.Vector.field(3, dtype=ti_dtype, shape=self.max_n_panels)
        self.surface_velocity = ti.Vector.field(3, dtype=ti_dtype, shape=self.max_n_panels)

        # 3. Strategy pattern for linear solver
        if self.linear_solver_name == "SCIPY":
            self.solver_strategy = PanelScipySolver()
        else:
            self.solver_strategy = PanelBiCGSTABSolver(self.max_n_panels, ti_dtype)

        print(
            f"   [Panel Solver] Initialized (max_n_panels={self.max_n_panels}, dtype={self.float_dtype}, solver={self.linear_solver_name})"
        )

    def add_surface(
        self,
        uid: str,
        stl_path: str,
        kinematics: Any = None,
        group_id: int = 0,
    ) -> None:
        self._ensure_initialized()
        add_body_from_mesh_stl(
            self.lattice, uid, stl_path, kinematics=kinematics, group_id=group_id
        )

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
                        "rotation_degrees": [rx, ry, rz],
                        "rotation_centre": [cx, cy, cz],
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
        ``HeavingPanel``, ``ManeuverPanel``, ``CompositePanel``, and
        ``RampedRotatingPanel``.
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

            kinematics = kin_cls(**kin_kwargs) if kin_kwargs else kin_cls()

            self.add_surface(uid=uid, stl_path=stl_file, kinematics=kinematics, group_id=group_id)

    def save_results(self, filename: str, time: float = 0.0) -> None:
        """
        Save panel results to VTK file.

        Args:
            filename: Output filename (without extension)
            time: Physical simulation time stored in the output field.
        """
        self._ensure_initialized()
        n = self.lattice.n_panels
        if n == 0:
            return

        vertex_position = self.lattice.vertex_position.to_numpy()[:n]
        panel_centre = self.lattice.panel_centre.to_numpy()[:n]
        normal = self.lattice.normal.to_numpy()[:n]
        doublet_strength = self.lattice.doublet_strength.to_numpy()[:n]
        area = self.lattice.area.to_numpy()[:n]
        pressure_coefficient = self.lattice.pressure_coefficient.to_numpy()[:n]
        group_id = self.lattice.group_id.to_numpy()[:n]
        panel_force = (
            self.panel_force.to_numpy()[:n] if self.panel_force is not None else np.zeros((n, 3))
        )

        panel_mesh_to_vtp(
            vertex_position=vertex_position,
            panel_centre=panel_centre,
            normal=normal,
            doublet_strength=doublet_strength,
            area=area,
            pressure_coefficient=pressure_coefficient,
            panel_force=panel_force,
            group_id=group_id,
            time=time,
            filepath=f"{filename}.vtp",
        )

    def initialize(self, force: bool = False) -> None:
        self._ensure_initialized()
        # Fallback to ensuring mesh is generated if we have bodies but no panels (if applicable)
        if force or not self.is_initialized:
            n = self.lattice.n_panels
            if n > 0:
                if self.boundary_condition_type == "DIRICHLET":
                    build_dirichlet_aerodynamic_influence_coefficient_matrix(
                        self.lattice.vertex_position,
                        self.lattice.panel_centre,
                        self.lattice.normal,
                        self.aerodynamic_influence_coefficient,
                        n,
                    )
                else:
                    build_source_aerodynamic_influence_coefficient_matrix(
                        self.lattice.vertex_position,
                        self.lattice.panel_centre,
                        self.lattice.normal,
                        self.aerodynamic_influence_coefficient,
                        n,
                    )
                self.is_initialized = True

    def _resolve_wake_field(self) -> ti.Vector.field:
        n = self.lattice.n_panels
        if hasattr(self, "_zero_wake") and self._zero_wake.shape[0] == n:
            return self._zero_wake
        self._zero_wake = ti.Vector.field(3, dtype=self.lattice.ti_dtype, shape=n)
        self._zero_wake.fill(0.0)
        return self._zero_wake

    def solve(self, freestream_velocity: np.ndarray, wake_velocity: Any, time: float) -> None:
        self._ensure_initialized()
        n = self.lattice.n_panels
        self.initialize()

        if wake_velocity is None:
            wake_velocity = self._resolve_wake_field()

        ti_v_inf = ti.Vector(freestream_velocity.tolist())

        if self.boundary_condition_type == "DIRICHLET":
            compute_dirichlet_right_hand_side_with_sources(
                self.lattice.vertex_position,
                self.lattice.panel_centre,
                self.lattice.normal,
                ti_v_inf,
                wake_velocity,
                self.right_hand_side,
                n,
            )
        else:
            compute_right_hand_side(
                self.lattice.panel_centre,
                self.lattice.normal,
                ti_v_inf,
                wake_velocity,
                self.right_hand_side,
                n,
            )

        doublet_strength = (
            self.lattice.doublet_strength
            if self.boundary_condition_type == "DIRICHLET"
            else self.lattice.source_strength
        )
        success = self.solver_strategy.solve(
            self.aerodynamic_influence_coefficient, self.right_hand_side, doublet_strength, n
        )
        if not success:
            logger.error("Panel linear solver failed to converge.")
        elif self.boundary_condition_type == "NEUMANN":
            values = doublet_strength.to_numpy()
            area = self.lattice.area.to_numpy()
            values[:n] -= np.dot(values[:n], area[:n]) / np.sum(area[:n])
            doublet_strength.from_numpy(values)
        self._solved = success
        self.results["diagnostic_history"].append(
            {
                "step": int(self.step),
                "time": float(time),
                "n_panels": int(n),
                "linear_solver": type(self.solver_strategy).__name__,
                "linear_solver_success": bool(success),
                "force_method": self.force_config.method,
                "boundary_condition_type": self.boundary_condition_type,
            }
        )

    def ensure_mesh_generated(self) -> None:
        self._ensure_initialized()
        if self.lattice.n_panels <= 0:
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
        rotation_centre: np.ndarray,
        body_range,
    ) -> None:
        self._ensure_initialized()
        panel_update_rotation_kernel(
            self.lattice, body_range, rotation_matrix, angular_velocity, rotation_centre
        )

    def _update_surface_velocity(
        self, freestream_velocity: np.ndarray, wake_velocity: Any = None
    ) -> None:
        """Fill ``self.surface_velocity`` from the source-doublet representation.

        Single evaluation point for both force and post-processing paths, so the
        two cannot disagree.  Under DIRICHLET the source doublet_strength are the known
        ``-n·freestream_velocity`` that cancels the freestream normal component; under NEUMANN
        they are the solved unknowns and the doublet field is unused.
        """
        n = self.lattice.n_panels
        if n == 0:
            return
        if wake_velocity is None:
            wake_velocity = self._resolve_wake_field()

        if self.boundary_condition_type == "DIRICHLET":
            normals_np = self.lattice.normal.to_numpy()[:n]
            ti_dtype = np.float32 if self.float_dtype == "f32" else np.float64
            source_full = np.zeros(self.lattice.max_n_panels, dtype=ti_dtype)
            source_full[:n] = (-np.dot(normals_np, freestream_velocity)).astype(ti_dtype)
            self.lattice.source_strength.from_numpy(source_full)

        compute_surface_velocity_with_sources(
            self.lattice.vertex_position,
            self.lattice.panel_centre,
            self.lattice.normal,
            self.lattice.doublet_strength,
            self.lattice.source_strength,
            ti.Vector(np.asarray(freestream_velocity, dtype=float).tolist()),
            wake_velocity,
            self.surface_velocity,
            n,
        )

    def compute_forces(
        self,
        freestream_velocity: np.ndarray,
        wake_velocity: Any,
        time_step_size: float,
        density: float,
    ) -> dict[int, np.ndarray]:
        """
        Compute integrated force vector per body group using Bernoulli or Impulse.
        """
        n = self.lattice.n_panels
        if n == 0:
            return {}

        v_inf_mag = np.linalg.norm(freestream_velocity)
        if v_inf_mag < 1e-10:
            v_inf_mag = 1.0

        if wake_velocity is None:
            wake_velocity = self._resolve_wake_field()

        # Surface velocity comes from the shared source-doublet evaluation.  The
        # doublet-only kernel used here previously omitted the source panels
        # entirely, which under NEUMANN — where the solve fills source_strength
        # and leaves doublet_strength at zero — dropped the whole body contribution.
        self._update_surface_velocity(freestream_velocity, wake_velocity)

        if self.force_config.method == "BERNOULLI":
            compute_pressure_bernoulli(
                self.surface_velocity, float(v_inf_mag), self.lattice.pressure_coefficient, n
            )
            compute_forces_bernoulli(
                self.surface_velocity,
                float(v_inf_mag),
                self.lattice.area,
                self.lattice.normal,
                density,
                self.panel_force,
                n,
            )
        elif self.force_config.method == "KUTTA_JOUKOWSKI":
            compute_forces_kutta_joukowski(
                self.lattice.doublet_strength,
                self.surface_velocity,
                self.lattice.vertex_position,
                self.lattice.area,
                self.lattice.normal,
                density,
                self.panel_force,
                n,
            )
        else:
            raise ValueError(f"Unknown panel force method: {self.force_config.method}")

        # Summarize forces by group_id
        panel_force = self.panel_force.to_numpy()[:n]
        total_forces: dict[int, np.ndarray] = {}

        for body in self.lattice.bodies:
            gid = body.group_id
            body_force = np.sum(panel_force[body.start_idx : body.start_idx + body.count], axis=0)
            if gid not in total_forces:
                total_forces[gid] = np.zeros(3)
            total_forces[gid] += body_force

        self.results["force_history"].append(total_forces)

        for gid, f in total_forces.items():
            logger.info(f"Step {self.step}: Group {gid} Force = {f}")
        return total_forces

    def compute_loads(
        self,
        freestream_velocity: np.ndarray,
        wake_velocity: Any,
        time_step_size: float,
        density: float,
    ) -> dict[int, np.ndarray]:
        return self.compute_forces(freestream_velocity, wake_velocity, time_step_size, density)

    def compute_induced_velocity(self, points: np.ndarray) -> np.ndarray:
        self._ensure_initialized()
        dtype = np.float32 if self.float_dtype == "f32" else np.float64
        points = np.asarray(points, dtype=dtype)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        n_panels = self.lattice.n_panels
        if n_panels == 0:
            return np.zeros_like(points)

        vertex_position = self.lattice.vertex_position.to_numpy()[:n_panels].astype(
            dtype, copy=False
        )
        velocity = np.zeros_like(points)

        if self.boundary_condition_type == "NEUMANN":
            normal = self.lattice.normal.to_numpy()[:n_panels].astype(dtype, copy=False)
            doublet_strength = self.lattice.source_strength.to_numpy()[:n_panels].astype(
                dtype, copy=False
            )
            compute_source_induced_velocity_kernel(
                vertex_position, normal, doublet_strength, points, velocity
            )
            return velocity

        doublet_strength = self.lattice.doublet_strength.to_numpy()[:n_panels].astype(
            dtype, copy=False
        )
        if n_panels >= 1000 or n_panels * len(points) >= 100_000:
            compute_induced_velocity_kernel(vertex_position, doublet_strength, points, velocity)
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
                return -coeff * r1xr2

            for j in range(n_panels):
                v0, v1, v2 = vertex_position[j, 0], vertex_position[j, 1], vertex_position[j, 2]
                vortex_strength = doublet_strength[j]
                for i in range(points.shape[0]):
                    p = points[i]
                    velocity[i] += vortex_strength * (
                        _segment_velocity(p, v0, v1)
                        + _segment_velocity(p, v1, v2)
                        + _segment_velocity(p, v2, v0)
                    )

        return velocity

    def advance(
        self,
        particles: Any = None,
        physics: Any = None,
        config: Any = None,
        freestream_velocity: np.ndarray | None = None,
        wake_velocity: Any = None,
        time_step_size: float | None = None,
        time: float | None = None,
        step: int | None = None,
        logging_interval_steps: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        """
        Advance panel simulation by one time step.

        Args:
            particles: Live VPM particle container (coupled mode).
            physics: VPM physics/evaluation backend used for induced velocity.
            config: VPM setup providing the freestream velocity when not given.
            freestream_velocity: Freestream velocity vector; defaults to the
                configured solver value.
            wake_velocity: Prescribed wake-induced velocity (standalone mode).
            time_step_size: Time step size for this advance.
            time: Current physical time.
            step: Current step index.
            logging_interval_steps: Steps between force log reports.

        Returns newly shed wake particles to be added to the VPM system.
        """
        if freestream_velocity is None and config is not None:
            freestream_velocity = np.array(getattr(config, "freestream_velocity", [1.0, 0.0, 0.0]))
        if freestream_velocity is None:
            freestream_velocity = (
                self.freestream_velocity
                if self.freestream_velocity is not None
                else np.array([1.0, 0.0, 0.0])
            )
        if time_step_size is None:
            time_step_size = 0.01
        if time is None:
            time = self._current_time
        if step is not None:
            self.step = step
        if logging_interval_steps is not None:
            self.logging_interval_steps = logging_interval_steps

        self.ensure_mesh_generated()

        # 1. Save history for BDF2
        self.lattice.save_old_doublet_strength()

        # 2. Update geometry via kinematics (if any)
        for body in self.lattice.bodies:
            kinematics = getattr(body, "kinematics", None)
            if kinematics is None:
                kinematics = getattr(body, "kinematics", None)
            if kinematics is not None:
                kinematics.update(
                    self,
                    time,
                    time_step_size,
                    (body.start_idx, body.start_idx + body.count),
                )

        # 3. Compute VPM-induced velocity at collocation points (if coupled)
        if particles is not None and physics is not None:
            # NumPy targets follow PhysicsBase's configured TREECODE route.
            # Passing the Taichi field directly bypassed that branch and launched
            # a direct M-by-N target kernel at every panel centre.
            n_panels = self.lattice.n_panels
            centres = self.lattice.panel_centre.to_numpy()[:n_panels]
            induced = physics.compute_target_velocity(
                particles,
                centres,
                include_freestream=False,  # Freestream added separately
            )
            lattice_velocity = self.lattice.velocity.to_numpy()
            lattice_velocity[:n_panels] = np.asarray(induced, dtype=lattice_velocity.dtype)
            self.lattice.velocity.from_numpy(lattice_velocity)
            wake_velocity = self.lattice.velocity

        # 4. Solve potential flow
        self.solve(freestream_velocity, wake_velocity, time)

        # cube_flow's vpm_boundary_condition-only panel supplies the irrotational boundary
        # correction, while the FVM owns force/pressure_coefficient reporting.  Avoid a second
        # surface-velocity/force pass unless this panel is authoritative for
        # particle dynamics or post-processing.
        if self.coupling_scope != "vpm_boundary_condition":
            self.compute_postprocess(
                freestream_velocity,
                freestream_velocity,
                self.density,
                time_step_size=time_step_size,
                coupled=(particles is not None),
            )
            self.compute_loads(freestream_velocity, wake_velocity, time_step_size, self.density)
            log_freq = self.logging_interval_steps
            if log_freq > 0 and self.step % log_freq == 0:
                try:
                    self.log_forces_table(self.density, freestream_velocity)
                except Exception as e:
                    print(f"   (Warning) Could not compute panel forces: {e}")

        self.results["time_history"].append(float(time))
        self._current_time = float(time)
        self._last_freestream_velocity = freestream_velocity
        self._last_wake_velocity = wake_velocity

        self.step += 1
        return None

    def advance_time(self, time_step_size: float, current_time: float) -> None:
        """Advance kinematics state and geometry for all surfaces (VLM-compatible API)."""
        self.ensure_mesh_generated()
        self._current_time = float(current_time)

        freestream_velocity = (
            self._last_freestream_velocity
            if hasattr(self, "_last_freestream_velocity")
            else np.array([1.0, 0.0, 0.0])
        )
        wake_velocity = self._last_wake_velocity if hasattr(self, "_last_wake_velocity") else None

        self.lattice.save_old_doublet_strength()
        for body in self.lattice.bodies:
            kinematics = getattr(body, "kinematics", None) or getattr(body, "kinematics", None)
            if kinematics is not None:
                kinematics.update(
                    self,
                    current_time,
                    time_step_size,
                    (body.start_idx, body.start_idx + body.count),
                )
        self.solve(freestream_velocity, wake_velocity, current_time)

    def compute_induced_velocity_direct(self, particles) -> None:
        """
        Compute velocity induced by panels on VPM particles and add to particles.velocity.

        This method is called by the core VPM solver to couple the panel method
        with the particle method. It extracts particle position, computes the
        induced velocity using the existing compute_induced_velocity method,
        and adds the result to particles.velocity.

        Args:
            particles: VPM particles object with position and velocity fields
        """
        if self.lattice is None or self.lattice.n_panels == 0:
            return

        n_particles_total = particles.n_particles_total
        if n_particles_total == 0:
            return

        np_dtype = getattr(particles, "_np_float_dtype", np.float32)

        # Synchronize any pending Taichi kernel writes before reading
        # particle data (compute_self_induced_velocity may have just written to these fields).
        ti.sync()

        # Extract particle position in the container's native dtype
        position = particles.position.to_numpy()[:n_particles_total].astype(np_dtype)

        # Compute induced velocity at particle position
        v_induced = self.compute_induced_velocity(position)

        # Add to particle velocity.
        # v_induced is f64 from numpy operations; v_current is f32.
        # We cast the sum to f32 for the taichi field.
        v_current = particles.velocity.to_numpy()[:n_particles_total]
        v_new = (v_current + v_induced).astype(np_dtype)

        # Write back using the particle container's copy kernel which
        # handles dtype conversion safely (avoids from_numpy issues on
        # fields previously written by Taichi kernels).
        particles._copy_to_taichi_vectors(v_new, particles.velocity, 0, n_particles_total)

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
        if self.lattice is None or self.lattice.n_panels == 0:
            return 0

        n_particles_total = particles.n_particles_total
        n_panels = self.lattice.n_panels

        if n_particles_total == 0 or n_panels == 0:
            return 0

        # Extract data
        position = particles.position.to_numpy()[:n_particles_total]
        panel_centre = self.lattice.panel_centre.to_numpy()[:n_panels]
        normal = self.lattice.normal.to_numpy()[:n_panels]
        vertex_position = self.lattice.vertex_position.to_numpy()[:n_panels]

        # Simple collision detection: check if particle is inside the body
        # by checking if it's on the negative side of all panel normal
        # (assuming outward-pointing normal for a closed body)
        keep_mask = np.ones(n_particles_total, dtype=bool)

        for i in range(n_particles_total):
            p = position[i]
            # Check distance to each panel center
            for j in range(n_panels):
                # Vector from panel center to particle
                r = p - panel_centre[j]
                # Perpendicular distance to panel plane
                d_perp = abs(np.dot(r, normal[j]))
                if d_perp < tolerance:
                    # Check if projection is inside triangle (simplified check)
                    # Project onto panel plane
                    p_proj = p - d_perp * normal[j] * np.sign(np.dot(r, normal[j]))
                    # Check if inside triangle using barycentric coordinates
                    v0 = vertex_position[j, 1] - vertex_position[j, 0]
                    v1 = vertex_position[j, 2] - vertex_position[j, 0]
                    v2 = p_proj - vertex_position[j, 0]
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
            particles.n_particles_total = 0
            return n_removed

        # Extract kept particles
        new_position = position[keep_mask]
        new_velocity = particles.velocity.to_numpy()[:n_particles_total][keep_mask]
        new_vortex_strength = particles.vortex_strength.to_numpy()[:n_particles_total][keep_mask]
        new_vorticity = particles.vorticity.to_numpy()[:n_particles_total][keep_mask]
        new_core_radius = particles.core_radius.to_numpy()[:n_particles_total][keep_mask]
        new_particle_volume = particles.particle_volume.to_numpy()[:n_particles_total][keep_mask]
        new_kinematic_viscosity = particles.kinematic_viscosity.to_numpy()[:n_particles_total][
            keep_mask
        ]
        new_eddy_viscosity = particles.eddy_viscosity.to_numpy()[:n_particles_total][keep_mask]
        new_effective_viscosity = particles.effective_viscosity.to_numpy()[:n_particles_total][
            keep_mask
        ]
        new_group_id = particles.group_id.to_numpy()[:n_particles_total][keep_mask]
        new_velocity_gradient = particles.velocity_gradient.to_numpy()[:n_particles_total][
            keep_mask
        ]
        new_strain_rate = particles.strain_rate.to_numpy()[:n_particles_total][keep_mask]

        # Write back to particles
        particles._copy_to_taichi_vectors(new_position, particles.position, 0, n_keep)
        particles._copy_to_taichi_vectors(new_velocity, particles.velocity, 0, n_keep)
        particles._copy_to_taichi_vectors(new_vortex_strength, particles.vortex_strength, 0, n_keep)
        particles._copy_to_taichi_vectors(new_vorticity, particles.vorticity, 0, n_keep)
        particles._copy_to_taichi_scalars(new_core_radius, particles.core_radius, 0, n_keep)
        particles._copy_to_taichi_scalars(new_particle_volume, particles.particle_volume, 0, n_keep)
        particles._copy_to_taichi_scalars(
            new_kinematic_viscosity, particles.kinematic_viscosity, 0, n_keep
        )
        particles._copy_to_taichi_scalars(new_eddy_viscosity, particles.eddy_viscosity, 0, n_keep)
        particles._copy_to_taichi_scalars(
            new_effective_viscosity, particles.effective_viscosity, 0, n_keep
        )
        particles._copy_to_taichi_scalars(new_group_id, particles.group_id, 0, n_keep)
        particles._copy_to_taichi_matrices(
            new_velocity_gradient, particles.velocity_gradient, 0, n_keep
        )
        particles._copy_to_taichi_matrices(new_strain_rate, particles.strain_rate, 0, n_keep)

        particles.n_particles_total = n_keep
        particles.sync_device_counter()

        print(f"   (Panel) Absorbed {n_removed} particles impinging on surface.")
        return n_removed

    def compute_postprocess(
        self,
        external_velocity_np: np.ndarray,
        reference_velocity: np.ndarray,
        density: float,
        time_step_size: float | None = None,
        coupled: bool = False,
    ) -> None:
        """
        Compute derived quantities (velocity, pressures, forces) after solve.

        Args:
            external_velocity_np: External velocity (N, 3) or (3,)
            reference_velocity: Reference velocity vector [ux, uy, uz] (m/s)
            density: Fluid density
            time_step_size: Time step size
            coupled: Whether in coupled mode (unused for panel method)
        """
        if self.lattice is None or self.lattice.n_panels == 0:
            return

        n = self.lattice.n_panels
        reference_velocity_mag = np.linalg.norm(reference_velocity)
        if reference_velocity_mag < 1e-10:
            reference_velocity_mag = 1.0

        # Resolve freestream_velocity
        if external_velocity_np.ndim == 1 and external_velocity_np.shape[0] == 3:
            freestream_velocity = external_velocity_np
        else:
            freestream_velocity = np.mean(external_velocity_np, axis=0)

        self._update_surface_velocity(freestream_velocity)

        # Compute pressure coefficients
        compute_pressure_bernoulli(
            self.surface_velocity,
            float(reference_velocity_mag),
            self.lattice.pressure_coefficient,
            n,
        )

        # Compute forces
        compute_forces_bernoulli(
            self.surface_velocity,
            float(reference_velocity_mag),
            self.lattice.area,
            self.lattice.normal,
            density,
            self.panel_force,
            n,
        )

    def compute_forces_coefficients(
        self,
        density: float,
        reference_velocity: np.ndarray | None = None,
        reference_area: float | None = None,
        reference_chord: float | None = None,
        reference_span: float | None = None,
    ) -> dict[str, float]:
        """
        Compute integrated aerodynamic forces and moments as coefficients.

        Returns a dictionary with canonical force, moment, coefficient, and
        reference-quantity names.

        Args:
            density: Fluid density (kg/m³)
            reference_velocity: Reference velocity vector [ux, uy, uz] (for coefficients and L/D axes).
                   If None, uses self.freestream_velocity or auto-computed.
            reference_area: Reference area (m²). If None, uses sum of panel area.
            reference_chord: Reference chord (m). If None, uses sqrt(reference_area).
            reference_span: Reference span (m). If None, uses sqrt(reference_area).

        Returns:
            Dictionary with force components and coefficients
        """
        if self.lattice is None or self.lattice.n_panels == 0:
            return {}

        if not self._solved:
            raise RuntimeError("Must solve system before computing forces")

        # Resolve reference values
        if reference_velocity is None:
            reference_velocity = (
                self.freestream_velocity
                if self.freestream_velocity is not None
                else np.array([1.0, 0.0, 0.0])
            )
        reference_velocity_mag = np.linalg.norm(reference_velocity)
        if reference_velocity_mag < 1e-10:
            reference_velocity_mag = 1.0

        n = self.lattice.n_panels
        panel_force = self.panel_force.to_numpy()[:n]
        total_force = np.sum(panel_force, axis=0)
        force_x, force_y, force_z = total_force

        # Decompose into lift, drag, side-force
        reference_direction = reference_velocity / reference_velocity_mag
        vertical_direction = np.array([0.0, 0.0, 1.0])
        lift_direction = (
            vertical_direction
            - np.dot(vertical_direction, reference_direction) * reference_direction
        )
        lift_direction_magnitude = np.linalg.norm(lift_direction)
        lift_direction = (
            lift_direction / lift_direction_magnitude
            if lift_direction_magnitude > 1e-10
            else vertical_direction
        )
        side_force_direction = np.cross(reference_direction, lift_direction)

        lift = float(np.dot(total_force, lift_direction))
        drag = float(np.dot(total_force, reference_direction))
        side_force = float(np.dot(total_force, side_force_direction))

        # Compute moments about reference center
        panel_centre = self.lattice.panel_centre.to_numpy()[:n]
        reference_point = np.mean(panel_centre, axis=0)
        total_moment = np.sum(np.cross(panel_centre - reference_point, panel_force), axis=0)
        moment_x, moment_y, moment_z = total_moment

        # Reference values
        if reference_area is None:
            panel_area = self.lattice.area.to_numpy()[:n]
            reference_area = float(np.sum(panel_area))
        if reference_chord is None:
            reference_chord = float(np.sqrt(reference_area))
        if reference_span is None:
            reference_span = float(np.sqrt(reference_area))

        # Non-dimensionalize
        dynamic_pressure = 0.5 * density * reference_velocity_mag**2
        force_normalization = dynamic_pressure * reference_area
        chord_moment_normalization = force_normalization * reference_chord
        span_moment_normalization = force_normalization * reference_span

        if dynamic_pressure > 1e-10 and force_normalization > 1e-10:
            lift_coefficient = lift / force_normalization
            drag_coefficient = drag / force_normalization
            side_force_coefficient = side_force / force_normalization
            force_coefficient_x = force_x / force_normalization
            force_coefficient_y = force_y / force_normalization
            force_coefficient_z = force_z / force_normalization
            rolling_moment_coefficient = moment_x / span_moment_normalization
            pitching_moment_coefficient = moment_y / chord_moment_normalization
            yawing_moment_coefficient = moment_z / span_moment_normalization
        else:
            lift_coefficient = drag_coefficient = side_force_coefficient = force_coefficient_x = (
                force_coefficient_y
            ) = force_coefficient_z = 0.0
            rolling_moment_coefficient = 0.0
            pitching_moment_coefficient = 0.0
            yawing_moment_coefficient = 0.0

        result = {
            "force_x": force_x,
            "force_y": force_y,
            "force_z": force_z,
            "lift": lift,
            "drag": drag,
            "side_force": side_force,
            "moment_x": moment_x,
            "moment_y": moment_y,
            "moment_z": moment_z,
            "force_coefficient_x": force_coefficient_x,
            "force_coefficient_y": force_coefficient_y,
            "force_coefficient_z": force_coefficient_z,
            "lift_coefficient": lift_coefficient,
            "drag_coefficient": drag_coefficient,
            "side_force_coefficient": side_force_coefficient,
            "rolling_moment_coefficient": rolling_moment_coefficient,
            "pitching_moment_coefficient": pitching_moment_coefficient,
            "yawing_moment_coefficient": yawing_moment_coefficient,
            "dynamic_pressure": dynamic_pressure,
            "reference_area": reference_area,
            "reference_chord": reference_chord,
            "reference_span": reference_span,
            "reference_point": reference_point,
        }

        self._last_forces = result
        self._last_reference_velocity = reference_velocity
        return result

    def log_forces_table(
        self, density: float, reference_velocity: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Log panel forces in a formatted table matching VLM diagnostics style.

        Prints per-surface forces and total forces with descriptions.

        Args:
            density: Fluid density (kg/m^3)
            reference_velocity: Reference velocity vector

        Returns:
            Dictionary of force coefficients
        """
        print("\n" + "-" * 60)
        print("PANEL AERODYNAMIC FORCES")
        print("-" * 60)
        method_name = self.force_config.method
        print(f"  Force computation method: {method_name}")
        print("    lift = force perpendicular to freestream")
        print("    drag = force parallel to freestream")
        print("    lift_coefficient, drag_coefficient = normalized force coefficients")
        print()

        # Get total forces
        total_forces = self.compute_forces_coefficients(density, reference_velocity)

        # Print totals
        L = total_forces.get("lift", 0.0)
        D = total_forces.get("drag", 0.0)
        lift_coefficient = total_forces.get("lift_coefficient", 0.0)
        drag_coefficient = total_forces.get("drag_coefficient", 0.0)
        side_force_coefficient = total_forces.get("side_force_coefficient", 0.0)
        L_D = L / D if abs(D) > 1e-10 else float("inf")

        print(
            f"  {'TOTAL':<15} {L:>12.3f} {D:>12.3f} {lift_coefficient:>10.3f} {drag_coefficient:>10.3f}"
        )
        print()
        print(f"  Lift/Drag Ratio (L/D)    : {L_D:.2f}")
        print(f"  Side-force coefficient   : {side_force_coefficient:.3f}")

        # Moments
        rolling_moment_coefficient = total_forces.get("rolling_moment_coefficient", 0.0)
        pitching_moment_coefficient = total_forces.get("pitching_moment_coefficient", 0.0)
        yawing_moment_coefficient = total_forces.get("yawing_moment_coefficient", 0.0)

        print()
        print("  Moment Coefficients:")
        print(f"    rolling_moment_coefficient : {rolling_moment_coefficient:>12.3f}")
        print(f"    pitching_moment_coefficient: {pitching_moment_coefficient:>12.3f}")
        print(f"    yawing_moment_coefficient  : {yawing_moment_coefficient:>12.3f}")

        if "reference_point" in total_forces:
            r = total_forces["reference_point"]
            print(f"    Ref Center   : [{r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}]")

        print("-" * 60, flush=True)

        return total_forces

    def compute_per_surface_forces(
        self,
        density: float,
        reference_velocity: np.ndarray | None = None,
        reference_area: float | None = None,
        reference_chord: float | None = None,
        reference_span: float | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute forces for each individual surface (body group).

        Args:
            density: Fluid density (kg/m³)
            reference_velocity: Reference velocity vector [ux, uy, uz]
            reference_area: Reference area (m²). If None, uses sum of panel area.
            reference_chord: Reference chord (m). If None, uses sqrt(reference_area).
            reference_span: Reference span (m). If None, uses sqrt(reference_area).

        Returns:
            Dictionary mapping surface name to force dictionary
        """
        if self.lattice is None or self.lattice.n_panels == 0:
            return {}

        if not self._solved:
            return {}

        # Resolve reference values
        if reference_velocity is None:
            reference_velocity = (
                self.freestream_velocity
                if self.freestream_velocity is not None
                else np.array([1.0, 0.0, 0.0])
            )
        reference_velocity_mag = np.linalg.norm(reference_velocity)
        if reference_velocity_mag < 1e-10:
            reference_velocity_mag = 1.0

        n = self.lattice.n_panels
        panel_force = self.panel_force.to_numpy()[:n]
        panel_area = self.lattice.area.to_numpy()[:n]

        if reference_area is None:
            reference_area = float(np.sum(panel_area))
        if reference_chord is None:
            reference_chord = float(np.sqrt(reference_area))
        if reference_span is None:
            reference_span = float(np.sqrt(reference_area))

        dynamic_pressure = 0.5 * density * reference_velocity_mag**2
        force_normalization = dynamic_pressure * reference_area
        reference_direction = (
            reference_velocity / reference_velocity_mag
            if reference_velocity_mag > 1e-10
            else np.array([1.0, 0.0, 0.0])
        )

        result = {}
        for body in self.lattice.bodies:
            uid = body.uid
            body_force = panel_force[body.start_idx : body.start_idx + body.count]
            surface_force = np.sum(body_force, axis=0)
            force_x, force_y, force_z = surface_force
            drag = np.dot(surface_force, reference_direction)
            lift_vector = surface_force - drag * reference_direction
            lift = np.linalg.norm(lift_vector)
            if np.dot(lift_vector, np.array([0, 0, 1])) < 0:
                lift = -lift
            lift_coefficient = lift / force_normalization if force_normalization > 1e-10 else 0.0
            drag_coefficient = drag / force_normalization if force_normalization > 1e-10 else 0.0
            result[uid] = {
                "lift": lift,
                "drag": drag,
                "force_x": force_x,
                "force_y": force_y,
                "force_z": force_z,
                "lift_coefficient": lift_coefficient,
                "drag_coefficient": drag_coefficient,
                "panel_count": body.count,
            }

        return result
