"""Discrete direct-forcing IBM operators (Pinelli et al. 2010 / Constant et al.).

Builds the interpolation (Eulerian → Lagrangian) and spreading
(Lagrangian → Eulerian) operators from the regularised 3-point Roma–Peskin
delta kernel, with the Pinelli quadrature weights solved so that spreading is
the adjoint-consistent inverse of
interpolation (spread-then-interpolate reproduces constants).

Per momentum predictor the forcing is

    F_s = (prescribed_velocity_s − I[u*]_s) / Δt (Lagrangian, acceleration)
    f_j = Σ_s F_s δ_h(x_j − X_s) ε_s           (Eulerian, acceleration)

and ``ρ f`` enters the momentum equation through the existing
``source_explicit`` hook. The validation reference is
``docs/literature/Constant2016.pdf``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.spatial import cKDTree  # type: ignore[missing-module-attribute]

from .body import ImmersedBody


def roma_delta_1d(r: np.ndarray) -> np.ndarray:
    """Roma–Peskin 3-point regularised delta φ(r), support |r| ≤ 1.5.

    Constant et al., Eq. (6).  ``r`` is the marker–cell distance in units of
    the grid spacing; the physical kernel is divided by that spacing per
    direction.
    """
    r = np.abs(np.asarray(r, dtype=np.float64))
    kernel_weight = np.zeros_like(r)
    inner = r <= 0.5
    kernel_weight[inner] = (1.0 + np.sqrt(np.maximum(-3.0 * r[inner] ** 2 + 1.0, 0.0))) / 3.0
    outer = (r > 0.5) & (r <= 1.5)
    kernel_weight[outer] = (
        5.0 - 3.0 * r[outer] - np.sqrt(np.maximum(-3.0 * (1.0 - r[outer]) ** 2 + 1.0, 0.0))
    ) / 6.0
    return kernel_weight


def _detect_empty_axis(mesh_data, geo_data) -> int | None:
    """Return the out-of-plane axis index for 2D single-layer meshes, else None.

    Detected from ``empty`` boundary patches: their face normals point along
    the extruded (non-solved) direction.
    """
    for b in mesh_data["boundary"]:
        if b.get("velocity_type") == "empty":
            sf = geo_data["face_area_vector"][b["start_face"] : b["start_face"] + b["n_faces"]]
            n = np.abs(sf).sum(axis=0)
            return int(np.argmax(n))
    return None


class IBMForcing:
    """Interpolation/spreading operators and force computation for a set of bodies.

    Args:
        mesh_data: Mesh dictionary (needs owners/boundary for 2D detection).
        geo_data:  Geometry dictionary (cell_centre, volumes, face_area_vector).
        bodies:    List of :class:`ImmersedBody`.
        grid_spacing: Eulerian grid spacing in the uniform region around the
                   bodies.  If ``None``, inferred from the median cell size of
                   the support cells.
        empty_axis: Out-of-plane axis for 2D meshes (0/1/2), ``None`` for 3D,
                   or ``"auto"`` (default) to detect from ``empty`` patches.

    Notes:
        The mesh must be approximately uniform within two grid spacings of
        every marker; markers should be spaced about one grid spacing apart. Both are
        checked and reported by :meth:`diagnostics`.
    """

    def __init__(
        self,
        mesh_data,
        geo_data,
        bodies,
        grid_spacing: float | None = None,
        empty_axis="auto",
    ):
        if isinstance(bodies, ImmersedBody):
            bodies = [bodies]
        if not bodies:
            raise ValueError("IBMForcing requires at least one ImmersedBody")
        if grid_spacing is not None and (not np.isfinite(grid_spacing) or grid_spacing <= 0.0):
            raise ValueError("IBM grid_spacing must be finite and positive")
        self.bodies = list(bodies)
        self.mesh_data = mesh_data
        self.geo_data = geo_data

        n_cells = mesh_data["n_cells"]
        cell_centre = geo_data["cell_centre"][:n_cells]
        volumes = geo_data["cell_volume"][:n_cells]

        # Marker bookkeeping: one global array, slices per body.
        self._body_slices = []
        start = 0
        for b in self.bodies:
            self._body_slices.append(slice(start, start + b.n_markers))
            start += b.n_markers
        self.marker_position = np.vstack([body.position for body in self.bodies])
        self.prescribed_velocity = np.vstack([b.prescribed_velocity for b in self.bodies])
        n_markers_total = self.marker_position.shape[0]

        # Active axes (2D single-layer meshes: skip the extruded direction).
        if empty_axis == "auto":
            empty_axis = _detect_empty_axis(mesh_data, geo_data)
        self.empty_axis = empty_axis
        axes = [a for a in range(3) if a != empty_axis]
        self.axes = axes
        ndim = len(axes)

        # Effective quadrature volume: cell volume, divided by the z-extent
        # for 2D meshes so the kernel (1/h per active direction) integrates
        # to one against it.
        if empty_axis is not None:
            pts = mesh_data["vertex_position"][:, empty_axis]
            self._depth = float(pts.max() - pts.min())
            dv = volumes / self._depth
        else:
            self._depth = 1.0
            dv = volumes

        # Grid spacing h near the body.
        tree = cKDTree(cell_centre)
        if grid_spacing is None:
            _, nearest = tree.query(self.marker_position, k=1)
            grid_spacing = float(np.median(dv[nearest] ** (1.0 / ndim)))
        self.grid_spacing = float(grid_spacing)
        if not np.isfinite(self.grid_spacing) or self.grid_spacing <= 0.0:
            raise ValueError("Inferred IBM grid spacing is not finite and positive")

        # --- Support search + kernel evaluation --------------------------- #
        # Kernel support is 1.5h per active axis; search a bounding sphere.
        support_radius = 1.5 * self.grid_spacing * np.sqrt(ndim) * 1.001
        supports = tree.query_ball_point(self.marker_position, r=support_radius)

        rows, cols, delta_vals = [], [], []
        for s, cells in enumerate(supports):
            if not cells:
                continue
            cells = np.asarray(cells, dtype=np.int64)
            d = cell_centre[cells] - self.marker_position[s]
            kernel_weight = np.ones(len(cells))
            for a in axes:
                kernel_weight *= roma_delta_1d(d[:, a] / self.grid_spacing)
            nz = kernel_weight > 0.0
            rows.append(np.full(nz.sum(), s, dtype=np.int64))
            cols.append(cells[nz])
            delta_vals.append(kernel_weight[nz] / self.grid_spacing**ndim)

        if not rows or sum(len(r) for r in rows) == 0:
            raise ValueError(
                "IBM markers have no Eulerian support cells - are the bodies "
                "inside the mesh, and is h consistent with the local spacing?"
            )
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        delta_vals = np.concatenate(delta_vals)

        # D: raw kernel values delta_h(x_j - X_s)   (Ns x Ncells)
        self._kernel_matrix = csr_matrix(
            (delta_vals, (rows, cols)), shape=(n_markers_total, n_cells)
        )
        # W: interpolation weights delta * dV, row-normalised so constants are
        # reproduced exactly even on mildly non-uniform supports.
        weighted_kernel_matrix = self._kernel_matrix.multiply(dv[np.newaxis, :]).tocsr()
        self._row_sums = np.asarray(weighted_kernel_matrix.sum(axis=1)).ravel()
        if np.any(self._row_sums <= 0.1):
            bad = int(np.sum(self._row_sums <= 0.1))
            raise ValueError(
                f"{bad} IBM markers have (near-)empty kernel support "
                f"(minimum row sum {self._row_sums.min():.3g}) - check "
                f"grid_spacing={self.grid_spacing:.4g} "
                "against the local mesh spacing."
            )
        inv = 1.0 / self._row_sums
        self._interpolation_matrix = weighted_kernel_matrix.multiply(inv[:, np.newaxis]).tocsr()

        # Solve the Pinelli quadrature system so the transfer pair is consistent.
        quadrature_system_matrix = (self._interpolation_matrix @ self._kernel_matrix.T).tocsr()
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=MatrixRankWarning)
            self.quadrature_weight = spsolve(
                quadrature_system_matrix.tocsc(), np.ones(n_markers_total)
            )
        if not np.all(np.isfinite(self.quadrature_weight)):
            raise ValueError("IBM marker quadrature system is singular or ill-conditioned")
        quadrature_weight_scale = max(
            float(np.max(np.abs(self.quadrature_weight))),
            np.finfo(np.float64).tiny,
        )
        if float(np.min(self.quadrature_weight)) < -1.0e-10 * quadrature_weight_scale:
            raise ValueError(
                "IBM marker quadrature produced negative weights; the marker cloud is "
                "ill-conditioned. Increase the marker spacing or separate nearby surfaces."
            )
        self._max_quadrature_residual = float(
            np.abs(quadrature_system_matrix @ self.quadrature_weight - 1.0).max()
        )
        if self._max_quadrature_residual > 1.0e-8:
            raise ValueError(
                "IBM maximum marker quadrature residual "
                f"{self._max_quadrature_residual:.3e} exceeds 1e-8"
            )

        # State from the last compute_force call (for force logging).
        self.last_marker_acceleration = np.zeros((n_markers_total, 3))
        self.last_slip = 0.0
        self._solid_cell_masks = [
            (
                body.contains(cell_centre, include_boundary=False)
                if body.has_solid_geometry
                else np.zeros(n_cells, dtype=bool)
            )
            for body in self.bodies
        ]
        self._fictitious_fluid_momentum_rate = {
            body.name: np.zeros(3, dtype=np.float64) for body in self.bodies
        }

    # ------------------------------------------------------------------ #
    # Operators
    # ------------------------------------------------------------------ #

    def interpolate(self, velocity: np.ndarray) -> np.ndarray:
        """Interpolate a cell field to the marker positions."""
        n = self.mesh_data["n_cells"]
        return self._interpolation_matrix @ velocity[:n]

    def spread(self, marker_acceleration: np.ndarray) -> np.ndarray:
        """Spread a Lagrangian marker-acceleration field to cells."""
        return self._kernel_matrix.T @ (marker_acceleration * self.quadrature_weight[:, np.newaxis])

    def compute_force(self, velocity_star: np.ndarray, time_step_size: float) -> np.ndarray:
        """Direct-forcing acceleration field from the predictor velocity.

        Returns the Eulerian acceleration ``f`` (n_elements, 3); the momentum
        source is ``ρ f``.  Also records the Lagrangian force and the slip
        error for diagnostics.
        """
        marker_velocity = self.interpolate(velocity_star)
        marker_acceleration = (self.prescribed_velocity - marker_velocity) / time_step_size
        self.last_marker_acceleration = marker_acceleration
        self.last_slip = float(
            np.linalg.norm(self.prescribed_velocity - marker_velocity, axis=1).max()
        )
        return self.spread(marker_acceleration)

    def begin_step(self) -> None:
        """Reset the per-step Lagrangian force accumulator (for force logging)."""
        self.last_marker_acceleration = np.zeros_like(self.last_marker_acceleration)

    def multidirect_correct(
        self, velocity: np.ndarray, time_step_size: float, n_iterations: int = 2
    ) -> None:
        """Multidirect forcing iterations (Kempe & Fröhlich 2012 / Breugem 2012).

        After the forced momentum solve the marker slip is not exactly zero
        (the implicit operator and the pressure projection redistribute part
        of the force).  Each iteration here applies the *residual* force
        explicitly, ``velocity += Δt·S[(prescribed_velocity − I[velocity])/Δt]`` — a unit-gain update
        at the markers, so it converges the no-slip condition within the step
        without feedback overshoot.  The applied increments are accumulated
        into ``last_marker_acceleration`` so the logged body force stays consistent with the
        momentum actually imparted to the fluid.

        Args:
            velocity: Velocity field, mutated in place for interior cells.
            time_step_size: Time-step size.
            n_iterations: Number of residual-forcing iterations.
        """
        n = self.mesh_data["n_cells"]
        for _ in range(n_iterations):
            marker_acceleration_increment = (
                self.prescribed_velocity - self.interpolate(velocity)
            ) / time_step_size
            velocity[:n] += time_step_size * self.spread(marker_acceleration_increment)
            self.last_marker_acceleration = (
                self.last_marker_acceleration + marker_acceleration_increment
            )
        self.last_slip = float(
            np.linalg.norm(self.prescribed_velocity - self.interpolate(velocity), axis=1).max()
        )

    def slip_error(self, velocity: np.ndarray) -> float:
        """Max marker slip ``max_s |interpolated_velocity_s − prescribed_velocity_s|``."""
        u_marker = self.interpolate(velocity)
        return float(np.linalg.norm(self.prescribed_velocity - u_marker, axis=1).max())

    def update_fictitious_fluid_momentum_rate(
        self,
        velocity: np.ndarray,
        velocity_old: np.ndarray,
        time_step_size: float,
    ) -> None:
        r"""Update the interior-fluid term used by the hydrodynamic force.

        A volume-filled direct-forcing method accelerates numerical fluid on
        both sides of the immersed surface.  The reaction to the forcing must
        therefore be corrected by the momentum rate of the fictitious fluid
        inside the solid:

        The returned body force combines the reaction to immersed forcing with
        the time rate of change of fictitious-fluid momentum inside the body.

        Bodies without exact solid geometry retain the uncorrected transfer
        force because their interior is undefined.
        """
        if not np.isfinite(time_step_size) or time_step_size <= 0.0:
            raise ValueError("IBM force correction requires a finite positive time step")
        volumes = np.asarray(self.geo_data["cell_volume"], dtype=np.float64)
        delta_velocity = (
            np.asarray(velocity)[: len(volumes)] - np.asarray(velocity_old)[: len(volumes)]
        )
        for body, mask in zip(self.bodies, self._solid_cell_masks, strict=True):
            if np.any(mask):
                self._fictitious_fluid_momentum_rate[body.name] = np.sum(
                    delta_velocity[mask] * volumes[mask, np.newaxis], axis=0
                ) / float(time_step_size)
            else:
                self._fictitious_fluid_momentum_rate[body.name].fill(0.0)

    def forcing_reaction_forces(self, density: float = 1.0) -> dict:
        """Return the raw reaction to the distributed IBM forcing.

        This quantity includes momentum spent accelerating the fictitious
        fluid inside a volume-filled immersed body.  Use :meth:`body_forces`
        for the corrected hydrodynamic load.
        """
        out = {}
        scale = self.quadrature_weight * self._row_sums
        for body, sl in zip(self.bodies, self._body_slices, strict=True):
            force = -density * np.sum(
                self.last_marker_acceleration[sl] * scale[sl, np.newaxis], axis=0
            )
            if self.empty_axis is not None:
                force = force * self._depth
            out[body.name] = force
        return out

    def body_forces(self, density: float = 1.0) -> dict:
        """Hydrodynamic force on each body from the last forcing evaluation.

        The force the fluid exerts on the body is minus the momentum injected
        by the forcing plus the momentum rate of the fictitious fluid inside
        the solid.  Before a solver time step is completed (or for marker-only
        bodies without solid geometry), the latter is zero.  For 2D meshes
        this is the force on the full extruded depth.
        """
        out = self.forcing_reaction_forces(density=density)
        for body in self.bodies:
            out[body.name] = (
                out[body.name] + density * self._fictitious_fluid_momentum_rate[body.name]
            )
        return out

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def diagnostics(self) -> dict:
        """Return setup-quality metrics for marker spacing and quadrature."""
        marker_spacing_ratio_by_body = {}
        for body, sl in zip(self.bodies, self._body_slices, strict=True):
            body_position = self.marker_position[sl]
            if body_position.shape[0] > 1:
                tree = cKDTree(body_position)
                distance, _ = tree.query(body_position, k=2)
                marker_spacing_ratio_by_body[body.name] = float(
                    np.median(distance[:, 1]) / self.grid_spacing
                )
            else:
                marker_spacing_ratio_by_body[body.name] = float("nan")
        return {
            "n_markers_total": int(self.marker_position.shape[0]),
            "grid_spacing": self.grid_spacing,
            "marker_spacing_ratio_by_body": marker_spacing_ratio_by_body,
            "min_kernel_row_sum": float(self._row_sums.min()),
            "max_kernel_row_sum": float(self._row_sums.max()),
            "max_quadrature_residual": self._max_quadrature_residual,
            "median_quadrature_weight_to_cell_measure_ratio": float(
                np.median(self.quadrature_weight) / self.grid_spacing ** len(self.axes)
            ),
            "min_quadrature_weight": float(np.min(self.quadrature_weight)),
            "max_quadrature_weight": float(np.max(self.quadrature_weight)),
        }
