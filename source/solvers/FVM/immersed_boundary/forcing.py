"""Discrete direct-forcing IBM operators (Pinelli et al. 2010 / Constant et al.).

Builds the interpolation (Eulerian → Lagrangian) and spreading
(Lagrangian → Eulerian) operators from the regularised 3-point Roma–Peskin
delta kernel, with the Pinelli quadrature weights ``ε`` solved from
``A ε = 1`` so that spreading is the adjoint-consistent inverse of
interpolation (spread-then-interpolate reproduces constants).

Per momentum predictor the forcing is

    F_s = (U_target_s − I[u*]_s) / Δt          (Lagrangian, acceleration)
    f_j = Σ_s F_s δ_h(x_j − X_s) ε_s           (Eulerian, acceleration)

and ``ρ f`` enters the momentum equation through the existing
``source_explicit`` hook.  See docs/plans/2026-07-fvm-ibm-design.md.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .body import ImmersedBody


def roma_delta_1d(r: np.ndarray) -> np.ndarray:
    """Roma–Peskin 3-point regularised delta φ(r), support |r| ≤ 1.5.

    Constant et al., Eq. (6).  ``r`` is the marker–cell distance in units of
    the grid spacing ``h``; the physical kernel is ``φ(r)/h`` per direction.
    """
    r = np.abs(np.asarray(r, dtype=np.float64))
    phi = np.zeros_like(r)
    inner = r <= 0.5
    phi[inner] = (1.0 + np.sqrt(np.maximum(-3.0 * r[inner] ** 2 + 1.0, 0.0))) / 3.0
    outer = (r > 0.5) & (r <= 1.5)
    phi[outer] = (
        5.0 - 3.0 * r[outer] - np.sqrt(np.maximum(-3.0 * (1.0 - r[outer]) ** 2 + 1.0, 0.0))
    ) / 6.0
    return phi


def _detect_empty_axis(mesh_data, geo_data) -> int | None:
    """Return the out-of-plane axis index for 2D single-layer meshes, else None.

    Detected from ``empty`` boundary patches: their face normals point along
    the extruded (non-solved) direction.
    """
    for b in mesh_data["boundary"]:
        if b.get("type") == "empty" or b.get("bc_type_U") == "empty":
            sf = geo_data["face_sf"][b["startFace"] : b["startFace"] + b["nFaces"]]
            n = np.abs(sf).sum(axis=0)
            return int(np.argmax(n))
    return None


class IBMForcing:
    """Interpolation/spreading operators and force computation for a set of bodies.

    Args:
        mesh_data: Mesh dictionary (needs owners/boundary for 2D detection).
        geo_data:  Geometry dictionary (centroids, volumes, face_sf).
        bodies:    List of :class:`ImmersedBody`.
        h:         Eulerian grid spacing in the (uniform) region around the
                   bodies.  If ``None``, inferred from the median cell size of
                   the support cells.
        empty_axis: Out-of-plane axis for 2D meshes (0/1/2), ``None`` for 3D,
                   or ``"auto"`` (default) to detect from ``empty`` patches.

    Notes:
        The mesh must be approximately uniform (spacing ``h``) within ~2h of
        every marker; markers should be spaced ``≈ h`` apart.  Both are
        checked and reported by :meth:`diagnostics`.
    """

    def __init__(self, mesh_data, geo_data, bodies, h: float | None = None, empty_axis="auto"):
        if isinstance(bodies, ImmersedBody):
            bodies = [bodies]
        if not bodies:
            raise ValueError("IBMForcing requires at least one ImmersedBody")
        self.bodies = list(bodies)
        self.mesh_data = mesh_data
        self.geo_data = geo_data

        n_elements = mesh_data["n_elements"]
        centroids = geo_data["element_centroids"][:n_elements]
        volumes = geo_data["element_volumes"][:n_elements]

        # Marker bookkeeping: one global array, slices per body.
        self._body_slices = []
        start = 0
        for b in self.bodies:
            self._body_slices.append(slice(start, start + b.n_markers))
            start += b.n_markers
        self.X = np.vstack([b.X for b in self.bodies])
        self.U_target = np.vstack([b.U_target for b in self.bodies])
        ns = self.X.shape[0]

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
            pts = mesh_data["points"][:, empty_axis]
            self._depth = float(pts.max() - pts.min())
            dv = volumes / self._depth
        else:
            self._depth = 1.0
            dv = volumes

        # Grid spacing h near the body.
        tree = cKDTree(centroids)
        if h is None:
            _, nearest = tree.query(self.X, k=1)
            h = float(np.median(dv[nearest] ** (1.0 / ndim)))
        self.h = float(h)

        # --- Support search + kernel evaluation --------------------------- #
        # Kernel support is 1.5h per active axis; search a bounding sphere.
        radius = 1.5 * self.h * np.sqrt(ndim) * 1.001
        supports = tree.query_ball_point(self.X, r=radius)

        rows, cols, delta_vals = [], [], []
        for s, cells in enumerate(supports):
            if not cells:
                continue
            cells = np.asarray(cells, dtype=np.int64)
            d = centroids[cells] - self.X[s]
            phi = np.ones(len(cells))
            for a in axes:
                phi *= roma_delta_1d(d[:, a] / self.h)
            nz = phi > 0.0
            rows.append(np.full(nz.sum(), s, dtype=np.int64))
            cols.append(cells[nz])
            delta_vals.append(phi[nz] / self.h**ndim)

        if not rows or sum(len(r) for r in rows) == 0:
            raise ValueError(
                "IBM markers have no Eulerian support cells - are the bodies "
                "inside the mesh, and is h consistent with the local spacing?"
            )
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        delta_vals = np.concatenate(delta_vals)

        # D: raw kernel values delta_h(x_j - X_s)   (Ns x Ncells)
        self._D = csr_matrix((delta_vals, (rows, cols)), shape=(ns, n_elements))
        # W: interpolation weights delta * dV, row-normalised so constants are
        # reproduced exactly even on mildly non-uniform supports.
        w_raw = self._D.multiply(dv[np.newaxis, :]).tocsr()
        self._row_sums = np.asarray(w_raw.sum(axis=1)).ravel()
        if np.any(self._row_sums <= 0.1):
            bad = int(np.sum(self._row_sums <= 0.1))
            raise ValueError(
                f"{bad} IBM markers have (near-)empty kernel support "
                f"(row sums min {self._row_sums.min():.3g}) - check h={self.h:.4g} "
                "against the local mesh spacing."
            )
        inv = 1.0 / self._row_sums
        self._W = w_raw.multiply(inv[:, np.newaxis]).tocsr()

        # Pinelli quadrature: solve A eps = 1 with A_sk = sum_j W[s,j] D[k,j],
        # so that interpolate(spread(F)) ~ F (consistency of the transfer pair).
        A = (self._W @ self._D.T).tocsr()
        self.eps = spsolve(A.tocsc(), np.ones(ns))
        self._quadrature_residual = float(np.abs(A @ self.eps - 1.0).max())

        # State from the last compute_force call (for force logging).
        self.last_F = np.zeros((ns, 3))
        self.last_slip = 0.0

    # ------------------------------------------------------------------ #
    # Operators
    # ------------------------------------------------------------------ #

    def interpolate(self, U: np.ndarray) -> np.ndarray:
        """Interpolate a cell field to the markers: ``I[U]_s`` (Ns, ...)."""
        n = self.mesh_data["n_elements"]
        return self._W @ U[:n]

    def spread(self, F: np.ndarray) -> np.ndarray:
        """Spread a Lagrangian field to cells: ``S[F]_j = Σ_s F_s δ_s(x_j) ε_s``."""
        return self._D.T @ (F * self.eps[:, np.newaxis])

    def compute_force(self, U_star: np.ndarray, dt: float) -> np.ndarray:
        """Direct-forcing acceleration field from the predictor velocity.

        Returns the Eulerian acceleration ``f`` (n_elements, 3); the momentum
        source is ``ρ f``.  Also records the Lagrangian force and the slip
        error for diagnostics.
        """
        u_marker = self.interpolate(U_star)
        F = (self.U_target - u_marker) / dt
        self.last_F = F
        self.last_slip = float(np.linalg.norm(self.U_target - u_marker, axis=1).max())
        return self.spread(F)

    def begin_step(self) -> None:
        """Reset the per-step Lagrangian force accumulator (for force logging)."""
        self.last_F = np.zeros_like(self.last_F)

    def multidirect_correct(self, U: np.ndarray, dt: float, n_iter: int = 2) -> None:
        """Multidirect forcing iterations (Kempe & Fröhlich 2012 / Breugem 2012).

        After the forced momentum solve the marker slip is not exactly zero
        (the implicit operator and the pressure projection redistribute part
        of the force).  Each iteration here applies the *residual* force
        explicitly, ``U += Δt·S[(U_target − I[U])/Δt]`` — a unit-gain update
        at the markers, so it converges the no-slip condition within the step
        without feedback overshoot.  The applied increments are accumulated
        into ``last_F`` so the logged body force stays consistent with the
        momentum actually imparted to the fluid.

        Args:
            U:      Velocity field (mutated in place, interior cells).
            dt:     Time-step size.
            n_iter: Number of residual-forcing iterations.
        """
        n = self.mesh_data["n_elements"]
        for _ in range(n_iter):
            dF = (self.U_target - self.interpolate(U)) / dt
            U[:n] += dt * self.spread(dF)
            self.last_F = self.last_F + dF
        self.last_slip = float(np.linalg.norm(self.U_target - self.interpolate(U), axis=1).max())

    def slip_error(self, U: np.ndarray) -> float:
        """Max marker slip ``max_s |I[U]_s − U_target_s|`` (no-slip monitor)."""
        u_marker = self.interpolate(U)
        return float(np.linalg.norm(self.U_target - u_marker, axis=1).max())

    def body_forces(self, rho: float = 1.0) -> dict:
        """Hydrodynamic force on each body from the last forcing evaluation.

        The force the fluid exerts on the body is minus the momentum injected
        by the forcing: ``F_body = −ρ Σ_s F_s ε_s w_s`` (identical to
        ``−ρ Σ_j f_j V_j``).  For 2D meshes this is the force on the full
        extruded depth.
        """
        out = {}
        scale = self.eps * self._row_sums
        for body, sl in zip(self.bodies, self._body_slices, strict=True):
            f = -rho * np.sum(self.last_F[sl] * scale[sl, np.newaxis], axis=0)
            if self.empty_axis is not None:
                f = f * self._depth
            out[body.name] = f
        return out

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def diagnostics(self) -> dict:
        """Setup-quality metrics (marker spacing ratio, kernel sums, ε residual)."""
        # Marker spacing ratio alpha = ds / h per body (nearest-neighbour).
        alphas = {}
        for body, sl in zip(self.bodies, self._body_slices, strict=True):
            Xb = self.X[sl]
            if Xb.shape[0] > 1:
                t = cKDTree(Xb)
                dist, _ = t.query(Xb, k=2)
                alphas[body.name] = float(np.median(dist[:, 1]) / self.h)
            else:
                alphas[body.name] = float("nan")
        return {
            "n_markers": int(self.X.shape[0]),
            "h": self.h,
            "alpha": alphas,
            "kernel_row_sum_min": float(self._row_sums.min()),
            "kernel_row_sum_max": float(self._row_sums.max()),
            "quadrature_residual": self._quadrature_residual,
            "eps_over_h": float(np.median(self.eps) / self.h ** len(self.axes)),
        }
