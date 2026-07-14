"""Subgrid-scale LES eddy-viscosity models for the FVM solver.

All models expose the same interface as :class:`..smagorinsky.Smagorinsky`
(``compute_nut(U, mesh_data, geo_data) -> nut[n_elements]`` and
``get_filter_info()``), so they are drop-in alternatives selected by
``TurbulenceConfig.model``.

Implemented here:

* **WALE** (Nicoud & Ducros 1999) — wall-adapting; ν_t → 0 with the correct
  y³ near-wall scaling and vanishes identically in pure shear (no test filter).
* **sigma** (Nicoud et al. 2011) — singular-value based; ν_t = 0 for any 2D,
  axisymmetric or pure-shear/solid-rotation flow, so no spurious dissipation of
  laminar or transitional regions.
* **DynamicSmagorinsky** (Germano–Lilly) — Smagorinsky coefficient computed
  dynamically via a box test filter; globally averaged + clipped for stability
  (appropriate for homogeneous-direction LES; the box filter on a general
  unstructured mesh is approximate — see the class docstring).

The classical (constant-coefficient) Smagorinsky and the ``"none"`` (ILES /
no-model) path live alongside these and are wired in :func:`create_model`.
"""

from __future__ import annotations

import numpy as np

from ..fields import gradients
from .smagorinsky import Smagorinsky, _compute_filter_width


def _velocity_gradient_tensor(U, mesh_data, geo_data):
    """Compute the velocity-gradient tensor for interior cells.

    Returns ``g[c, i, j] = ∂u_i/∂x_j`` for each cell *c*.

    The underlying gradient function (from
    :func:`..fields.gradients._resolve_gradient_fn`) returns
    ``grad[c, k, i] = ∂U_i/∂x_k``, so we transpose the last two axes.

    Args:
        U:        Velocity field ``(n_elements + n_boundary, 3)``.
        mesh_data: Mesh dictionary.
        geo_data:  Geometry dictionary.

    Returns:
        Velocity-gradient tensor ``(n_elements, 3, 3)``.
    """
    n_elements = mesh_data["n_elements"]
    grad_fn = gradients._resolve_gradient_fn(geo_data)
    grad = grad_fn(U, mesh_data, geo_data)[:n_elements]  # (n, 3, 3): [c,k,i]
    return np.transpose(grad, (0, 2, 1))  # (n, 3, 3): [c,i,j] = ∂u_i/∂x_j


def _strain_rate(g):
    """Symmetric part S_ij and its magnitude-squared S:S = S_ij S_ij."""
    S = 0.5 * (g + np.transpose(g, (0, 2, 1)))
    return S, np.sum(S * S, axis=(1, 2))


class WALE:
    """Wall-Adapting Local Eddy-viscosity (Nicoud & Ducros, 1999).

    The WALE model computes subgrid viscosity as

        ν_t = (C_w Δ)² · (S^d_ij S^d_ij)^{3/2}
                          / ( (S_ij S_ij)^{5/2} + (S^d_ij S^d_ij)^{5/4} )

    where *S^d_ij* is the traceless symmetric part of the velocity-gradient
    squared tensor.  The model recovers the correct y³ near-wall scaling
    and vanishes in pure shear.

    Args:
        mesh_data: Mesh dictionary.
        geo_data:  Geometry dictionary.
        Cw:        WALE model coefficient (default 0.325).
    """

    def __init__(self, mesh_data, geo_data, Cw=0.325):
        self.Cw = Cw
        self.mesh_data = mesh_data
        self.geo_data = geo_data

    def get_filter_info(self):
        """Return filter-width statistics and model name.

        Returns:
            Dict with keys ``model``, ``Cs`` (carries *C_w*), and
            ``filter_width_min/max/mean``.
        """
        delta = self.geo_data["element_volumes"] ** (1.0 / 3.0)
        return {
            "model": "WALE",
            "Cs": self.Cw,
            "filter_width_min": float(np.min(delta)),
            "filter_width_max": float(np.max(delta)),
            "filter_width_mean": float(np.mean(delta)),
        }

    def compute_nut(self, U, mesh_data=None, geo_data=None):
        """Compute the subgrid-scale turbulent viscosity (WALE model).

        Args:
            U:        Velocity field ``(n_elements + n_boundary, 3)``.
            mesh_data: Optional override mesh dictionary.
            geo_data:  Optional override geometry dictionary.

        Returns:
            Per-element eddy viscosity ``(n_elements,)``, clipped ≥ 0.
        """
        mesh_data = mesh_data or self.mesh_data
        geo_data = geo_data or self.geo_data

        g = _velocity_gradient_tensor(U, mesh_data, geo_data)
        _, SS = _strain_rate(g)

        # Traceless symmetric part of g² : Sd_ij = ½(g²_ij + g²_ji) − ⅓ δ_ij g²_kk
        g2 = np.einsum("cik,ckj->cij", g, g)
        trace = np.einsum("cii->c", g2)
        Sd = 0.5 * (g2 + np.transpose(g2, (0, 2, 1)))
        Sd[:, 0, 0] -= trace / 3.0
        Sd[:, 1, 1] -= trace / 3.0
        Sd[:, 2, 2] -= trace / 3.0
        SdSd = np.sum(Sd * Sd, axis=(1, 2))

        eps = 1e-30
        op = SdSd**1.5 / (SS**2.5 + SdSd**1.25 + eps)

        delta = _compute_filter_width(geo_data["element_volumes"], mesh_data, geo_data)
        nut = (self.Cw * delta) ** 2 * op
        return np.nan_to_num(np.clip(nut, 0.0, None), nan=0.0, posinf=0.0, neginf=0.0)


class Sigma:
    """sigma subgrid-scale model (Nicoud, Toda, Cabrit, Bose & Lee, 2011).

    Computes eddy viscosity from the singular values of the velocity-gradient
    tensor:

        ν_t = (C_σ Δ)² · σ₃(σ₁ − σ₂)(σ₂ − σ₃) / σ₁²

    The model yields ν_t = 0 for any 2D, axisymmetric, pure-shear, or
    solid-rotation flow, making it suitable for transitional and
    intermittently laminar regions.

    Args:
        mesh_data: Mesh dictionary.
        geo_data:  Geometry dictionary.
        Csigma:    sigma model coefficient (default 1.35).
    """

    def __init__(self, mesh_data, geo_data, Csigma=1.35):
        self.Csigma = Csigma
        self.mesh_data = mesh_data
        self.geo_data = geo_data

    def get_filter_info(self):
        """Return filter-width statistics and model name.

        Returns:
            Dict with keys ``model``, ``Cs`` (carries *C_σ*), and
            ``filter_width_min/max/mean``.
        """
        delta = self.geo_data["element_volumes"] ** (1.0 / 3.0)
        return {
            "model": "sigma",
            "Cs": self.Csigma,
            "filter_width_min": float(np.min(delta)),
            "filter_width_max": float(np.max(delta)),
            "filter_width_mean": float(np.mean(delta)),
        }

    def compute_nut(self, U, mesh_data=None, geo_data=None):
        """Compute the subgrid-scale turbulent viscosity (sigma model).

        Args:
            U:        Velocity field ``(n_elements + n_boundary, 3)``.
            mesh_data: Optional override mesh dictionary.
            geo_data:  Optional override geometry dictionary.

        Returns:
            Per-element eddy viscosity ``(n_elements,)``, clipped ≥ 0.
        """
        mesh_data = mesh_data or self.mesh_data
        geo_data = geo_data or self.geo_data

        g = _velocity_gradient_tensor(U, mesh_data, geo_data)
        # Singular values σ1 ≥ σ2 ≥ σ3 ≥ 0 of the velocity-gradient tensor.
        sv = np.linalg.svd(g, compute_uv=False)  # (n, 3), descending
        s1, s2, s3 = sv[:, 0], sv[:, 1], sv[:, 2]

        eps = 1e-30
        d_sigma = s3 * (s1 - s2) * (s2 - s3) / (s1 * s1 + eps)

        delta = _compute_filter_width(geo_data["element_volumes"], mesh_data, geo_data)
        nut = (self.Csigma * delta) ** 2 * d_sigma
        return np.nan_to_num(np.clip(nut, 0.0, None), nan=0.0, posinf=0.0, neginf=0.0)


class DynamicSmagorinsky:
    """Germano–Lilly dynamic Smagorinsky with a box test filter.

    The Smagorinsky coefficient is computed from the resolved field via the
    Germano identity and Lilly's least-squares contraction

        C = ⟨L_ij M_ij⟩ / ⟨M_ij M_ij⟩,   ν_t = C Δ² |S|,

    with ``L_ij`` the test-scale Leonard stresses and
    ``M_ij = 2Δ²[ (|S|S_ij)~ − α² |S~| S~_ij ]``.  The test filter ``~`` is a
    volume-weighted one-ring box average (test/grid width ratio α≈2 ⇒ α²≈4) and
    ``⟨·⟩`` is a **global** average — the most robust variant, suited to flows
    with at least one homogeneous direction.  On a general unstructured mesh the
    box filter is only approximate; for wall-bounded cases without a homogeneous
    direction prefer WALE or sigma.
    """

    def __init__(self, mesh_data, geo_data, alpha2=4.0):
        """Initialise the dynamic Smagorinsky model.

        Pre-computes the static volume-weighted box-filter denominator
        (one-ring neighbour sum) from the mesh topology.

        Args:
            mesh_data: Mesh dictionary.
            geo_data:  Geometry dictionary.
            alpha2:    Test-to-grid filter-width ratio squared (default 4.0).
        """
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.alpha2 = alpha2
        self.last_C = 0.0
        # Pre-compute the static box-filter denominator Σ V over the one-ring.
        n_int = mesh_data["n_interior_faces"]
        vol = geo_data["element_volumes"]
        own = mesh_data["owners"][:n_int]
        nei = mesh_data["neighbours"][:n_int]
        denom = vol.copy()
        np.add.at(denom, own, vol[nei])
        np.add.at(denom, nei, vol[own])
        self._own, self._nei, self._vol, self._denom = own, nei, vol, denom

    def _box_filter(self, f):
        """Apply a volume-weighted one-ring box filter to a cell field.

        For each cell, the filtered value is the volume-weighted average
        of the cell and its face neighbours (the one-ring).

        Args:
            f: Cell-centred field ``(n_elements, ...)``.

        Returns:
            Filtered field with the same shape as *f*.
        """
        vol = self._vol
        num = (vol.reshape((-1,) + (1,) * (f.ndim - 1)) * f).copy()
        np.add.at(
            num, self._own, (vol[self._nei].reshape((-1,) + (1,) * (f.ndim - 1))) * f[self._nei]
        )
        np.add.at(
            num, self._nei, (vol[self._own].reshape((-1,) + (1,) * (f.ndim - 1))) * f[self._own]
        )
        return num / self._denom.reshape((-1,) + (1,) * (f.ndim - 1))

    def get_filter_info(self):
        """Return filter-width statistics, model name, and last C value.

        Returns:
            Dict with keys ``model``, ``Cs`` (sqrt of last C), and
            ``filter_width_min/max/mean``.
        """
        delta = self.geo_data["element_volumes"] ** (1.0 / 3.0)
        return {
            "model": "dynamicSmagorinsky",
            "Cs": float(np.sqrt(max(self.last_C, 0.0))),
            "filter_width_min": float(np.min(delta)),
            "filter_width_max": float(np.max(delta)),
            "filter_width_mean": float(np.mean(delta)),
        }

    def compute_nut(self, U, mesh_data=None, geo_data=None):
        """Compute the subgrid-scale turbulent viscosity (dynamic procedure).

        Uses the Germano identity with a volume-weighted one-ring box test
        filter and a global (volume-averaged) Lilly least-squares contraction.

        Args:
            U:        Velocity field ``(n_elements + n_boundary, 3)``.
            mesh_data: Optional override mesh dictionary.
            geo_data:  Optional override geometry dictionary.

        Returns:
            Per-element eddy viscosity ``(n_elements,)``, clipped ≥ 0.
        """
        mesh_data = mesh_data or self.mesh_data
        geo_data = geo_data or self.geo_data
        n_elem = mesh_data["n_elements"]

        g = _velocity_gradient_tensor(U, mesh_data, geo_data)
        S, SS = _strain_rate(g)
        Smag = np.sqrt(2.0 * SS)  # |S| = sqrt(2 S_ij S_ij)

        u = U[:n_elem]
        delta2 = _compute_filter_width(geo_data["element_volumes"], mesh_data, geo_data) ** 2

        # Leonard stresses L_ij = (u_i u_j)~ − u~_i u~_j
        uu = u[:, :, None] * u[:, None, :]  # (n,3,3)
        u_f = self._box_filter(u)
        L = self._box_filter(uu) - u_f[:, :, None] * u_f[:, None, :]

        # M_ij = 2Δ²[ (|S| S_ij)~ − α² |S~| S~_ij ]
        SmagS = Smag[:, None, None] * S
        Sf = self._box_filter(S)
        Smag_f = np.sqrt(2.0 * np.sum(Sf * Sf, axis=(1, 2)))
        M = (
            2.0
            * delta2[:, None, None]
            * (self._box_filter(SmagS) - self.alpha2 * Smag_f[:, None, None] * Sf)
        )

        LM = np.sum(L * M, axis=(1, 2))
        MM = np.sum(M * M, axis=(1, 2))
        C = float(np.sum(LM) / (np.sum(MM) + 1e-30))  # global Lilly average
        C = max(C, 0.0)  # clip backscatter for stability
        self.last_C = C

        nut = C * delta2 * Smag
        return np.nan_to_num(np.clip(nut, 0.0, None), nan=0.0, posinf=0.0, neginf=0.0)


def create_model(config, mesh_data, geo_data):
    """Build the configured LES model instance from a :class:`TurbulenceConfig`.

    Factory function that dispatches to the appropriate model class based
    on the ``config.model`` string (case-insensitive).

    Recognised models:
    - ``"none"``, ``"iles"``, ``"dns"`` → ``None`` (no subgrid model).
    - ``"smagorinsky"`` → :class:`Smagorinsky` (or
      :class:`DynamicSmagorinsky` if ``config.dynamic is True``).
    - ``"dynamicsmagorinsky"`` / ``"dynamic_smagorinsky"`` →
      :class:`DynamicSmagorinsky`.
    - ``"wale"`` → :class:`WALE`.
    - ``"sigma"`` → :class:`Sigma`.

    Args:
        config:   :class:`TurbulenceConfig` instance (may be ``None``).
        mesh_data: Mesh dictionary.
        geo_data:  Geometry dictionary.

    Returns:
        An LES model instance with a ``compute_nut(U)`` interface,
        or ``None`` for no-model (ILES/DNS).

    Raises:
        ValueError: If the model name is not recognised.
    """
    if config is None:
        return None
    name = str(config.model).lower()
    if name in ("none", "", "iles", "dns"):
        return None
    if name == "wale":
        return WALE(mesh_data, geo_data, Cw=getattr(config, "Cs", 0.325) or 0.325)
    if name == "sigma":
        return Sigma(mesh_data, geo_data, Csigma=getattr(config, "Cs", 1.35) or 1.35)
    if name in ("dynamicsmagorinsky", "dynamic_smagorinsky"):
        return DynamicSmagorinsky(mesh_data, geo_data)
    if name == "smagorinsky":
        if getattr(config, "dynamic", False):
            return DynamicSmagorinsky(mesh_data, geo_data)
        return Smagorinsky(mesh_data, geo_data, Cs=config.Cs, dynamic=False)
    raise ValueError(f"Unknown turbulence model '{config.model}'")
