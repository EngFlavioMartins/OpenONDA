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
    """Return the velocity-gradient tensor g[c,i,j] = ∂u_i/∂x_j for interior cells.

    ``_resolve_gradient_fn`` returns grad[c,k,i] = ∂U_i/∂x_k, so the velocity
    gradient tensor is its transpose over the last two axes.
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
    """Wall-Adapting Local Eddy-viscosity (Nicoud & Ducros, 1999)."""

    def __init__(self, mesh_data, geo_data, Cw=0.325):
        self.Cw = Cw
        self.mesh_data = mesh_data
        self.geo_data = geo_data

    def get_filter_info(self):
        delta = self.geo_data["element_volumes"] ** (1.0 / 3.0)
        return {
            "model": "WALE",
            "Cs": self.Cw,
            "filter_width_min": float(np.min(delta)),
            "filter_width_max": float(np.max(delta)),
            "filter_width_mean": float(np.mean(delta)),
        }

    def compute_nut(self, U, mesh_data=None, geo_data=None):
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
    """sigma model (Nicoud, Toda, Cabrit, Bose & Lee, 2011)."""

    def __init__(self, mesh_data, geo_data, Csigma=1.35):
        self.Csigma = Csigma
        self.mesh_data = mesh_data
        self.geo_data = geo_data

    def get_filter_info(self):
        delta = self.geo_data["element_volumes"] ** (1.0 / 3.0)
        return {
            "model": "sigma",
            "Cs": self.Csigma,
            "filter_width_min": float(np.min(delta)),
            "filter_width_max": float(np.max(delta)),
            "filter_width_mean": float(np.mean(delta)),
        }

    def compute_nut(self, U, mesh_data=None, geo_data=None):
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
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.alpha2 = alpha2
        self.last_C = 0.0
        # Pre-compute the static box-filter denominator Σ V over the one-ring.
        n_elem = mesh_data["n_elements"]
        n_int = mesh_data["n_interior_faces"]
        vol = geo_data["element_volumes"]
        own = mesh_data["owners"][:n_int]
        nei = mesh_data["neighbours"][:n_int]
        denom = vol.copy()
        np.add.at(denom, own, vol[nei])
        np.add.at(denom, nei, vol[own])
        self._own, self._nei, self._vol, self._denom = own, nei, vol, denom

    def _box_filter(self, f):
        """Volume-weighted one-ring box filter of a cell field (n_elements, ...)."""
        vol = self._vol
        shape = (vol.shape[0],) + f.shape[1:]
        num = (vol.reshape((-1,) + (1,) * (f.ndim - 1)) * f).copy()
        np.add.at(num, self._own, (vol[self._nei].reshape((-1,) + (1,) * (f.ndim - 1))) * f[self._nei])
        np.add.at(num, self._nei, (vol[self._own].reshape((-1,) + (1,) * (f.ndim - 1))) * f[self._own])
        return num / self._denom.reshape((-1,) + (1,) * (f.ndim - 1))

    def get_filter_info(self):
        delta = self.geo_data["element_volumes"] ** (1.0 / 3.0)
        return {
            "model": "dynamicSmagorinsky",
            "Cs": float(np.sqrt(max(self.last_C, 0.0))),
            "filter_width_min": float(np.min(delta)),
            "filter_width_max": float(np.max(delta)),
            "filter_width_mean": float(np.mean(delta)),
        }

    def compute_nut(self, U, mesh_data=None, geo_data=None):
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
        M = 2.0 * delta2[:, None, None] * (
            self._box_filter(SmagS) - self.alpha2 * Smag_f[:, None, None] * Sf
        )

        LM = np.sum(L * M, axis=(1, 2))
        MM = np.sum(M * M, axis=(1, 2))
        C = float(np.sum(LM) / (np.sum(MM) + 1e-30))  # global Lilly average
        C = max(C, 0.0)  # clip backscatter for stability
        self.last_C = C

        nut = C * delta2 * Smag
        return np.nan_to_num(np.clip(nut, 0.0, None), nan=0.0, posinf=0.0, neginf=0.0)


def create_model(config, mesh_data, geo_data):
    """Build the configured LES model, or ``None`` for no-model (ILES / DNS).

    ``config`` is a ``TurbulenceConfig`` (``.model``, ``.Cs``, ``.dynamic``).
    Recognised models (case-insensitive): ``none``, ``smagorinsky``
    (``dynamic=True`` → dynamic Smagorinsky), ``wale``, ``sigma``,
    ``dynamicsmagorinsky``.
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
