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
from ..fields.filters import CellBoxFilter
from .smagorinsky import EquilibriumSmagorinsky, Smagorinsky, _compute_filter_width


def _validated_eddy_viscosity(nut: np.ndarray, model: str) -> np.ndarray:
    """Return a valid non-negative eddy viscosity or fail the run."""
    if not np.all(np.isfinite(nut)):
        raise FloatingPointError(f"{model} produced non-finite eddy viscosity")
    if np.any(nut < 0.0):
        raise FloatingPointError(f"{model} produced negative eddy viscosity")
    return nut


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
    if not np.all(np.isfinite(grad)):
        raise FloatingPointError("LES velocity gradient contains non-finite values")
    return np.transpose(grad, (0, 2, 1))  # (n, 3, 3): [c,i,j] = ∂u_i/∂x_j


def _strain_rate(g):
    """Symmetric part S_ij and its magnitude-squared S:S = S_ij S_ij."""
    S = 0.5 * (g + np.transpose(g, (0, 2, 1)))
    return S, np.sum(S * S, axis=(1, 2))


def _wale_operator(g: np.ndarray) -> np.ndarray:
    """Evaluate the WALE velocity-gradient invariant."""
    _, strain_squared = _strain_rate(g)
    gradient_squared = np.einsum("cik,ckj->cij", g, g)
    trace = np.einsum("cii->c", gradient_squared)
    traceless_symmetric = 0.5 * (gradient_squared + np.transpose(gradient_squared, (0, 2, 1)))
    for axis in range(3):
        traceless_symmetric[:, axis, axis] -= trace / 3.0
    invariant = np.sum(traceless_symmetric * traceless_symmetric, axis=(1, 2))
    return invariant**1.5 / (strain_squared**2.5 + invariant**1.25 + 1e-30)


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
        if not np.isfinite(Cw) or Cw < 0.0:
            raise ValueError("WALE coefficient Cw must be finite and non-negative")
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
        mesh_data = self.mesh_data if mesh_data is None else mesh_data
        geo_data = self.geo_data if geo_data is None else geo_data

        g = _velocity_gradient_tensor(U, mesh_data, geo_data)
        op = _wale_operator(g)

        delta = _compute_filter_width(geo_data["element_volumes"], mesh_data, geo_data)
        nut = (self.Cw * delta) ** 2 * op
        return _validated_eddy_viscosity(nut, "WALE")


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
        if not np.isfinite(Csigma) or Csigma < 0.0:
            raise ValueError("sigma coefficient must be finite and non-negative")
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
        mesh_data = self.mesh_data if mesh_data is None else mesh_data
        geo_data = self.geo_data if geo_data is None else geo_data

        g = _velocity_gradient_tensor(U, mesh_data, geo_data)
        # Singular values σ1 ≥ σ2 ≥ σ3 ≥ 0 of the velocity-gradient tensor.
        sv = np.linalg.svd(g, compute_uv=False)  # (n, 3), descending
        s1, s2, s3 = sv[:, 0], sv[:, 1], sv[:, 2]

        eps = 1e-30
        d_sigma = s3 * (s1 - s2) * (s2 - s3) / (s1 * s1 + eps)

        delta = _compute_filter_width(geo_data["element_volumes"], mesh_data, geo_data)
        nut = (self.Csigma * delta) ** 2 * d_sigma
        return _validated_eddy_viscosity(nut, "sigma")


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
        if not np.isfinite(alpha2) or alpha2 <= 1.0:
            raise ValueError("Dynamic Smagorinsky alpha2 must be finite and greater than one")
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.alpha2 = alpha2
        self.last_C = 0.0
        self._filter = CellBoxFilter(mesh_data, geo_data)

    def _box_filter(self, f):
        """Volume-weighted one-ring box filter — the dynamic model's test filter.

        Thin alias for the shared :class:`CellBoxFilter` so the coupler's blending zone
        relaxation and this model cannot drift apart.
        """
        return self._filter(f)

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
        mesh_data = self.mesh_data if mesh_data is None else mesh_data
        geo_data = self.geo_data if geo_data is None else geo_data
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
        if not np.isfinite(C):
            raise FloatingPointError("Dynamic Smagorinsky produced a non-finite coefficient")
        C = max(C, 0.0)  # clip backscatter for stability
        self.last_C = C

        nut = C * delta2 * Smag
        return _validated_eddy_viscosity(nut, "Dynamic Smagorinsky")


def create_model(config, mesh_data, geo_data):
    """Build the configured LES model instance from a :class:`TurbulenceConfig`.

    Factory function that dispatches to the appropriate model class based
    on the ``config.model`` string (case-insensitive).

    Recognised models:
    - ``"none"``, ``"iles"``, ``"dns"`` → ``None`` (no subgrid model).
    - ``"smagorinsky"`` → :class:`Smagorinsky` (or
      :class:`DynamicSmagorinsky` if ``config.dynamic is True``).
    - ``"equilibriumsmagorinsky"`` → :class:`EquilibriumSmagorinsky`, using
      the algebraic-equilibrium equations and default coefficients.
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
        return WALE(mesh_data, geo_data, Cw=config.Cs)
    if name == "sigma":
        return Sigma(mesh_data, geo_data, Csigma=config.Cs)
    if name in ("dynamicsmagorinsky", "dynamic_smagorinsky"):
        return DynamicSmagorinsky(mesh_data, geo_data)
    if name in ("equilibriumsmagorinsky", "equilibrium_smagorinsky"):
        return EquilibriumSmagorinsky(mesh_data, geo_data, Ck=config.Ck, Ce=config.Ce)
    if name == "smagorinsky":
        if getattr(config, "dynamic", False):
            return DynamicSmagorinsky(mesh_data, geo_data)
        return Smagorinsky(mesh_data, geo_data, Cs=config.Cs)
    raise ValueError(f"Unknown turbulence model '{config.model}'")
