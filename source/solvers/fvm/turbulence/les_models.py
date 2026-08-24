"""Subgrid-scale LES eddy-viscosity models for the FVM solver.

All models expose the same interface as :class:`..smagorinsky.Smagorinsky`
(``compute_eddy_viscosity(velocity, mesh_data, geo_data) -> eddy_viscosity[n_cells]`` and
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


def _validated_eddy_viscosity(eddy_viscosity: np.ndarray, model: str) -> np.ndarray:
    """Return a valid non-negative eddy viscosity or fail the run."""
    if not np.all(np.isfinite(eddy_viscosity)):
        raise FloatingPointError(f"{model} produced non-finite eddy viscosity")
    if np.any(eddy_viscosity < 0.0):
        raise FloatingPointError(f"{model} produced negative eddy viscosity")
    return eddy_viscosity


def _velocity_gradient_tensor(velocity, mesh_data, geo_data):
    """Compute the velocity-gradient tensor for interior cells.

    Returns ``g[c, i, j] = ∂u_i/∂x_j`` for each cell *c*.

    The underlying gradient function (from
    :func:`..fields.gradients._resolve_gradient_fn`) returns
    ``grad[c, k, i] = ∂U_i/∂x_k``, so we transpose the last two axes.

    Args:
        velocity: Velocity field ``(n_elements + n_boundary, 3)``.
        mesh_data: Mesh dictionary.
        geo_data:  Geometry dictionary.

    Returns:
        Velocity-gradient tensor ``(n_elements, 3, 3)``.
    """
    n_cells = mesh_data["n_cells"]
    grad_fn = gradients._resolve_gradient_fn(geo_data)
    grad = grad_fn(velocity, mesh_data, geo_data)[:n_cells]  # (n, 3, 3): [c,k,i]
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
        wale_coefficient:        WALE model coefficient (default 0.325).
    """

    def __init__(self, mesh_data, geo_data, wale_coefficient=0.325):
        if not np.isfinite(wale_coefficient) or wale_coefficient < 0.0:
            raise ValueError("WALE coefficient wale_coefficient must be finite and non-negative")
        self.wale_coefficient = wale_coefficient
        self.mesh_data = mesh_data
        self.geo_data = geo_data

    def get_filter_info(self):
        """Return filter-width statistics and model name.

        Returns:
            Dict with keys ``model``, ``smagorinsky_coefficient`` (carries *C_w*), and
            ``min_filter_width/max/mean``.
        """
        delta = self.geo_data["cell_volume"] ** (1.0 / 3.0)
        return {
            "model": "WALE",
            "wale_coefficient": self.wale_coefficient,
            "min_filter_width": float(np.min(delta)),
            "max_filter_width": float(np.max(delta)),
            "mean_filter_width": float(np.mean(delta)),
        }

    def compute_eddy_viscosity(self, velocity, mesh_data=None, geo_data=None):
        """Compute the subgrid-scale turbulent viscosity (WALE model).

        Args:
            velocity: Velocity field ``(n_cells + n_boundary, 3)``.
            mesh_data: Optional override mesh dictionary.
            geo_data:  Optional override geometry dictionary.

        Returns:
            Per-cell eddy viscosity ``(n_cells,)``, clipped ≥ 0.
        """
        mesh_data = self.mesh_data if mesh_data is None else mesh_data
        geo_data = self.geo_data if geo_data is None else geo_data

        g = _velocity_gradient_tensor(velocity, mesh_data, geo_data)
        op = _wale_operator(g)

        delta = _compute_filter_width(geo_data["cell_volume"], mesh_data, geo_data)
        eddy_viscosity = (self.wale_coefficient * delta) ** 2 * op
        return _validated_eddy_viscosity(eddy_viscosity, "WALE")


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
        sigma_coefficient:    sigma model coefficient (default 1.35).
    """

    def __init__(self, mesh_data, geo_data, sigma_coefficient=1.35):
        if not np.isfinite(sigma_coefficient) or sigma_coefficient < 0.0:
            raise ValueError("sigma coefficient must be finite and non-negative")
        self.sigma_coefficient = sigma_coefficient
        self.mesh_data = mesh_data
        self.geo_data = geo_data

    def get_filter_info(self):
        """Return filter-width statistics and model name.

        Returns:
            Dict with keys ``model``, ``smagorinsky_coefficient`` (carries *C_σ*), and
            ``min_filter_width/max/mean``.
        """
        delta = self.geo_data["cell_volume"] ** (1.0 / 3.0)
        return {
            "model": "sigma",
            "sigma_coefficient": self.sigma_coefficient,
            "min_filter_width": float(np.min(delta)),
            "max_filter_width": float(np.max(delta)),
            "mean_filter_width": float(np.mean(delta)),
        }

    def compute_eddy_viscosity(self, velocity, mesh_data=None, geo_data=None):
        """Compute the subgrid-scale turbulent viscosity (sigma model).

        Args:
            velocity: Velocity field ``(n_elements + n_boundary, 3)``.
            mesh_data: Optional override mesh dictionary.
            geo_data:  Optional override geometry dictionary.

        Returns:
            Per-element eddy viscosity ``(n_elements,)``, clipped ≥ 0.
        """
        mesh_data = self.mesh_data if mesh_data is None else mesh_data
        geo_data = self.geo_data if geo_data is None else geo_data

        g = _velocity_gradient_tensor(velocity, mesh_data, geo_data)
        # Singular values σ1 ≥ σ2 ≥ σ3 ≥ 0 of the velocity-gradient tensor.
        sv = np.linalg.svd(g, compute_uv=False)  # (n, 3), descending
        s1, s2, s3 = sv[:, 0], sv[:, 1], sv[:, 2]

        eps = 1e-30
        d_sigma = s3 * (s1 - s2) * (s2 - s3) / (s1 * s1 + eps)

        delta = _compute_filter_width(geo_data["cell_volume"], mesh_data, geo_data)
        eddy_viscosity = (self.sigma_coefficient * delta) ** 2 * d_sigma
        return _validated_eddy_viscosity(eddy_viscosity, "sigma")


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

    def __init__(self, mesh_data, geo_data, test_filter_width_ratio_squared=4.0):
        """Initialise the dynamic Smagorinsky model.

        Pre-computes the static volume-weighted box-filter denominator
        (one-ring neighbour sum) from the mesh topology.

        Args:
            mesh_data: Mesh dictionary.
            geo_data:  Geometry dictionary.
            test_filter_width_ratio_squared:    Test-to-grid filter-width ratio squared (default 4.0).
        """
        if (
            not np.isfinite(test_filter_width_ratio_squared)
            or test_filter_width_ratio_squared <= 1.0
        ):
            raise ValueError(
                "Dynamic Smagorinsky test_filter_width_ratio_squared must be finite and greater than one"
            )
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.test_filter_width_ratio_squared = test_filter_width_ratio_squared
        self.last_smagorinsky_coefficient_squared = 0.0
        self._filter = CellBoxFilter(mesh_data, geo_data)

    def _box_filter(self, f):
        """Volume-weighted one-ring box filter — the dynamic model's test filter.

        Delegates to the shared :class:`CellBoxFilter` so the coupler's blending
        zone relaxation and this model use the same operator.
        """
        return self._filter(f)

    def get_filter_info(self):
        """Return filter-width statistics, model name, and last C value.

        Returns:
            Dict with keys ``model``, ``smagorinsky_coefficient`` (sqrt of last C), and
            ``min_filter_width/max/mean``.
        """
        delta = self.geo_data["cell_volume"] ** (1.0 / 3.0)
        return {
            "model": "dynamicSmagorinsky",
            "smagorinsky_coefficient": float(
                np.sqrt(max(self.last_smagorinsky_coefficient_squared, 0.0))
            ),
            "min_filter_width": float(np.min(delta)),
            "max_filter_width": float(np.max(delta)),
            "mean_filter_width": float(np.mean(delta)),
        }

    def compute_eddy_viscosity(self, velocity, mesh_data=None, geo_data=None):
        """Compute the subgrid-scale turbulent viscosity (dynamic procedure).

        Uses the Germano identity with a volume-weighted one-ring box test
        filter and a global (volume-averaged) Lilly least-squares contraction.

        Args:
            velocity: Velocity field ``(n_elements + n_boundary, 3)``.
            mesh_data: Optional override mesh dictionary.
            geo_data:  Optional override geometry dictionary.

        Returns:
            Per-element eddy viscosity ``(n_elements,)``, clipped ≥ 0.
        """
        mesh_data = self.mesh_data if mesh_data is None else mesh_data
        geo_data = self.geo_data if geo_data is None else geo_data
        n_cells = mesh_data["n_cells"]

        velocity_gradient = _velocity_gradient_tensor(velocity, mesh_data, geo_data)
        strain_rate, strain_rate_contraction = _strain_rate(velocity_gradient)
        strain_rate_magnitude = np.sqrt(2.0 * strain_rate_contraction)

        interior_velocity = velocity[:n_cells]
        filter_width_squared = (
            _compute_filter_width(geo_data["cell_volume"], mesh_data, geo_data) ** 2
        )

        # Leonard stresses L_ij = (u_i u_j)~ − u~_i u~_j
        velocity_outer_product = interior_velocity[:, :, None] * interior_velocity[:, None, :]
        filtered_velocity = self._box_filter(interior_velocity)
        leonard_stress = self._box_filter(velocity_outer_product) - (
            filtered_velocity[:, :, None] * filtered_velocity[:, None, :]
        )

        # M_ij = 2Δ²[ (|S| S_ij)~ − α² |S~| S~_ij ]
        magnitude_weighted_strain_rate = strain_rate_magnitude[:, None, None] * strain_rate
        filtered_strain_rate = self._box_filter(strain_rate)
        filtered_strain_rate_magnitude = np.sqrt(
            2.0 * np.sum(filtered_strain_rate * filtered_strain_rate, axis=(1, 2))
        )
        model_tensor = (
            2.0
            * filter_width_squared[:, None, None]
            * (
                self._box_filter(magnitude_weighted_strain_rate)
                - self.test_filter_width_ratio_squared
                * filtered_strain_rate_magnitude[:, None, None]
                * filtered_strain_rate
            )
        )

        leonard_model_contraction = np.sum(leonard_stress * model_tensor, axis=(1, 2))
        model_self_contraction = np.sum(model_tensor * model_tensor, axis=(1, 2))
        coefficient_squared = float(
            np.sum(leonard_model_contraction) / (np.sum(model_self_contraction) + 1e-30)
        )
        if not np.isfinite(coefficient_squared):
            raise FloatingPointError("Dynamic Smagorinsky produced a non-finite coefficient")
        coefficient_squared = max(coefficient_squared, 0.0)  # clip backscatter for stability
        self.last_smagorinsky_coefficient_squared = coefficient_squared

        eddy_viscosity = coefficient_squared * filter_width_squared * strain_rate_magnitude
        return _validated_eddy_viscosity(eddy_viscosity, "Dynamic Smagorinsky")


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
        An LES model instance with a ``compute_eddy_viscosity(velocity)`` interface,
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
        return WALE(mesh_data, geo_data, wale_coefficient=config.wale_coefficient)
    if name == "sigma":
        return Sigma(mesh_data, geo_data, sigma_coefficient=config.sigma_coefficient)
    if name in ("dynamicsmagorinsky", "dynamic_smagorinsky"):
        return DynamicSmagorinsky(mesh_data, geo_data)
    if name in ("equilibriumsmagorinsky", "equilibrium_smagorinsky"):
        return EquilibriumSmagorinsky(
            mesh_data,
            geo_data,
            subgrid_kinetic_energy_coefficient=config.subgrid_kinetic_energy_coefficient,
            subgrid_dissipation_coefficient=config.subgrid_dissipation_coefficient,
        )
    if name == "smagorinsky":
        if getattr(config, "dynamic", False):
            return DynamicSmagorinsky(mesh_data, geo_data)
        return Smagorinsky(
            mesh_data, geo_data, smagorinsky_coefficient=config.smagorinsky_coefficient
        )
    raise ValueError(f"Unknown turbulence model '{config.model}'")
