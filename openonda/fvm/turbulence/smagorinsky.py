"""Smagorinsky turbulence models for the FVM solver.

Two distinct formulations are provided:

``SmagorinskyModel``
    The familiar incompressible ``C_s`` form,
    ``eddy_viscosity = (C_s Delta)^2 |S|``.

``EquilibriumSmagorinskyModel``
    The algebraic-equilibrium implementation. It computes the
    SGS kinetic energy from ``C_k``, ``C_e`` and the complete symmetric
    velocity gradient before evaluating ``eddy_viscosity = C_k Delta sqrt(k)``.  The
    defaults are ``C_k=0.094`` and ``C_e=1.048``.

Both use the ``cubeRootVol`` filter ``Delta = V^(1/3)``, computed once per
mesh, and both need a three-dimensional mesh.  Neither has a wall
treatment: there is no van Driest damping and no wall model, so next to a
wall the eddy viscosity follows the resolved strain of the wall-adjacent
cell.

"""

from __future__ import annotations

import numpy as np

from ..fields import gradients

EQUILIBRIUM_CK = 0.094
"""Default algebraic-equilibrium SGS kinetic-energy coefficient."""

EQUILIBRIUM_CE = 1.048
"""Default algebraic-equilibrium SGS dissipation coefficient."""


def _filter_width(mesh_data: dict, geo_data: dict) -> np.ndarray:
    """Return the ``cubeRootVol`` filter width ``V^(1/3)`` of every cell.

    A subgrid model is a three-dimensional closure; a one-cell-thick mesh
    with ``empty`` patches has no resolved spanwise scale and is rejected.

    Raises:
        ValueError: If the mesh has an ``empty`` patch.
    """
    empty = [
        boundary.get("name")
        for boundary in mesh_data.get("boundary", [])
        if boundary.get("velocity_type") == "empty"
    ]
    if empty:
        raise ValueError(
            f"A subgrid-scale model needs a three-dimensional mesh; patches {empty} are empty"
        )
    return np.asarray(geo_data["cell_volume"], dtype=np.float64) ** (1.0 / 3.0)


class SmagorinskyModel:
    """Classical Smagorinsky LES eddy-viscosity model.

    Computes the subgrid-scale turbulent viscosity as
    ``ν_t = (C_s Δ)² |S|`` where ``Δ`` is the filter width (``V^{1/3}``),
    ``|S| = √(2 S_ij S_ij)`` is the strain-rate magnitude, and *C_s* is the
    Smagorinsky coefficient (typical value 0.17).

    The filter width is fixed at construction (:func:`_filter_width`).

    Args:
        mesh_data: Mesh dictionary.
        geo_data:  Geometry dictionary (needs ``cell_volume``).
        smagorinsky_coefficient:        Smagorinsky coefficient (default 0.17).
    """

    def __init__(
        self,
        mesh_data: dict[str, object],
        geo_data: dict[str, object],
        smagorinsky_coefficient: float = 0.17,
    ) -> None:
        """Bind the classical Smagorinsky model to FVM mesh geometry.

        Parameters
        ----------
        mesh_data, geo_data : dict
            Native topology and geometry. Cell volumes define the filter
            width; the supplied mappings are not mutated.
        smagorinsky_coefficient : float, default=0.17
            Dimensionless ``C_s`` coefficient.

        Raises
        ------
        ValueError
            If ``smagorinsky_coefficient`` is non-finite or negative, or the
            mesh has an ``empty`` patch.
        """
        if not np.isfinite(smagorinsky_coefficient) or smagorinsky_coefficient < 0.0:
            raise ValueError(
                "Smagorinsky coefficient smagorinsky_coefficient must be finite and non-negative"
            )
        self.smagorinsky_coefficient = smagorinsky_coefficient
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.filter_width = _filter_width(mesh_data, geo_data)

    def get_filter_info(self) -> dict[str, object]:
        """Return a dictionary of filter parameters for logging and diagnostics.

        Returns:
            Dict with keys ``model``, ``smagorinsky_coefficient``, ``min_filter_width``,
            ``max_filter_width``, ``mean_filter_width``.
        """
        delta = self.filter_width
        return {
            "model": "smagorinsky",
            "smagorinsky_coefficient": self.smagorinsky_coefficient,
            "min_filter_width": float(np.min(delta)),
            "max_filter_width": float(np.max(delta)),
            "mean_filter_width": float(np.mean(delta)),
        }

    def compute_eddy_viscosity(self, velocity: np.ndarray) -> np.ndarray:
        """Compute eddy viscosity per cell.

        Args:
            velocity: Velocity field with shape ``(n_cells + n_boundary, 3)``.

        Returns:
            Eddy viscosity with shape ``(n_cells,)``.
        """
        mesh_data = self.mesh_data
        geo_data = self.geo_data
        n_cells = mesh_data["n_cells"]

        # Compute velocity gradient on elements (returns gradients for interior+boundary elements)
        _grad_fn = gradients._resolve_gradient_fn(geo_data)
        velocity_gradient = _grad_fn(velocity, mesh_data, geo_data)

        # We are only interested in interior element gradients (first n_cells)
        interior_velocity_gradient = velocity_gradient[:n_cells]

        # Interior velocity-gradient shape: (n_cells, 3, 3) for vector fields.
        if (
            interior_velocity_gradient.ndim == 3
            and interior_velocity_gradient.shape[1] == 3
            and interior_velocity_gradient.shape[2] == 3
        ):
            # Compute strain-rate tensor S_ij = 0.5*(dU_i/dx_j + dU_j/dx_i)
            strain_rate = 0.5 * (
                interior_velocity_gradient + np.transpose(interior_velocity_gradient, (0, 2, 1))
            )
            # S_ij S_ij
            strain_rate_squared = np.sum(strain_rate * strain_rate, axis=(1, 2))
            strain_rate_magnitude = np.sqrt(2.0 * strain_rate_squared)
        else:
            raise ValueError(
                "Velocity gradient has shape "
                f"{interior_velocity_gradient.shape}; expected ({n_cells}, 3, 3)"
            )

        eddy_viscosity = (
            self.smagorinsky_coefficient * self.filter_width
        ) ** 2 * strain_rate_magnitude

        if not np.all(np.isfinite(eddy_viscosity)) or np.any(eddy_viscosity < 0.0):
            raise FloatingPointError("Smagorinsky model produced invalid eddy viscosity")
        return eddy_viscosity


class EquilibriumSmagorinskyModel:
    r"""Algebraic-equilibrium Smagorinsky LES model.

    This implements the equilibrium equations:

    .. math::

       a &= C_e / \Delta, \\
       b &= \tfrac{2}{3}\,\mathrm{tr}(D), \\
       c &= 2 C_k \Delta\,[\mathrm{dev}(D):D], \\
       k &= \left(\frac{-b + \sqrt{b^2 + 4ac}}{2a}\right)^2, \\
       \nu_t &= C_k \Delta \sqrt{k},

    where ``D`` is the symmetric velocity-gradient tensor and ``Delta`` is the
    filter width ``V^(1/3)`` of :func:`_filter_width`.  For exactly incompressible flow this reduces to the
    classical model with ``C_s^2 = C_k sqrt(C_k/C_e)``.  Keeping the full
    algebraic expression remains valid when the discrete velocity field has a
    small non-zero divergence.

    Parameters
    ----------
    mesh_data:
        Native finite-volume mesh dictionary.
    geo_data:
        Geometry dictionary containing at least ``cell_volume`` and the
        selected gradient reconstruction data.
    subgrid_kinetic_energy_coefficient:
        SGS kinetic-energy coefficient. Default ``0.094``.
    subgrid_dissipation_coefficient:
        SGS dissipation coefficient. Default ``1.048``.

    Examples
    --------
    Configure a case with ``Numerics(turbulence=TurbulenceConfig.equilibrium_smagorinsky())``.

    Construct the model directly for diagnostics::

        model = EquilibriumSmagorinskyModel(mesh_data, geo_data)
        eddy_viscosity = model.compute_eddy_viscosity(velocity)

    """

    def __init__(
        self,
        mesh_data: dict,
        geo_data: dict,
        subgrid_kinetic_energy_coefficient: float = EQUILIBRIUM_CK,
        subgrid_dissipation_coefficient: float = EQUILIBRIUM_CE,
    ) -> None:
        """Bind the equilibrium Smagorinsky closure to FVM mesh geometry.

        Parameters
        ----------
        mesh_data, geo_data : dict
            Native topology and geometry read by the closure; cell volumes
            determine the filter width.
        subgrid_kinetic_energy_coefficient : float, default=EQUILIBRIUM_CK
            Dimensionless equilibrium ``C_K`` coefficient.
        subgrid_dissipation_coefficient : float, default=EQUILIBRIUM_CE
            Positive dimensionless equilibrium ``C_E`` coefficient.

        Raises
        ------
        ValueError
            If either coefficient is non-finite, ``C_K`` is negative,
            ``C_E`` is non-positive, or the mesh has an ``empty`` patch.
        """
        if (
            not np.isfinite(subgrid_kinetic_energy_coefficient)
            or subgrid_kinetic_energy_coefficient < 0.0
        ):
            raise ValueError(
                "Equilibrium Smagorinsky subgrid_kinetic_energy_coefficient must be finite and non-negative"
            )
        if (
            not np.isfinite(subgrid_dissipation_coefficient)
            or subgrid_dissipation_coefficient <= 0.0
        ):
            raise ValueError(
                "Equilibrium Smagorinsky subgrid_dissipation_coefficient must be finite and positive"
            )
        self.subgrid_kinetic_energy_coefficient = float(subgrid_kinetic_energy_coefficient)
        self.subgrid_dissipation_coefficient = float(subgrid_dissipation_coefficient)
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.filter_width = _filter_width(mesh_data, geo_data)

    @property
    def equivalent_smagorinsky_coefficient(self) -> float:
        """Return the classical incompressible ``C_s`` for these coefficients."""
        return (
            self.subgrid_kinetic_energy_coefficient**0.75
            / self.subgrid_dissipation_coefficient**0.25
        )

    def _compute_sgs_state(self, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(k_sgs, delta)`` with bounded tensor working storage."""
        mesh = self.mesh_data
        geometry = self.geo_data
        n_cells = int(mesh["n_cells"])
        grad_fn = gradients._resolve_gradient_fn(geometry)
        velocity_gradient = np.asarray(grad_fn(velocity, mesh, geometry), dtype=np.float64)
        if velocity_gradient.shape[1:] != (3, 3) or velocity_gradient.shape[0] < n_cells:
            raise ValueError(
                f"Velocity gradient has shape {velocity_gradient.shape}; expected at least ({n_cells}, 3, 3)"
            )
        if not np.all(np.isfinite(velocity_gradient)):
            raise FloatingPointError("Smagorinsky velocity gradient contains non-finite values")

        delta = self.filter_width
        k_sgs = np.empty(n_cells, dtype=np.float64)
        chunk_size = 100_000
        for start in range(0, n_cells, chunk_size):
            stop = min(start + chunk_size, n_cells)
            gradient = velocity_gradient[start:stop]
            strain = 0.5 * (gradient + np.transpose(gradient, (0, 2, 1)))
            trace = np.einsum("fii->f", strain)
            # dev(D):D = D:D - tr(D)^2/3.  This avoids a second full tensor
            # copy and bounds all remaining temporaries to one cell block.
            contraction = np.sum(strain * strain, axis=(1, 2)) - trace * trace / 3.0
            delta_block = delta[start:stop]
            a = self.subgrid_dissipation_coefficient / delta_block
            b = (2.0 / 3.0) * trace
            c = 2.0 * self.subgrid_kinetic_energy_coefficient * delta_block * contraction
            discriminant = np.maximum(b * b + 4.0 * a * c, 0.0)
            sqrt_k = (-b + np.sqrt(discriminant)) / (2.0 * a)
            k_sgs[start:stop] = sqrt_k * sqrt_k

        if not np.all(np.isfinite(k_sgs)) or np.any(k_sgs < 0.0):
            raise FloatingPointError("Equilibrium Smagorinsky produced invalid SGS energy")
        return k_sgs, delta

    def compute_eddy_viscosity(self, velocity: np.ndarray) -> np.ndarray:
        """Compute algebraic-equilibrium SGS kinematic viscosity ``eddy_viscosity``."""
        k_sgs, delta = self._compute_sgs_state(velocity)
        eddy_viscosity = self.subgrid_kinetic_energy_coefficient * delta * np.sqrt(k_sgs)
        if not np.all(np.isfinite(eddy_viscosity)) or np.any(eddy_viscosity < 0.0):
            raise FloatingPointError("Equilibrium Smagorinsky produced invalid eddy viscosity")
        return eddy_viscosity

    def get_filter_info(self) -> dict[str, float | str]:
        """Return model coefficients and ``cubeRootVol`` filter statistics."""
        delta = self.filter_width
        return {
            "model": "equilibrium_smagorinsky",
            "subgrid_kinetic_energy_coefficient": self.subgrid_kinetic_energy_coefficient,
            "subgrid_dissipation_coefficient": self.subgrid_dissipation_coefficient,
            "smagorinsky_coefficient": self.equivalent_smagorinsky_coefficient,
            "min_filter_width": float(np.min(delta)),
            "max_filter_width": float(np.max(delta)),
            "mean_filter_width": float(np.mean(delta)),
        }
