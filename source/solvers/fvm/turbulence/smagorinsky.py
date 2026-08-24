"""Smagorinsky turbulence models for the FVM solver.

Two deliberately distinct formulations are provided:

``Smagorinsky``
    The familiar incompressible ``C_s`` form,
    ``eddy_viscosity = (C_s Delta)^2 |S|``.

``EquilibriumSmagorinsky``
    The algebraic-equilibrium implementation. It computes the
    SGS kinetic energy from ``C_k``, ``C_e`` and the complete symmetric
    velocity gradient before evaluating ``eddy_viscosity = C_k Delta sqrt(k)``.  The
    defaults are ``C_k=0.094`` and ``C_e=1.048``.

Both use the ``cubeRootVol`` filter in 3-D: ``Delta = V^(1/3)``.

"""

from __future__ import annotations

import numpy as np

from ..fields import gradients

EQUILIBRIUM_CK = 0.094
"""Default algebraic-equilibrium SGS kinetic-energy coefficient."""

EQUILIBRIUM_CE = 1.048
"""Default algebraic-equilibrium SGS dissipation coefficient."""


def _detect_2d_mesh(mesh_data: dict) -> bool:
    """Check whether the mesh is pseudo-2D (has ``empty`` boundary patches).

    A pseudo-2D mesh is a single-layer extruded mesh where the front/back
    faces use the ``empty`` boundary type.  This affects the filter-width
    computation.

    Args:
        mesh_data: Mesh dictionary (must contain a ``"boundary"`` list).

    Returns:
        ``True`` if any boundary patch has type ``"empty"``.
    """
    return any(
        boundary.get("velocity_type") == "empty" for boundary in mesh_data.get("boundary", [])
    )


def _compute_empty_bc_thickness(mesh_data: dict, geo_data: dict) -> float:
    """Infer the pseudo-2D mesh thickness from an ``empty`` boundary patch.

    The thickness is computed as twice the normal distance from the first
    empty-patch face centre to its owner cell centre.

    Args:
        mesh_data: Mesh dictionary.
        geo_data:  Geometry dictionary (needs ``face_centre``,
                   ``cell_centre``).

    Returns:
        Mesh thickness (float); ``1.0`` if no empty patch found.
    """
    for boundary in mesh_data.get("boundary", []):
        if boundary.get("velocity_type") == "empty":
            start = boundary["start_face"]
            own = mesh_data["owners"][start]
            face_c = geo_data["face_centre"][start]
            cell_c = geo_data["cell_centre"][own]
            return float(2.0 * np.linalg.norm(face_c - cell_c))
    return 1.0


def _compute_filter_width(cell_volume: np.ndarray, mesh_data: dict, geo_data: dict) -> np.ndarray:
    """Compute the per-cell LES filter width Δ.

    For 3D meshes: ``Δ = V^{1/3}`` (cube root of cell volume).
    For pseudo-2D meshes: ``Δ = √(V / t)`` where *t* is the out-of-plane
    thickness from :func:`_compute_empty_bc_thickness`.

    Args:
        cell_volume:      Cell volume array ``(n_elements,)``.
        mesh_data: Mesh dictionary (used for 2-D detection).
        geo_data:  Geometry dictionary (used for thickness inference).

    Returns:
        Per-cell filter width array ``(n_elements,)``.
    """
    if _detect_2d_mesh(mesh_data):
        thickness = _compute_empty_bc_thickness(mesh_data, geo_data)
        return np.sqrt(cell_volume / thickness)
    return cell_volume ** (1.0 / 3.0)


def _symmetric_velocity_gradient(velocity, mesh_data: dict, geo_data: dict) -> np.ndarray:
    """Return the symmetric velocity-gradient tensor for the interior cells.

    The native gradient layout is ``grad[c, j, i] = d(U_i)/d(x_j)``.  A
    transpose only changes the storage convention, not the symmetric tensor,
    so the expression below is independent of the gradient storage convention.
    """
    n_cells = mesh_data["n_cells"]
    grad_fn = gradients._resolve_gradient_fn(geo_data)
    velocity_gradient = np.asarray(
        grad_fn(velocity, mesh_data, geo_data)[:n_cells], dtype=np.float64
    )
    if velocity_gradient.shape != (n_cells, 3, 3):
        raise ValueError(
            f"Velocity gradient has shape {velocity_gradient.shape}; expected ({n_cells}, 3, 3)"
        )
    if not np.all(np.isfinite(velocity_gradient)):
        raise FloatingPointError("Smagorinsky velocity gradient contains non-finite values")
    return 0.5 * (velocity_gradient + np.transpose(velocity_gradient, (0, 2, 1)))


class Smagorinsky:
    """Classical Smagorinsky LES eddy-viscosity model.

    Computes the subgrid-scale turbulent viscosity as
    ``ν_t = (C_s Δ)² |S|`` where ``Δ = V^{1/3}`` is the filter width,
    ``|S| = √(2 S_ij S_ij)`` is the strain-rate magnitude, and *C_s* is the
    Smagorinsky coefficient (typical value 0.17).

    Handles both 3D and pseudo-2D (extruded) meshes via
    :func:`_compute_filter_width`.

    Args:
        mesh_data: Mesh dictionary.
        geo_data:  Geometry dictionary (needs ``cell_volume``).
        smagorinsky_coefficient:        Smagorinsky coefficient (default 0.17).
    """

    def __init__(self, mesh_data, geo_data, smagorinsky_coefficient=0.17):
        if not np.isfinite(smagorinsky_coefficient) or smagorinsky_coefficient < 0.0:
            raise ValueError(
                "Smagorinsky coefficient smagorinsky_coefficient must be finite and non-negative"
            )
        self.smagorinsky_coefficient = smagorinsky_coefficient
        self.mesh_data = mesh_data
        self.geo_data = geo_data

    def get_filter_info(self):
        """Return a dictionary of filter parameters for logging and diagnostics.

        Returns:
            Dict with keys ``model``, ``smagorinsky_coefficient``, ``min_filter_width``,
            ``max_filter_width``, ``mean_filter_width``.
        """
        cell_volume = self.geo_data["cell_volume"]
        delta = cell_volume ** (1.0 / 3.0)
        return {
            "model": "Smagorinsky",
            "smagorinsky_coefficient": self.smagorinsky_coefficient,
            "min_filter_width": float(np.min(delta)),
            "max_filter_width": float(np.max(delta)),
            "mean_filter_width": float(np.mean(delta)),
        }

    def compute_eddy_viscosity(self, velocity, mesh_data=None, geo_data=None):
        """Compute eddy viscosity per cell.

        Args:
            velocity: Velocity field with shape ``(n_cells + n_boundary, 3)``.

        Returns:
            Eddy viscosity with shape ``(n_cells,)``.
        """
        mesh_data = self.mesh_data if mesh_data is None else mesh_data
        geo_data = self.geo_data if geo_data is None else geo_data
        n_cells = mesh_data["n_cells"]

        # Compute velocity gradient on elements (returns gradients for interior+boundary elements)
        _grad_fn = gradients._resolve_gradient_fn(geo_data)
        velocity_gradient = _grad_fn(velocity, mesh_data, geo_data)

        # We are only interested in interior element gradients (first n_elements)
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

        # Filter width Delta
        cell_volume = geo_data["cell_volume"]
        filter_width = _compute_filter_width(cell_volume, mesh_data, geo_data)

        smagorinsky_coefficient = self.smagorinsky_coefficient
        eddy_viscosity = (smagorinsky_coefficient * filter_width) ** 2 * strain_rate_magnitude

        if not np.all(np.isfinite(eddy_viscosity)) or np.any(eddy_viscosity < 0.0):
            raise FloatingPointError("Smagorinsky model produced invalid eddy viscosity")
        return eddy_viscosity


class EquilibriumSmagorinsky:
    r"""Algebraic-equilibrium Smagorinsky LES model.

    This implements the equilibrium equations:

    .. math::

       a &= C_e / \Delta, \\
       b &= \tfrac{2}{3}\,\mathrm{tr}(D), \\
       c &= 2 C_k \Delta\,[\mathrm{dev}(D):D], \\
       k &= \left(\frac{-b + \sqrt{b^2 + 4ac}}{2a}\right)^2, \\
       \nu_t &= C_k \Delta \sqrt{k},

    where ``D`` is the symmetric velocity-gradient tensor and the cube-root
    cell-volume filter is
    ``Delta = V^(1/3)``.  For exactly incompressible flow this reduces to the
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
    Configure a solver through the public factory::

        setup = FVMSetup(
            turbulence=TurbulenceConfig.equilibrium_smagorinsky()
        )

    Construct the model directly for diagnostics::

        model = EquilibriumSmagorinsky(mesh_data, geo_data)
        eddy_viscosity = model.compute_eddy_viscosity(velocity)
        subgrid_kinetic_energy = model.compute_sgs_kinetic_energy(velocity)

    """

    def __init__(
        self,
        mesh_data: dict,
        geo_data: dict,
        subgrid_kinetic_energy_coefficient: float = EQUILIBRIUM_CK,
        subgrid_dissipation_coefficient: float = EQUILIBRIUM_CE,
    ) -> None:
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

    @property
    def equivalent_smagorinsky_coefficient(self) -> float:
        """Return the classical incompressible ``C_s`` for these coefficients."""
        return (
            self.subgrid_kinetic_energy_coefficient**0.75
            / self.subgrid_dissipation_coefficient**0.25
        )

    def _resolve_inputs(
        self,
        velocity,
        mesh_data: dict | None,
        geo_data: dict | None,
    ) -> tuple[dict, dict, np.ndarray, np.ndarray]:
        mesh = self.mesh_data if mesh_data is None else mesh_data
        geometry = self.geo_data if geo_data is None else geo_data
        strain = _symmetric_velocity_gradient(velocity, mesh, geometry)
        delta = _compute_filter_width(geometry["cell_volume"], mesh, geometry)
        return mesh, geometry, strain, delta

    def _compute_sgs_state(
        self,
        velocity,
        mesh_data: dict | None,
        geo_data: dict | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(k_sgs, delta)`` with bounded tensor working storage."""
        mesh = self.mesh_data if mesh_data is None else mesh_data
        geometry = self.geo_data if geo_data is None else geo_data
        n_cells = int(mesh["n_cells"])
        grad_fn = gradients._resolve_gradient_fn(geometry)
        velocity_gradient = np.asarray(grad_fn(velocity, mesh, geometry), dtype=np.float64)
        if velocity_gradient.shape[1:] != (3, 3) or velocity_gradient.shape[0] < n_cells:
            raise ValueError(
                f"Velocity gradient has shape {velocity_gradient.shape}; expected at least ({n_cells}, 3, 3)"
            )
        if not np.all(np.isfinite(velocity_gradient)):
            raise FloatingPointError("Smagorinsky velocity gradient contains non-finite values")

        delta = _compute_filter_width(geometry["cell_volume"], mesh, geometry)
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

    def compute_sgs_kinetic_energy(
        self,
        velocity,
        mesh_data: dict | None = None,
        geo_data: dict | None = None,
    ) -> np.ndarray:
        """Compute algebraic SGS kinetic energy ``k`` per cell.

        Parameters
        ----------
        velocity:
            Velocity values for interior and boundary-ghost cells, shaped
            ``(n_cells_with_ghosts, 3)``.
        mesh_data, geo_data:
            Optional mesh/geometry overrides, matching the common turbulence
            model interface.

        Returns
        -------
        numpy.ndarray
            Non-negative SGS kinetic energy with one value per interior cell.
        """
        k_sgs, _ = self._compute_sgs_state(velocity, mesh_data, geo_data)
        return k_sgs

    def compute_eddy_viscosity(
        self,
        velocity,
        mesh_data: dict | None = None,
        geo_data: dict | None = None,
    ) -> np.ndarray:
        """Compute algebraic-equilibrium SGS kinematic viscosity ``eddy_viscosity``."""
        k_sgs, delta = self._compute_sgs_state(velocity, mesh_data, geo_data)
        eddy_viscosity = self.subgrid_kinetic_energy_coefficient * delta * np.sqrt(k_sgs)
        if not np.all(np.isfinite(eddy_viscosity)) or np.any(eddy_viscosity < 0.0):
            raise FloatingPointError("Equilibrium Smagorinsky produced invalid eddy viscosity")
        return eddy_viscosity

    def get_filter_info(self) -> dict[str, float | str]:
        """Return model coefficients and ``cubeRootVol`` filter statistics."""
        delta = _compute_filter_width(self.geo_data["cell_volume"], self.mesh_data, self.geo_data)
        return {
            "model": "EquilibriumSmagorinsky",
            "subgrid_kinetic_energy_coefficient": self.subgrid_kinetic_energy_coefficient,
            "subgrid_dissipation_coefficient": self.subgrid_dissipation_coefficient,
            "smagorinsky_coefficient": self.equivalent_smagorinsky_coefficient,
            "min_filter_width": float(np.min(delta)),
            "max_filter_width": float(np.max(delta)),
            "mean_filter_width": float(np.mean(delta)),
        }
