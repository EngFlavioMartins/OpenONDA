"""Smagorinsky turbulence models for the FVM solver.

Two deliberately distinct formulations are provided:

``Smagorinsky``
    The familiar incompressible ``C_s`` form,
    ``nu_t = (C_s Delta)^2 |S|``.

``OpenFOAMSmagorinsky``
    OpenCFD OpenFOAM's algebraic-equilibrium implementation.  It computes the
    SGS kinetic energy from ``C_k``, ``C_e`` and the complete symmetric
    velocity gradient before evaluating ``nu_t = C_k Delta sqrt(k)``.  The
    OpenFOAM defaults are ``C_k=0.094`` and ``C_e=1.048``.

Both use the ``cubeRootVol`` filter in 3-D: ``Delta = V^(1/3)``.

"""

from __future__ import annotations

import numpy as np

from ..fields import gradients

OPENFOAM_CK = 0.094
"""Default ``Ck`` used by OpenFOAM's ``LESModels::Smagorinsky``."""

OPENFOAM_CE = 1.048
"""Default ``Ce`` used by OpenFOAM's base LES model."""


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
    return any(boundary.get("bc_type_U") == "empty" for boundary in mesh_data.get("boundary", []))


def _compute_empty_bc_thickness(mesh_data: dict, geo_data: dict) -> float:
    """Infer the pseudo-2D mesh thickness from an ``empty`` boundary patch.

    The thickness is computed as twice the normal distance from the first
    empty-patch face centroid to its owner element centroid.

    Args:
        mesh_data: Mesh dictionary.
        geo_data:  Geometry dictionary (needs ``face_centroids``,
                   ``element_centroids``).

    Returns:
        Mesh thickness (float); ``1.0`` if no empty patch found.
    """
    for boundary in mesh_data.get("boundary", []):
        if boundary.get("bc_type_U") == "empty":
            start = boundary["startFace"]
            own = mesh_data["owners"][start]
            face_c = geo_data["face_centroids"][start]
            elem_c = geo_data["element_centroids"][own]
            return float(2.0 * np.linalg.norm(face_c - elem_c))
    return 1.0


def _compute_filter_width(vol: np.ndarray, mesh_data: dict, geo_data: dict) -> np.ndarray:
    """Compute the per-cell LES filter width Δ.

    For 3D meshes: ``Δ = V^{1/3}`` (cube root of cell volume).
    For pseudo-2D meshes: ``Δ = √(V / t)`` where *t* is the out-of-plane
    thickness from :func:`_compute_empty_bc_thickness`.

    Args:
        vol:      Cell volume array ``(n_elements,)``.
        mesh_data: Mesh dictionary (used for 2-D detection).
        geo_data:  Geometry dictionary (used for thickness inference).

    Returns:
        Per-cell filter width array ``(n_elements,)``.
    """
    if _detect_2d_mesh(mesh_data):
        thickness = _compute_empty_bc_thickness(mesh_data, geo_data)
        return np.sqrt(vol / thickness)
    return vol ** (1.0 / 3.0)


def _symmetric_velocity_gradient(U, mesh_data: dict, geo_data: dict) -> np.ndarray:
    """Return ``D = symm(grad(U))`` for the interior cells.

    The native gradient layout is ``grad[c, j, i] = d(U_i)/d(x_j)``.  A
    transpose only changes the storage convention, not the symmetric tensor,
    so the expression below is the same ``symm(fvc::grad(U))`` used by
    OpenFOAM.
    """
    n_elements = mesh_data["n_elements"]
    grad_fn = gradients._resolve_gradient_fn(geo_data)
    grad_u = np.asarray(grad_fn(U, mesh_data, geo_data)[:n_elements], dtype=np.float64)
    if grad_u.shape != (n_elements, 3, 3):
        raise ValueError(
            f"Velocity gradient has shape {grad_u.shape}; expected ({n_elements}, 3, 3)"
        )
    if not np.all(np.isfinite(grad_u)):
        raise FloatingPointError("Smagorinsky velocity gradient contains non-finite values")
    return 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))


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
        geo_data:  Geometry dictionary (needs ``element_volumes``).
        Cs:        Smagorinsky coefficient (default 0.17).
    """

    def __init__(self, mesh_data, geo_data, Cs=0.17):
        if not np.isfinite(Cs) or Cs < 0.0:
            raise ValueError("Smagorinsky coefficient Cs must be finite and non-negative")
        self.Cs = Cs
        self.mesh_data = mesh_data
        self.geo_data = geo_data

    def get_filter_info(self):
        """Return a dictionary of filter parameters for logging and diagnostics.

        Returns:
            Dict with keys ``model``, ``Cs``, ``filter_width_min``,
            ``filter_width_max``, ``filter_width_mean``.
        """
        vol = self.geo_data["element_volumes"]
        delta = vol ** (1.0 / 3.0)
        return {
            "model": "Smagorinsky",
            "Cs": self.Cs,
            "filter_width_min": float(np.min(delta)),
            "filter_width_max": float(np.max(delta)),
            "filter_width_mean": float(np.mean(delta)),
        }

    def compute_nut(self, U, mesh_data=None, geo_data=None):
        """Compute turbulent viscosity (nut) per element.

        Args:
            U: Velocity field as (n_elems + n_boundary, 3)
        Returns:
            nut: numpy array (n_elements,)
        """
        mesh_data = self.mesh_data if mesh_data is None else mesh_data
        geo_data = self.geo_data if geo_data is None else geo_data
        n_elements = mesh_data["n_elements"]

        # Compute velocity gradient on elements (returns gradients for interior+boundary elements)
        _grad_fn = gradients._resolve_gradient_fn(geo_data)
        grad_U = _grad_fn(U, mesh_data, geo_data)

        # We are only interested in interior element gradients (first n_elements)
        grad_U_int = grad_U[:n_elements]

        # grad_U_int shape: (n_elements,3,3) for vector fields
        if grad_U_int.ndim == 3 and grad_U_int.shape[1] == 3 and grad_U_int.shape[2] == 3:
            # Compute strain-rate tensor S_ij = 0.5*(dU_i/dx_j + dU_j/dx_i)
            S = 0.5 * (grad_U_int + np.transpose(grad_U_int, (0, 2, 1)))
            # S_ij S_ij
            S_sq = np.sum(S * S, axis=(1, 2))
            S_mag = np.sqrt(2.0 * S_sq)
        else:
            raise ValueError(
                f"Velocity gradient has shape {grad_U_int.shape}; expected ({n_elements}, 3, 3)"
            )

        # Filter width Delta
        vol = geo_data["element_volumes"]
        delta = _compute_filter_width(vol, mesh_data, geo_data)

        Cs = self.Cs
        nut = (Cs * delta) ** 2 * S_mag

        if not np.all(np.isfinite(nut)) or np.any(nut < 0.0):
            raise FloatingPointError("Smagorinsky model produced invalid eddy viscosity")
        return nut


class OpenFOAMSmagorinsky:
    r"""OpenFOAM-compatible algebraic-equilibrium Smagorinsky LES model.

    This implements the equations in OpenCFD OpenFOAM's
    ``LESModels::Smagorinsky`` class:

    .. math::

       a &= C_e / \Delta, \\
       b &= \tfrac{2}{3}\,\mathrm{tr}(D), \\
       c &= 2 C_k \Delta\,[\mathrm{dev}(D):D], \\
       k &= \left(\frac{-b + \sqrt{b^2 + 4ac}}{2a}\right)^2, \\
       \nu_t &= C_k \Delta \sqrt{k},

    where ``D = symm(grad(U))`` and the ``cubeRootVol`` filter is
    ``Delta = V^(1/3)``.  For exactly incompressible flow this reduces to the
    classical model with ``C_s^2 = C_k sqrt(C_k/C_e)``.  Keeping the full
    algebraic expression reproduces OpenFOAM when the discrete velocity field
    has a small non-zero divergence.

    Parameters
    ----------
    mesh_data:
        Native finite-volume mesh dictionary.
    geo_data:
        Geometry dictionary containing at least ``element_volumes`` and the
        selected gradient reconstruction data.
    Ck:
        SGS kinetic-energy coefficient.  Default ``0.094`` matches OpenFOAM.
    Ce:
        SGS dissipation coefficient.  Default ``1.048`` matches OpenFOAM.

    Examples
    --------
    Configure a solver through the public factory::

        setup = FVMSetup(
            turbulence=TurbulenceConfig.openfoam_smagorinsky()
        )

    Construct the model directly for diagnostics::

        model = OpenFOAMSmagorinsky(mesh_data, geo_data)
        nut = model.compute_nut(U)
        k_sgs = model.compute_sgs_kinetic_energy(U)

    Notes
    -----
    OpenFOAM source lineage: ``src/TurbulenceModels/turbulenceModels/LES/``
    ``Smagorinsky/Smagorinsky.C`` (GPL-3.0-or-later).  This is an independent
    NumPy expression of the published equations, with the upstream model and
    default coefficients explicitly credited.
    """

    def __init__(
        self,
        mesh_data: dict,
        geo_data: dict,
        Ck: float = OPENFOAM_CK,
        Ce: float = OPENFOAM_CE,
    ) -> None:
        if not np.isfinite(Ck) or Ck < 0.0:
            raise ValueError("OpenFOAM Smagorinsky Ck must be finite and non-negative")
        if not np.isfinite(Ce) or Ce <= 0.0:
            raise ValueError("OpenFOAM Smagorinsky Ce must be finite and positive")
        self.Ck = float(Ck)
        self.Ce = float(Ce)
        self.mesh_data = mesh_data
        self.geo_data = geo_data

    @property
    def equivalent_Cs(self) -> float:
        """Return the classical incompressible ``C_s`` for these coefficients."""
        return self.Ck**0.75 / self.Ce**0.25

    def _resolve_inputs(
        self,
        U,
        mesh_data: dict | None,
        geo_data: dict | None,
    ) -> tuple[dict, dict, np.ndarray, np.ndarray]:
        mesh = self.mesh_data if mesh_data is None else mesh_data
        geometry = self.geo_data if geo_data is None else geo_data
        strain = _symmetric_velocity_gradient(U, mesh, geometry)
        delta = _compute_filter_width(geometry["element_volumes"], mesh, geometry)
        return mesh, geometry, strain, delta

    def compute_sgs_kinetic_energy(
        self,
        U,
        mesh_data: dict | None = None,
        geo_data: dict | None = None,
    ) -> np.ndarray:
        """Compute OpenFOAM's algebraic SGS kinetic energy ``k`` per cell.

        Parameters
        ----------
        U:
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
        _, _, strain, delta = self._resolve_inputs(U, mesh_data, geo_data)
        trace = np.trace(strain, axis1=1, axis2=2)
        deviatoric = strain.copy()
        diagonal = np.arange(3)
        deviatoric[:, diagonal, diagonal] -= trace[:, None] / 3.0
        contraction = np.sum(deviatoric * strain, axis=(1, 2))

        a = self.Ce / delta
        b = (2.0 / 3.0) * trace
        c = 2.0 * self.Ck * delta * contraction
        discriminant = np.maximum(b * b + 4.0 * a * c, 0.0)
        sqrt_k = (-b + np.sqrt(discriminant)) / (2.0 * a)
        k_sgs = np.square(sqrt_k)
        if not np.all(np.isfinite(k_sgs)) or np.any(k_sgs < 0.0):
            raise FloatingPointError("OpenFOAM Smagorinsky produced invalid SGS energy")
        return k_sgs

    def compute_nut(
        self,
        U,
        mesh_data: dict | None = None,
        geo_data: dict | None = None,
    ) -> np.ndarray:
        """Compute OpenFOAM-equivalent SGS kinematic viscosity ``nu_t``."""
        mesh = self.mesh_data if mesh_data is None else mesh_data
        geometry = self.geo_data if geo_data is None else geo_data
        delta = _compute_filter_width(geometry["element_volumes"], mesh, geometry)
        k_sgs = self.compute_sgs_kinetic_energy(U, mesh, geometry)
        nut = self.Ck * delta * np.sqrt(k_sgs)
        if not np.all(np.isfinite(nut)) or np.any(nut < 0.0):
            raise FloatingPointError("OpenFOAM Smagorinsky produced invalid eddy viscosity")
        return nut

    def get_filter_info(self) -> dict[str, float | str]:
        """Return model coefficients and ``cubeRootVol`` filter statistics."""
        delta = _compute_filter_width(
            self.geo_data["element_volumes"], self.mesh_data, self.geo_data
        )
        return {
            "model": "OpenFOAMSmagorinsky",
            "Ck": self.Ck,
            "Ce": self.Ce,
            "Cs": self.equivalent_Cs,
            "filter_width_min": float(np.min(delta)),
            "filter_width_max": float(np.max(delta)),
            "filter_width_mean": float(np.mean(delta)),
        }
