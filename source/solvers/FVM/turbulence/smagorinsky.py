"""Smagorinsky turbulence model for FVM solver.

Implements classical Smagorinsky model:
    nut = (C_s * Delta)^2 * |S|
where |S| = sqrt(2 S_ij S_ij) and Delta = (V)^(1/3) (volume-based filter width).

For now the dynamic option is placeholder (not fully implemented).
"""

import numpy as np

from ..fields import gradients


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
    for boundary in mesh_data.get("boundary", []):
        if boundary.get("type") == "empty" or boundary.get("bc_type") == "empty":
            return True
    return False


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
        if boundary.get("type") == "empty" or boundary.get("bc_type") == "empty":
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
        dynamic:   Placeholder for dynamic procedure (not yet implemented).
    """

    def __init__(self, mesh_data, geo_data, Cs=0.17, dynamic=False):
        self.Cs = Cs
        self.dynamic = dynamic
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
            "model": "Smagorinsky (Dynamic)" if self.dynamic else "Smagorinsky",
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
        mesh_data = mesh_data or self.mesh_data
        geo_data = geo_data or self.geo_data
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
            # Fallback: compute magnitude of gradient components
            S_mag = np.linalg.norm(grad_U_int.reshape((n_elements, -1)), axis=1)

        # Filter width Delta
        vol = geo_data["element_volumes"]
        delta = _compute_filter_width(vol, self.mesh_data, geo_data)

        Cs = self.Cs
        nut = (Cs * delta) ** 2 * S_mag

        # Ensure non-negative and finite
        nut = np.nan_to_num(nut, nan=0.0, posinf=0.0, neginf=0.0)
        nut[nut < 0] = 0.0

        # Clip excessively large nut to a reasonable multiple of molecular nu (if available)
        return nut
