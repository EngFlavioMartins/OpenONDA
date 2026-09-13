"""Scoped outflow convection experiment; no production default is changed."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from source.solvers.fvm.assemble import convection
from source.solvers.fvm.schemes.boundaries import BoundaryStrategy


@contextmanager
def mixed_outflow_linear_upwind():
    """Use native owner-gradient extrapolation on mixed linearUpwind outflow.

    The advective face flux and the diffusion/ghost boundary are unchanged.
    Convection's upwind owner value is implicit and its gradient correction is
    deferred, as for an internal linearUpwind face. Inflow retains the original
    mixed-boundary reconstruction. This scope must not overlap another solver
    experiment in the same Python process; the original function is restored
    even if the experiment raises.
    """
    original = convection.assemble_convection_term_boundary

    def boundary(scalar_field, advective_face_flux, boundary_patch, mesh_data,
                 geo_data=None, scheme="upwind", scalar_field_gradient=None, *,
                 include_total_flux=True, component=None):
        result = original(
            scalar_field, advective_face_flux, boundary_patch, mesh_data, geo_data,
            scheme, scalar_field_gradient, include_total_flux=include_total_flux,
            component=component,
        )
        if (convection._convection_boundary_strategy(boundary_patch)
                is not BoundaryStrategy.NORMAL_VALUE_TANGENTIAL_GRADIENT
                or str(scheme).lower() != "linearupwind"):
            return result
        if scalar_field_gradient is None:
            raise ValueError("Experimental mixed linearUpwind outflow needs a cell gradient")
        gradient = np.asarray(scalar_field_gradient)
        if gradient.ndim == 3 and gradient.shape[2] == 1:
            gradient = gradient[:, :, 0]
        faces = result["face_indices"]
        flux = np.asarray(advective_face_flux)[faces]
        outward = flux > 0
        owners = mesh_data["owners"][faces]
        displacement = geo_data["face_centre"][faces] - geo_data["cell_centre"][owners]
        correction = np.sum(gradient[owners] * displacement, axis=1)
        result["flux_cf"][outward] = flux[outward]
        result["flux_vf"][outward] = (flux * correction)[outward]
        if "flux_tf" in result:
            result["flux_tf"][outward] = (
                flux * (scalar_field[owners] + correction)
            )[outward]
        return result

    convection.assemble_convection_term_boundary = boundary
    try:
        yield
    finally:
        convection.assemble_convection_term_boundary = original
