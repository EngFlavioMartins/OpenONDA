"""Native internal-face traces for the fully 3D cropped-FVM experiment.

These are diagnostic observations of the reference discretization, not new
coupler boundary conditions. In particular, a gradient inferred from a face
value need not equal the derivative used by a diffusive flux on a skew face.
"""

from __future__ import annotations

import numpy as np

from source.solvers.fvm.assemble.diffusion import assemble_diffusion_term_interior
from source.solvers.fvm.solve.simple_solver import (
    _compute_pressure_face_conductance,
    _pressure_interior_flux_scalar,
)


def tangential(values, normal):
    """Project vectors onto the plane normal to the given unit normals."""
    return values - np.sum(values * normal, axis=1)[:, None] * normal


class NativeFaceTrace:
    """Observe selected full-mesh interior faces in the cropped orientation.

    Velocity gradients follow the FVM convention (cell, derivative, component).
    Pressure gradients must come from the reference solver's configured scheme.
    Unit diffusivity and a unit scalar pressure-to-velocity coefficient isolate
    geometry and field reconstruction from viscosity and momentum diagonals.
    The pressure trace is therefore for a scalar-diagonal reference operator.
    """

    def __init__(self, mesh, geometry, faces, signs):
        faces = np.asarray(faces, dtype=int)
        signs = np.asarray(signs, dtype=float)
        if faces.ndim != 1 or signs.shape != faces.shape or not len(faces):
            raise ValueError("Nonempty one-dimensional faces and matching signs are required")
        if np.any(faces < 0) or np.any(faces >= mesh["n_interior_faces"]):
            raise ValueError("Native traces require full-mesh interior faces")
        if not np.all(np.isin(signs, [-1, 1])):
            raise ValueError("Face orientation signs must be +1 or -1")
        self.signs = signs.copy()
        self.mesh = {
            "n_cells": mesh["n_cells"],
            "n_faces": len(faces),
            "n_interior_faces": len(faces),
            "owners": np.asarray(mesh["owners"])[faces],
            "neighbours": np.asarray(mesh["neighbours"])[faces],
        }
        self.geometry = {
            key: np.asarray(geometry[key])[faces]
            for key in ("face_area_vector", "cell_connection_vector", "face_interpolation_weight")
        }
        self.area = np.linalg.norm(self.geometry["face_area_vector"], axis=1)
        self.normal = self.geometry["face_area_vector"] * (signs / self.area)[:, None]
        self.owner = np.where(signs > 0, self.mesh["owners"], self.mesh["neighbours"])
        displacement = geometry["face_centre"][faces] - geometry["cell_centre"][self.owner]
        self.distance = np.sum(displacement * self.normal, axis=1)
        if np.any(self.distance <= 0):
            raise ValueError("The cut face must lie outward from its cropped owner")
        self.tangential_displacement = tangential(displacement, self.normal)
        self.ones = np.ones(mesh["n_cells"])
        self.zeros = np.zeros((mesh["n_cells"], 3))
        self.pressure_conductance = _compute_pressure_face_conductance(
            self.mesh, self.geometry, self.ones
        )

    def interpolate(self, values):
        """Use the native neighbour weight, without an additional Taylor fit."""
        weight = self.geometry["face_interpolation_weight"]
        weight = weight.reshape((-1,) + (1,) * (np.asarray(values).ndim - 1))
        return weight * values[self.mesh["neighbours"]] + (1 - weight) * values[self.mesh["owners"]]

    def evaluate(self, velocity, velocity_gradient, pressure, pressure_gradient):
        """Return native flux derivatives and face-value-compatible increments.

        The native-flux velocity derivative is minus the assembled diffusion
        flux divided by face area, evaluated with unit viscosity. Pressure uses
        its own Rhie--Chow interior kernel with H/A=0, D=1 and no time-flux term.
        It intentionally retains that kernel's existing geometric regularizer.
        """
        normal_velocity_gradient = np.column_stack([
            -assemble_diffusion_term_interior(
                velocity[:, component], velocity_gradient[:, :, component],
                1.0, self.mesh, self.geometry,
            )["flux_tf"] * self.signs / self.area
            for component in range(3)
        ])
        pressure_flux = np.empty(len(self.area))
        _pressure_interior_flux_scalar(
            self.mesh["owners"], self.mesh["neighbours"],
            self.geometry["face_interpolation_weight"], self.geometry["face_area_vector"],
            self.geometry["cell_connection_vector"], self.ones, self.zeros,
            np.ascontiguousarray(pressure_gradient), np.ascontiguousarray(pressure),
            self.pressure_conductance, np.empty(0), pressure_flux,
        )
        face_velocity = self.interpolate(velocity)
        face_pressure = self.interpolate(pressure)
        return {
            "native_face_velocity": face_velocity,
            "native_face_pressure": face_pressure,
            "native_flux_velocity_normal_gradient": normal_velocity_gradient,
            "native_flux_tangential_gradient": tangential(normal_velocity_gradient, self.normal),
            "native_flux_pressure_normal_gradient": -pressure_flux * self.signs / self.area,
            "native_value_tangential_gradient": tangential(
                (face_velocity - velocity[self.owner]) / self.distance[:, None], self.normal
            ),
            "native_value_pressure_normal_gradient": (
                (face_pressure - pressure[self.owner]) / self.distance
            ),
        }
