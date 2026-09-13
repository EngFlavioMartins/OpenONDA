"""Observe a sampled velocity with the actual FVM interior-face operators.

This diagnostic consumes complete cell stencils on both sides of a selected
interior face. It does not prescribe a cropped-domain boundary condition.
"""

from __future__ import annotations

import numpy as np

from source.solvers.fvm.fields.gradients import compute_gauss_gradient
from studies.coupler_accuracy.native_face_trace_3d import NativeFaceTrace


class InteriorFaceVelocitySampler:
    """Identify the point samples needed by native Gauss/diffusion traces.

    Only uncoupled, serial, fully 3D meshes are supported. Both cells of every
    selected face must have entirely interior gradient stencils; rejecting a
    wall-dependent stencil prevents missing boundary observations from being
    silently replaced by the zero placeholders used outside the stencil.
    """

    def __init__(self, mesh, geometry, faces, signs):
        if any(p.get("velocity_type", p.get("type")) == "empty" for p in mesh["boundary"]):
            raise ValueError("A fully 3D mesh without empty patches is required")
        if np.any(np.asarray(mesh.get("boundary_neighbour_cell", [-1])) >= 0):
            raise ValueError("Coupled boundary stencils are outside this diagnostic")
        parallel = mesh.get("_parallel_context")
        if parallel is not None and parallel.is_partitioned:
            raise ValueError("A complete serial mesh is required")
        self.trace = NativeFaceTrace(mesh, geometry, faces, signs)
        self.mesh, self.geometry = mesh, geometry
        self.faces = np.asarray(faces, dtype=int).copy()
        self.gradient_cells = np.unique(np.r_[self.trace.mesh["owners"], self.trace.mesh["neighbours"]])
        selected = np.zeros(mesh["n_cells"], dtype=bool)
        selected[self.gradient_cells] = True
        ni = mesh["n_interior_faces"]
        owner, neighbour = np.asarray(mesh["owners"]), np.asarray(mesh["neighbours"])
        if np.any(selected[owner[ni:]]):
            raise ValueError("Selected face gradients require physical boundary observations")
        self.gradient_faces = np.flatnonzero(selected[owner[:ni]] | selected[neighbour[:ni]])
        self.sample_cells = np.unique(np.r_[owner[self.gradient_faces], neighbour[self.gradient_faces]])

    def evaluate(self, velocity):
        """Apply the production Gauss gradient and native diffusion face trace.

        ``velocity`` contains (sample_cells, 3) values, with the same cell-value
        interpretation as the FVM. Returned gradients use derivative first.
        No continuous velocity derivatives enter this calculation.
        """
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (len(self.sample_cells), 3) or not np.isfinite(velocity).all():
            raise ValueError("Finite three-component values are required at every sample cell")
        n = self.mesh["n_cells"]
        nt = n+self.mesh["n_faces"]-self.mesh["n_interior_faces"]
        padded = np.zeros((nt, 3))
        padded[self.sample_cells] = velocity
        gradient = compute_gauss_gradient(padded, self.mesh, self.geometry)
        observed = self.trace.evaluate(padded, gradient, np.zeros(nt), np.zeros((nt, 3)))
        return {"face_velocity": observed["native_face_velocity"],
                "normal_gradient": observed["native_flux_velocity_normal_gradient"],
                "tangential_gradient": observed["native_flux_tangential_gradient"],
                "value_tangential_gradient": observed["native_value_tangential_gradient"],
                "cell_gradient": gradient[self.gradient_cells]}
