"""Scoped live experiment: observe VPM velocity through native FVM stencils."""

from contextlib import contextmanager

import numpy as np


@contextmanager
def sampled_face_boundary(sampler, *, mode, callback=None, solver_class=None):
    """Replace only the mixed boundary target query during a study.

    The sampler contains geometry, topology and sample indices, not evolving
    reference fields. All velocity data comes from the queried particle solver.
    ``native_gradient`` retains the original point velocity and changes only its
    derivative; ``native_both`` also uses the native interpolated face velocity.
    """
    if mode not in ("native_gradient", "native_both"):
        raise ValueError("Choose native_gradient or native_both")
    if solver_class is None:
        from source.solvers.vpm.core.solver import VPMSolver

        solver_class = VPMSolver
    method = "compute_velocity_and_tangential_normal_gradient_at_points"
    original = getattr(solver_class, method)
    target = sampler.geometry["face_centre"][sampler.faces]
    sample_position = sampler.geometry["cell_centre"][sampler.sample_cells]

    def evaluate(self, evaluation_position, normal, *, particle_spacing):
        position, normals = np.asarray(evaluation_position), np.asarray(normal)
        if position.shape != target.shape or not np.allclose(position, target, rtol=0, atol=1e-12):
            raise ValueError("The sampled-boundary experiment received a different face set")
        if normals.shape != target.shape or np.any(~np.isfinite(normals)):
            raise ValueError("Finite nonzero face normals are required")
        length = np.linalg.norm(normals, axis=1)
        if np.any(length <= 0):
            raise ValueError("Finite nonzero face normals are required")
        if not np.allclose(normals/length[:, None], sampler.trace.normal, rtol=0, atol=1e-12):
            raise ValueError("The sampled-boundary experiment received different face orientations")
        sampled = np.asarray(self.compute_velocity_at_points(sample_position, include_freestream=True,
                                                             include_body=True, zone_mask=None), dtype=float)
        observed = sampler.evaluate(sampled)
        if mode == "native_gradient":
            velocity, _ = original(self, evaluation_position, normal, particle_spacing=particle_spacing)
        else:
            velocity = observed["face_velocity"]
        if callback is not None:
            callback(self, sampled, observed, np.asarray(velocity))
        return velocity, observed["tangential_gradient"]

    setattr(solver_class, method, evaluate)
    try:
        yield
    finally:
        setattr(solver_class, method, original)
