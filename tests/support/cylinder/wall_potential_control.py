"""Process-local impermeable-cylinder potential for autonomous CFD controls.

The measured Gaussian wall-normal velocity sets the harmonic coefficients at
each induction call. The correction has zero curl, divergence and circulation.
It enters both particle advection and FVM boundary evaluation. This prototype
belongs to verification until autonomous force results establish its effect.
"""

from contextlib import contextmanager

import numpy as np
import taichi as ti

from .audit_circular_wall_potential import fit_circular_wall_potential


def harmonic_velocity_gradient(points, coefficients, radius=0.5):
    """Evaluate the exterior harmonic U and J[velocity,coordinate] efficiently."""
    z = points[:, 0] + 1j * points[:, 1]
    if np.any(np.abs(z) < 0.49):
        raise ValueError("Exterior cylinder correction received a solid-interior target")
    first = np.zeros(len(z), dtype=complex)
    second = np.zeros(len(z), dtype=complex)
    for mode, (cosine, sine) in enumerate(coefficients.reshape(-1, 2), start=1):
        amplitude = cosine + 1j * sine
        derivative = -amplitude * (radius / z) ** (mode + 1)
        first += derivative
        second -= (mode + 1) * derivative / z
    velocity = np.column_stack((first.real, -first.imag, np.zeros(len(z))))
    gradient = np.zeros((len(z), 3, 3))
    gradient[:, 0, 0], gradient[:, 1, 1] = second.real, -second.real
    gradient[:, 0, 1] = gradient[:, 1, 0] = -second.imag
    return velocity, gradient


@contextmanager
def impermeable_cylinder_induction(owner, measurements, *, modes=16):
    particles = owner.vpm_solver
    induction = particles.induction
    original = induction.evaluate_targets
    points = np.asarray(owner.fvm_solver.get_boundary_face_centre_coordinates("cylinder"))
    normals = np.asarray(owner.fvm_solver.get_boundary_face_normal("cylinder"))
    area = np.asarray(owner.fvm_solver.get_boundary_face_area("cylinder"))
    wall_position = ti.Vector.field(3, dtype=induction._dtype, shape=len(points))
    wall_velocity = ti.Vector.field(3, dtype=induction._dtype, shape=len(points))
    wall_position.from_numpy(
        points.astype(np.float64 if induction._dtype == ti.f64 else np.float32)
    )
    measurements.update(
        calls=0,
        modes=modes,
        maximum_coefficient=0.0,
        maximum_remaining_normal_rms=0.0,
        maximum_original_normal_rms=0.0,
    )

    def evaluate(**values):
        original(**values)
        original(
            target_position=wall_position,
            source_position=values["source_position"],
            source_vortex_strength=values["source_vortex_strength"],
            source_core_radius=values["source_core_radius"],
            target_velocity=wall_velocity,
            target_velocity_gradient=None,
            target_count=len(points),
            source_count=values["source_count"],
            include_freestream=True,
            background_velocity=owner.freestream_velocity,
        )
        wall_values = wall_velocity.to_numpy()
        coefficient, fit = fit_circular_wall_potential(
            points, normals, area, wall_values, modes=modes
        )
        count = values["target_count"]
        target = particles.physics._download_vector_field(values["target_position"], count)
        velocity, gradient = harmonic_velocity_gradient(target, coefficient)
        if values["target_velocity"] is not None:
            field = particles.physics._download_vector_field(values["target_velocity"], count)
            particles.physics._upload_vector_array(
                field + velocity, values["target_velocity"], count
            )
        if values["target_velocity_gradient"] is not None:
            field = values["target_velocity_gradient"].to_numpy()
            field[:count] += gradient.transpose(0, 2, 1)
            values["target_velocity_gradient"].from_numpy(field)
        measurements["calls"] += 1
        measurements["maximum_coefficient"] = max(
            measurements["maximum_coefficient"], fit["maximum_coefficient"]
        )
        measurements["maximum_remaining_normal_rms"] = max(
            measurements["maximum_remaining_normal_rms"], fit["remaining_wall_normal_rms"]
        )
        measurements["maximum_original_normal_rms"] = max(
            measurements["maximum_original_normal_rms"],
            float(
                np.sqrt(np.average(np.einsum("fi,fi->f", wall_values, normals) ** 2, weights=area))
            ),
        )

    induction.evaluate_targets = evaluate
    try:
        yield
    finally:
        induction.evaluate_targets = original
