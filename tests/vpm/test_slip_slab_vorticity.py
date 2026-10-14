"""Stored particle vorticity includes every slip-slab image."""

import numpy as np
import pytest

from openonda import vpm
from openonda.vpm.kernels.registry import kernel_definition

SPACING = 0.04  # m
PLANES = (-0.5 * SPACING, 0.5 * SPACING)  # one particle layer


def _explicit_vorticity(targets, position, strength, sigma, kernel, periods=200):
    """Kernel vorticity of the particles and all their images up to ``periods`` periods away."""
    zeta = kernel_definition(kernel).zeta
    width = PLANES[1] - PLANES[0]
    reflected = position.copy()
    reflected[:, 2] = 2.0 * PLANES[0] - reflected[:, 2]
    total = np.zeros_like(targets)
    for shift in range(-periods, periods + 1):
        for source, gamma in ((position, strength), (reflected, strength * [-1.0, -1.0, 1.0])):
            image = source.copy()
            image[:, 2] += 2.0 * shift * width
            distance = np.linalg.norm(targets[:, None, :] - image[None, :, :], axis=-1)
            total += (zeta(distance / sigma) / sigma**3) @ gamma
    return total


@pytest.mark.parametrize("kernel", ["gaussian", "winckelmans"])
def test_one_layer_vorticity_includes_every_image(tmp_path, kernel):
    rng = np.random.default_rng(5)
    count = 30
    position = np.column_stack(
        (rng.uniform(-0.2, 0.2, count), rng.uniform(-0.2, 0.2, count), rng.uniform(*PLANES, count))
    )
    strength = rng.normal(size=(count, 3)) * SPACING**3
    sigma = SPACING
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            name="test",
            numerics=vpm.Numerics(
                time_step_size=0.01,
                compute_device="cpu",
                precision="f64",
                particle_kernel=kernel,
                max_n_particles=64,
                domain_bounds=(-0.4, 0.4, -0.4, 0.4, *PLANES),
                induction=vpm.SlipSlabInduction(
                    vpm.DirectInduction(), z_min=PLANES[0], z_max=PLANES[1]
                ),
                viscous=vpm.ViscousConfig.inviscid(),
            ),
        )
    )
    try:
        solver.add_vortex_particles(
            position,
            np.zeros_like(position),
            strength,
            np.full(count, sigma),
            np.full(count, sigma**3),
            kinematic_viscosity=np.zeros(count),
        )
        solver.physics.compute_vorticities(solver.particles)
        stored = solver.particle_vorticity
    finally:
        solver.close()

    expected = _explicit_vorticity(position, position, strength, sigma, kernel)
    # Images beyond the kernel reach carry at most 1e-4 of each core's circulation
    scale = np.abs(expected).max()
    np.testing.assert_allclose(stored, expected, rtol=0.0, atol=2e-3 * scale)
    physical_only = _explicit_vorticity(position, position, strength, sigma, kernel, periods=0)
    assert np.abs(physical_only - expected).max() > 0.1 * scale
