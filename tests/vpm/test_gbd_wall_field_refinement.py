"""Near-wall M4 remap field error under genuine curved-body refinement."""

import numpy as np
import pytest

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin, _m4_prime_1d


class _WallHarness(_GridDiffusionMixin):
    pass


def _remapped_velocity(spacing: float, phase: float):
    wall = _WallHarness()
    wall._init_grid_diffusion()
    wall.configure_body_cylinder((-0.5, 0.5, -0.5, 0.5, -1.0, 1.0))
    # Stay wall-adjacent at every resolution without changing the circulation.
    source = np.array([[0.5 + 0.3 * spacing, 0.08, 0.07]])
    strength = np.array([[0.25, -0.18, 0.9]])
    origin = np.array([-1.0 + phase * spacing, -1.0, -0.5])
    shape = (int(np.ceil(2.1 / spacing)),) * 3
    fraction = (source[0] - origin) / spacing
    offsets = np.stack(np.meshgrid(*([np.arange(-1, 3)] * 3), indexing="ij"), axis=-1)
    indices = np.floor(fraction).astype(int) + offsets.reshape(-1, 3)
    nodes = origin + spacing * indices
    fluid = ~wall._body_interior_at_particles(nodes)
    weights = np.prod(_m4_prime_1d(fraction - indices), axis=1)
    corrected_indices, correction, budget = wall._m4_wall_corrections(
        source, strength, origin, spacing, shape
    )
    assert budget["wall_adjacent_particles"] == 1
    assert budget["excluded_signed_weight_l1"] > 0
    strengths = weights[fluid, None] * strength
    correction_by_node = {
        tuple(node): delta for node, delta in zip(corrected_indices, correction, strict=True)
    }
    strengths += np.stack([correction_by_node[tuple(node)] for node in indices[fluid]])
    nodes = nodes[fluid]
    targets = np.array([[0.87, 0.12, 0.10], [0.72, -0.32, -0.08], [1.08, 0.3, 0.21]])
    kernel = make_vortex_kernel("GAUSSIAN")
    core_radius = 0.08  # fixed physical blob, independent of grid spacing
    remapped = kernel.velocity_pair(
        targets[:, None, :] - nodes[None, :, :],
        strengths[None, :, :],
        core_radius,
        core_radius,
    ).sum(axis=1)
    reference = kernel.velocity_pair(targets - source[0], strength[0], core_radius, core_radius)
    relative_error = np.linalg.norm(remapped - reference) / np.linalg.norm(reference)
    circulation_error = np.linalg.norm(strengths.sum(axis=0) - strength[0])
    moment_error = np.linalg.norm(
        (nodes[:, :, None] * strengths[:, None, :]).sum(axis=0) - source[0, :, None] * strength[0]
    )
    return relative_error, circulation_error, moment_error, budget


@pytest.mark.parametrize("phase", [0.0, 0.37])
def test_curved_wall_remap_induced_velocity_refines(phase):
    results = [_remapped_velocity(spacing, phase) for spacing in (0.1, 0.05, 0.025)]
    errors = np.array([result[0] for result in results])
    assert errors[2] < errors[1] < errors[0], errors
    assert np.all(np.array([result[1] for result in results]) < 1e-6)
    assert np.all(np.array([result[2] for result in results]) < 1e-6)
