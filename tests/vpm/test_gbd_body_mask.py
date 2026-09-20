"""Curved-body exclusion for grid-based VPM diffusion."""

from __future__ import annotations

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin


@ti.data_oriented
class _Harness(_GridDiffusionMixin):
    pass


def _ensure_taichi_cpu() -> None:
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu)


def test_cylinder_body_mask_excludes_only_open_solid_interior():
    _ensure_taichi_cpu()
    physics = _Harness()
    physics._init_grid_diffusion()
    physics._body_mask_grid = ti.field(dtype=ti.i32, shape=(7, 7, 7))
    physics.configure_body_cylinder((-0.5, 0.5, -0.5, 0.5, -1.0, 1.0), axis="z")

    physics._prepare_body_mask_current_grid(np.array([-1.5, -1.5, -1.5]), 0.5, 7, 7, 7)
    mask = physics._body_mask_grid.to_numpy()

    assert mask[3, 3, 3] == 1  # cylinder centre
    assert mask[4, 3, 3] == 0  # radial surface
    assert mask[3, 3, 5] == 0  # end-cap surface
    assert mask[5, 3, 3] == 0  # exterior fluid


def test_cylinder_body_mask_rejects_noncircular_transverse_bounds():
    physics = _Harness()
    physics._init_grid_diffusion()
    with pytest.raises(ValueError, match="circular diameter"):
        physics.configure_body_cylinder((-0.5, 0.5, -0.25, 0.25, -1.0, 1.0), axis=2)
