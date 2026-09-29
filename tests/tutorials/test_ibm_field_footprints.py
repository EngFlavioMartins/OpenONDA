"""Field plotting covers stretched FVM cells with their actual faces."""

import numpy as np
import pyvista as pv

from tests._tutorial_helpers import load_tutorial_module


def test_stretched_cell_footprints_cover_the_cross_section():
    plot_fields = load_tutorial_module("fvm/cylinder_ibm", "assets.plot_fields")
    x, y, z = np.meshgrid([0.0, 1.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.5], indexing="ij")
    mesh = pv.StructuredGrid(x, y, z).cast_to_unstructured_grid()

    footprints = plot_fields._cell_footprints(mesh)
    edges = np.roll(footprints, -1, axis=1)
    areas = 0.5 * np.abs(
        np.sum(footprints[:, :, 0] * edges[:, :, 1] - footprints[:, :, 1] * edges[:, :, 0], axis=1)
    )

    np.testing.assert_allclose(np.sort(areas), [1.0, 1.0, 2.0, 2.0])
    np.testing.assert_allclose(np.unique(footprints[:, :, 0]), [0.0, 1.0, 3.0])
    np.testing.assert_allclose(np.unique(footprints[:, :, 1]), [0.0, 1.0, 2.0])
    assert areas.sum() == 6.0
