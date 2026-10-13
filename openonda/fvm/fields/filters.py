"""Cell-field filters on an unstructured mesh.

One-ring, volume-weighted box filtering. Used by the coupler's blending-zone
relaxation to relax only the scales its target can represent.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

import numpy as np


class CellBoxFilter:
    """Volume-weighted one-ring box filter for cell-centred fields.

    The filtered value at a cell is the volume-weighted average over the cell
    and its face neighbours.  The static denominator ``Σ V`` over each one-ring
    is pre-computed once from the mesh topology, so applying the filter is two
    scatter-adds.

    Effective width is roughly one cell either side, i.e. it follows the LOCAL
    cell size rather than a fixed length.  That is usually what you want on a
    graded mesh, but it does mean the filter width varies through the domain —
    check ``geo_data['cell_volume']`` if a specific physical width matters.

    Parallel note: one application needs no halo exchange provided the caller's
    ghost rows are current, because owned rows only read their own one-ring.
    The ghost rows of the RESULT are wrong (their one-ring is truncated at the
    partition boundary), so chaining a second application requires an
    ``exchange_halo`` on the intermediate — see
    ``ParallelContext.exchange_halo``.

    The cell gets the same total weight as its neighbours combined, which is
    the unstructured generalisation of the (1, 2, 1)/4 stencil: DC gain exactly
    1, grid-scale gain exactly 0, and no negative lobe.  The classical box
    filter (centre weighted by its own volume) goes negative at the grid scale,
    which would amplify the residual ``f − G∗f`` that the coupler relaxes.
    """

    def __init__(self, mesh_data: dict, geo_data: dict) -> None:
        """Build one reusable one-ring filter from mesh connectivity.

        Parameters
        ----------
        mesh_data : dict
            Native topology containing interior owner/neighbour arrays.
        geo_data : dict
            Geometry containing positive cell volumes in m³.
        """
        n_int = int(mesh_data["n_interior_faces"])
        cell_volume = np.asarray(geo_data["cell_volume"], dtype=np.float64)
        own = np.asarray(mesh_data["owners"])[:n_int]
        nei = np.asarray(mesh_data["neighbours"])[:n_int]

        neighbour_sum = np.zeros_like(cell_volume)
        np.add.at(neighbour_sum, own, cell_volume[nei])
        np.add.at(neighbour_sum, nei, cell_volume[own])

        self._own, self._nei, self._vol = own, nei, cell_volume
        self._centre = neighbour_sum
        self._denom = 2.0 * neighbour_sum

    def __call__(self, f: np.ndarray) -> np.ndarray:
        """Filter a cell field ``(n_elements, ...)``; shape is preserved.

        Works for scalars ``(n,)``, vectors ``(n, 3)`` and tensors ``(n, 3, 3)``
        alike — the weights are reshaped to broadcast against whatever trailing
        dimensions ``f`` has.

        Parameters
        ----------
        f : numpy.ndarray
            Cell field with leading dimension ``n_cells`` and any trailing
            scalar/vector/tensor shape.

        Returns
        -------
        numpy.ndarray
            Filtered field with the same shape and units as ``f``.
        """
        cell_volume = self._vol
        shape = (-1,) + (1,) * (f.ndim - 1)
        num = (self._centre.reshape(shape) * f).copy()
        np.add.at(num, self._own, cell_volume[self._nei].reshape(shape) * f[self._nei])
        np.add.at(num, self._nei, cell_volume[self._own].reshape(shape) * f[self._own])
        return num / np.where(self._denom > 0.0, self._denom, 1.0).reshape(shape)
