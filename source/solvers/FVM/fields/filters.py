"""Cell-field filters on an unstructured mesh.

One-ring, volume-weighted box filtering. Used by the dynamic Smagorinsky model
(test filter) and by the coupler's fringe relaxation (to relax only the scales
its target can represent).

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
    check ``geo_data['element_volumes']`` if a specific physical width matters.

    Parallel note: one application needs no halo exchange provided the caller's
    ghost rows are current, because owned rows only read their own one-ring.
    The ghost rows of the RESULT are wrong (their one-ring is truncated at the
    partition boundary), so chaining a second application requires an
    ``exchange_halo`` on the intermediate — see
    ``ParallelContext.exchange_halo``.
    """

    def __init__(self, mesh_data: dict, geo_data: dict):
        n_int = int(mesh_data["n_interior_faces"])
        vol = np.asarray(geo_data["element_volumes"], dtype=np.float64)
        own = np.asarray(mesh_data["owners"])[:n_int]
        nei = np.asarray(mesh_data["neighbours"])[:n_int]
        denom = vol.copy()
        np.add.at(denom, own, vol[nei])
        np.add.at(denom, nei, vol[own])
        self._own, self._nei, self._vol, self._denom = own, nei, vol, denom

    def __call__(self, f: np.ndarray) -> np.ndarray:
        """Filter a cell field ``(n_elements, ...)``; shape is preserved.

        Works for scalars ``(n,)``, vectors ``(n, 3)`` and tensors ``(n, 3, 3)``
        alike — the volume weights are reshaped to broadcast against whatever
        trailing dimensions ``f`` has.
        """
        vol = self._vol
        shape = (-1,) + (1,) * (f.ndim - 1)
        num = (vol.reshape(shape) * f).copy()
        np.add.at(num, self._own, vol[self._nei].reshape(shape) * f[self._nei])
        np.add.at(num, self._nei, vol[self._own].reshape(shape) * f[self._own])
        return num / self._denom.reshape(shape)
