"""Cell-field filters on an unstructured mesh.

One-ring, volume-weighted box filtering. Used by the dynamic Smagorinsky model
(test filter) and by the coupler's blending-zone relaxation (to relax only the scales
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

    Two centre weightings are available, and the choice matters:

    ``centre_weight="volume"`` (default) is the classical box filter — the cell
    carries its own volume, so on a uniform hex mesh it is the plain (1+6)-point
    average.  Its transfer function goes NEGATIVE at the grid scale (−1/3 on a
    1-D chain, −5/7 on a hex interior).  That is fine for the dynamic
    Smagorinsky test filter, which only needs a wider filter, but it makes
    ``f − G∗f`` exceed ``f`` at the grid scale.

    ``centre_weight="neighbour_sum"`` gives the cell the same total weight as
    its neighbours combined, which is the unstructured generalisation of the
    (1, 2, 1)/4 stencil: DC gain exactly 1, grid-scale gain exactly 0, and no
    negative lobe.  Use this whenever the RESIDUAL ``f − G∗f`` is the quantity of
    interest — with a negative-lobe filter the residual is amplified, so a term
    built from it flips sign and drives the mode it was meant to leave alone.
    """

    def __init__(self, mesh_data: dict, geo_data: dict, centre_weight: str = "volume"):
        valid = ("volume", "neighbour_sum")
        if centre_weight not in valid:
            raise ValueError(f"centre_weight must be one of {valid}, got {centre_weight!r}")
        n_int = int(mesh_data["n_interior_faces"])
        vol = np.asarray(geo_data["element_volumes"], dtype=np.float64)
        own = np.asarray(mesh_data["owners"])[:n_int]
        nei = np.asarray(mesh_data["neighbours"])[:n_int]

        neighbour_sum = np.zeros_like(vol)
        np.add.at(neighbour_sum, own, vol[nei])
        np.add.at(neighbour_sum, nei, vol[own])

        centre = vol if centre_weight == "volume" else neighbour_sum
        self._own, self._nei, self._vol = own, nei, vol
        self._centre = centre
        self._denom = centre + neighbour_sum

    def __call__(self, f: np.ndarray) -> np.ndarray:
        """Filter a cell field ``(n_elements, ...)``; shape is preserved.

        Works for scalars ``(n,)``, vectors ``(n, 3)`` and tensors ``(n, 3, 3)``
        alike — the weights are reshaped to broadcast against whatever trailing
        dimensions ``f`` has.
        """
        vol = self._vol
        shape = (-1,) + (1,) * (f.ndim - 1)
        num = (self._centre.reshape(shape) * f).copy()
        np.add.at(num, self._own, vol[self._nei].reshape(shape) * f[self._nei])
        np.add.at(num, self._nei, vol[self._own].reshape(shape) * f[self._own])
        return num / np.where(self._denom > 0.0, self._denom, 1.0).reshape(shape)
