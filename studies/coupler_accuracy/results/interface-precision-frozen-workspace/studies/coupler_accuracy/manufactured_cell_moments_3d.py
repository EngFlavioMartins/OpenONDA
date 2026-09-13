"""First central vorticity moments from the actual 3D cell boundary."""

from __future__ import annotations

import numpy as np

from studies.coupler_accuracy.native_velocity_curl_integrals_3d import triangle_rule


def native_first_vorticity_moment(native, centroid, fields, velocity_integral, *, order, chunk_size=128, progress=None):
    """M_ij = int (y-c)_i omega_j = int_boundary (y-c)_i(n cross u)_j
    minus (e_i cross int_cell u)_j. Retain the face-fan normals and moments.
    """
    barycentric, weight = triangle_rule(order)
    result = np.zeros((len(fields), native.n_cells, 3, 3))
    for start in range(0, len(native.triangles), chunk_size):
        rows = slice(start, start+chunk_size)
        triangle = native.triangles[rows]
        centre = triangle.mean(axis=1)
        points = triangle[:, None, 0]+np.einsum("qa,tad->tqd", barycentric[:, 1:], triangle[:, 1:]-triangle[:, :1])
        vector_area = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])/2
        own, nei = native.owners[rows], native.neighbours[rows]
        interior = nei >= 0
        for index, field in enumerate(fields):
            curl_flux = np.cross(vector_area[:, None], field.velocity(points))
            gamma = np.einsum("q,tqi->ti", weight, curl_flux)
            moment = np.einsum("q,tqi,tqj->tij", weight, points-centre[:, None], curl_flux)
            np.add.at(result[index], own, moment+(centre-centroid[own])[:, :, None]*gamma[:, None])
            np.add.at(result[index], nei[interior], -moment[interior]
                      -(centre[interior]-centroid[nei[interior]])[:, :, None]*gamma[interior, None])
        if progress and (start % 16384 == 0 or start+chunk_size >= len(native.triangles)):
            progress(min(start+chunk_size, len(native.triangles)), len(native.triangles), order)
    result -= np.cross(np.eye(3)[None, None], np.asarray(velocity_integral)[:, :, None])
    return result
