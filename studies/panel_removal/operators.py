"""Independent induction operators for the panel-removal feasibility study.

These deliberately bypass particle renewal, FMM and the production panel kernels.
They are offline diagnostics, not alternate time-stepping operators.
"""

import math

from numba import njit
import numpy as np
from scipy.spatial import cKDTree


@njit
def volume_induction(target, position, strength, sigma):
    """Evaluate independent 3D Gaussian Biot–Savart velocity in float64.

    Targets have shape (M,3), positions and strengths (N,3), and sigma (N,).
    Coordinates and sigma are in m; strengths are Gamma in m³/s. Sigma is
    Gaussian standard deviation, not the VPM core-radius convention.
    Returns a new (M,3) velocity array in m/s without freestream. Input arrays
    are unchanged. Direct summation costs O(M*N); exact-centre pairs contribute zero.
    """
    out = np.zeros((len(target), 3))
    for i in range(len(target)):
        for j in range(len(position)):
            dx = target[i, 0] - position[j, 0]
            dy = target[i, 1] - position[j, 1]
            dz = target[i, 2] - position[j, 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < 1e-30:
                continue
            r = np.sqrt(r2)
            q = r / sigma[j]
            factor = (math.erf(q / np.sqrt(2)) - np.sqrt(2 / np.pi) * q * np.exp(-q * q / 2)) / (
                4 * np.pi * r2 * r
            )
            out[i, 0] += (strength[j, 1] * dz - strength[j, 2] * dy) * factor
            out[i, 1] += (strength[j, 2] * dx - strength[j, 0] * dz) * factor
            out[i, 2] += (strength[j, 0] * dy - strength[j, 1] * dx) * factor
    return out


def polygon_sources(points, faces, vorticity, order=8):
    """Oriented edge quadrature for piecewise constant 2D cell vorticity.

    Applying Green's theorem analytically removes the volume kernel's 1/r
    singularity. This contour integration needs no boundary-element solve.
    ``faces`` uses VTK's flat [count, vertex ids, count, ...] encoding.

    Parameters
    ----------
    points : numpy.ndarray, shape (Nv, >=2)
        Native polygon vertices in m; only XY coordinates are used.
    faces : numpy.ndarray
        Flat VTK polygon connectivity, one vertex count followed by its IDs.
    vorticity : numpy.ndarray, shape (Ncell,)
        Signed cell-constant omega_z in s⁻¹, in connectivity order.
    order : int, default=8
        Gauss–Legendre quadrature nodes per polygon edge.

    Returns
    -------
    positions, weights : numpy.ndarray, shape (Nquadrature,2)
        Detached XY edge positions in m and outward-normal edge weights
        multiplied by vorticity, in m/s. Polygon orientation is corrected
        locally without changing the input arrays.
    """
    mu, weight = np.polynomial.legendre.leggauss(order)
    mu, weight = (mu + 1) / 2, weight / 2
    positions, weights = [], []
    cursor = 0
    for omega in vorticity:
        count = int(faces[cursor])
        indices = faces[cursor + 1 : cursor + count + 1]
        cursor += count + 1
        polygon = points[indices, :2]
        following = np.roll(polygon, -1, axis=0)
        signed_area = np.sum(polygon[:, 0] * following[:, 1] - polygon[:, 1] * following[:, 0])
        if signed_area < 0:
            polygon = polygon[::-1]
            following = np.roll(polygon, -1, axis=0)
        edge = following - polygon
        normal_ds = np.column_stack((edge[:, 1], -edge[:, 0]))
        positions.append((polygon[:, None] + mu[None, :, None] * edge[:, None]).reshape(-1, 2))
        weights.append((normal_ds[:, None] * weight[None, :, None] * omega).reshape(-1, 2))
    return np.concatenate(positions), np.concatenate(weights)


@njit
def polygon_induction(target, points, weights, span=-1.0):
    """Midspan induction: negative span means infinite; positive means finite.

    For half-span a, the radial primitive is log(r/(a+sqrt(a*a+r*r))).
    Its gradient supplies the finite-filament factor a/sqrt(a*a+r*r).
    Targets must not coincide with edge quadrature nodes.

    Inputs target (M,2) and points (N,2) are XY coordinates in m; weights
    (N,2) are outward-normal edge quadrature weights in m/s from
    polygon_sources. span>0 is total filament length in m; span<=0 selects
    the infinite limit. Return a new float64 (M,2) velocity in m/s, excluding
    freestream, with O(M*N) pair work. Inputs are read only.
    """
    out = np.zeros((len(target), 2))
    for i in range(len(target)):
        for j in range(len(points)):
            dx = target[i, 0] - points[j, 0]
            dy = target[i, 1] - points[j, 1]
            radius = np.sqrt(dx * dx + dy * dy)
            potential = np.log(radius)
            if span > 0:
                potential -= np.log(span / 2 + np.sqrt((span / 2) ** 2 + radius**2))
            out[i, 0] += weights[j, 1] * potential / (2 * np.pi)
            out[i, 1] -= weights[j, 0] * potential / (2 * np.pi)
    return out


def affine_sample(points, values, targets):
    """Fit values to the twelve nearest native cell centres at each target.

    points and targets have shapes (N,d) and (M,d), in the same world frame
    and length units. values has N rows of scalar or vector measurements.
    Return a new array with M rows and the same value units. An unweighted
    least-squares affine fit is evaluated at each target; no span-invariance
    constraint is imposed. Inputs are unchanged.
    """
    _, indices = cKDTree(points).query(targets, k=12)
    result = []
    for target, neighbours in zip(targets, indices, strict=True):
        matrix = np.column_stack((np.ones(12), points[neighbours] - target))
        result.append(np.linalg.lstsq(matrix, values[neighbours], rcond=None)[0][0])
    return np.asarray(result)


def error_metrics(prediction, reference):
    """Vector RMS/max in m/s; U_inf=1 makes them fractions of U_inf here."""
    magnitude = np.linalg.norm(prediction - reference, axis=1)
    return {"rms": float(np.sqrt(np.mean(magnitude**2))), "maximum": float(magnitude.max())}


def boundary_targets(dimension):
    """Return fixed equally weighted probes on the study's boundary box.

    For dimension=2, return (320,2) XY coordinates in m along x/y=±1.5.
    Otherwise return (216,3) coordinates on the six faces of the 3 m cube.
    Probe nodes exclude corners/edges. The returned float64 array is owned
    by the caller; these fixed grids are diagnostic samples, not quadrature areas.
    """
    if dimension == 2:
        line = np.linspace(-1.47, 1.47, 80)
        return np.concatenate(
            [np.column_stack((line, np.full_like(line, y))) for y in (-1.5, 1.5)]
            + [np.column_stack((np.full_like(line, x), line)) for x in (-1.5, 1.5)]
        )
    line = np.linspace(-1.35, 1.35, 6)
    plane = np.array(np.meshgrid(line, line)).reshape(2, -1).T
    result = []
    for axis in range(3):
        for side in (-1.5, 1.5):
            block = np.zeros((len(plane), 3))
            block[:, axis] = side
            block[:, [k for k in range(3) if k != axis]] = plane
            result.append(block)
    return np.concatenate(result)
