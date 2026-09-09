# SPDX-License-Identifier: GPL-3.0-or-later
"""Scalar cfMesh volumeOptimizer kernel (upstream 3ff855551482).

Numba compiles the scalar loops without fast-math: reduction order, reciprocal
scaling and OpenFOAM's tensor inverse are part of this numerical algorithm.
There is no external cfMesh/OpenFOAM dependency in production.
"""

from __future__ import annotations

import math

import numpy as np

from source._numba import cacheable_njit as njit


@njit(cache=True)
def _squared_length(v: np.ndarray) -> float:
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


@njit(cache=True)
def _properties(values: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    volumes = np.empty(len(values))
    lengths = np.empty(len(values))
    minimum_volume = 1.0e300
    maximum_length = 0.0
    for i in range(len(values)):
        a, b, c = values[i]
        normal = np.cross(b - a, c - a)
        offset = point - a
        volume = (1.0 / 6.0) * (
            normal[0] * offset[0] + normal[1] * offset[1] + normal[2] * offset[2]
        )
        length = (
            _squared_length(point - a) + _squared_length(point - b) + _squared_length(point - c)
        )
        volumes[i] = volume
        lengths[i] = length
        minimum_volume = min(minimum_volume, volume)
        maximum_length = max(maximum_length, length)
    factor = 1.0e-15 * maximum_length if minimum_volume < 1.0e-15 * maximum_length else 0.0
    return volumes, lengths, factor


@njit(cache=True)
def _objective(values: np.ndarray, point: np.ndarray) -> float:
    volumes, lengths, factor = _properties(values, point)
    result = 0.0
    for i in range(len(values)):
        volume = volumes[i]
        stable_volume = 0.5 * (volume + math.sqrt(volume * volume + factor))
        if stable_volume <= 0.0:
            return math.inf
        result += lengths[i] / math.pow(stable_volume, 2.0 / 3.0)
    return result


@njit(cache=True)
def _gradients(values: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    volumes, lengths, factor = _properties(values, point)
    gradient = np.zeros(3)
    hessian = np.zeros((3, 3))
    constant = (2.0 / 3.0) * math.pow(0.5, 2.0 / 3.0)
    for i in range(len(values)):
        a, b, c = values[i]
        volume = volumes[i]
        length = lengths[i]
        volume_gradient = (1.0 / 6.0) * np.cross(b - a, c - a)
        stable = math.sqrt(volume * volume + factor)
        stable_volume = 0.5 * (volume + stable)
        if stable_volume < 1.0e-300:
            raise ValueError("cfMesh volume optimizer has a zero stabilised tetrahedron volume")
        stable_gradient = 0.5 * (volume_gradient + volume * volume_gradient / stable)
        length_gradient = 2.0 * (3.0 * point - a - b - c)
        root = math.pow(2.0 * stable_volume, 1.0 / 3.0)
        power = math.pow(stable_volume, 2.0 / 3.0)
        power_squared = power * power
        power_gradient = constant * (2.0 * stable_gradient) / root
        for row in range(3):
            gradient[row] += (
                length_gradient[row] / power - length * power_gradient[row] / power_squared
            )
            for column in range(3):
                outer = volume_gradient[row] * volume_gradient[column]
                stable_hessian = outer / stable - volume * volume * outer / math.pow(stable, 3.0)
                power_hessian = constant * (stable_hessian / root) - (constant / 3.0) * 4.0 * (
                    stable_gradient[row] * stable_gradient[column]
                ) / math.pow(root, 4.0)
                mixed = length_gradient[row] * (power_gradient[column] / power_squared)
                if row == column:
                    mixed = 2.0 * mixed
                else:
                    mixed += length_gradient[column] * (power_gradient[row] / power_squared)
                hessian[row, column] += (
                    (6.0 if row == column else 0.0) / power
                    - mixed
                    - length * power_hessian / power_squared
                    + 2.0
                    * length
                    * (power_gradient[row] * power_gradient[column])
                    / (power_squared * power)
                )
    return gradient, hessian


@njit(cache=True)
def _tensor_determinant(t: np.ndarray) -> float:
    # OpenFOAM Tensor::det(), not a factored or pivoted determinant.
    return (
        t[0, 0] * t[1, 1] * t[2, 2]
        + t[0, 1] * t[1, 2] * t[2, 0]
        + t[0, 2] * t[1, 0] * t[2, 1]
        - t[0, 0] * t[1, 2] * t[2, 1]
        - t[0, 1] * t[1, 0] * t[2, 2]
        - t[0, 2] * t[1, 1] * t[2, 0]
    )


@njit(cache=True)
def _tensor_solve(t: np.ndarray, determinant: float, gradient: np.ndarray) -> np.ndarray:
    inverse = (
        np.asarray(
            (
                (
                    t[1, 1] * t[2, 2] - t[2, 1] * t[1, 2],
                    t[0, 2] * t[2, 1] - t[0, 1] * t[2, 2],
                    t[0, 1] * t[1, 2] - t[0, 2] * t[1, 1],
                ),
                (
                    t[2, 0] * t[1, 2] - t[1, 0] * t[2, 2],
                    t[0, 0] * t[2, 2] - t[0, 2] * t[2, 0],
                    t[1, 0] * t[0, 2] - t[0, 0] * t[1, 2],
                ),
                (
                    t[1, 0] * t[2, 1] - t[1, 1] * t[2, 0],
                    t[0, 1] * t[2, 0] - t[0, 0] * t[2, 1],
                    t[0, 0] * t[1, 1] - t[1, 0] * t[0, 1],
                ),
            )
        )
        / determinant
    )
    return inverse[:, 0] * gradient[0] + inverse[:, 1] * gradient[1] + inverse[:, 2] * gradient[2]


@njit(cache=True)
def _optimise_triangles(values: np.ndarray, point: np.ndarray, tolerance: float) -> np.ndarray:
    lower = values[0, 0].copy()
    upper = lower.copy()
    for triangle in values:
        for vertex in triangle:
            lower = np.minimum(lower, vertex)
            upper = np.maximum(upper, vertex)
    magnitude = math.sqrt(_squared_length(upper - lower))
    if magnitude <= 1.0e-300:
        return point.copy()
    scale = 1.0 / magnitude
    values = values * scale
    lower = lower * scale
    upper = upper * scale
    candidate = 0.5 * (upper + lower)
    half_range = 0.5 * (upper - lower)
    after = _objective(values, candidate)
    for _iteration in range(100):
        before = after
        after = 1.0e300
        best = np.zeros(3)
        for direction in range(8):
            trial = candidate.copy()
            for axis in range(3):
                sign = 1.0 if direction & (1 << axis) else -1.0
                trial[axis] += 0.5 * sign * half_range[axis]
            value = _objective(values, trial)
            if value < after:
                after = value
                best = trial
        candidate = best
        half_range *= 0.5
        if abs(after - before) / after < tolerance:
            break
    divide_point = candidate.copy()
    divide_value = after
    after = _objective(values, candidate)
    for _iteration in range(100):
        original = candidate.copy()
        before = after
        gradient, hessian = _gradients(values, candidate)
        determinant = _tensor_determinant(hessian)
        finished = False
        if determinant > 1.0e-15:
            displacement = _tensor_solve(hessian, determinant, gradient)
            candidate -= displacement
            after = _objective(values, candidate)
            relaxation = 0.8
            loops = 0
            while after > before:
                candidate = original - relaxation * displacement
                relaxation *= 0.5
                after = _objective(values, candidate)
                if after < before:
                    continue
                loops += 1
                if loops == 5:
                    candidate = original
                    finished = True
                    after = before
            if abs(before - after) / before < tolerance:
                finished = True
        else:
            displacement = np.zeros(3)
            volumes, _lengths, _factor = _properties(values, candidate)
            for index in range(len(values)):
                if volumes[index] < 1.0e-15:
                    a, b, c = values[index]
                    normal = 0.5 * np.cross(b - a, c - a)
                    length = math.sqrt(_squared_length(normal))
                    if length > 1.0e-300:
                        displacement += 0.01 * (normal / length)
            candidate += displacement
            after = _objective(values, candidate)
        if finished:
            break
    if after > divide_value:
        candidate = divide_point
    return candidate / scale


def _optimise_volume_point(
    points: np.ndarray, triangles: np.ndarray, point: np.ndarray, tolerance: float = 1.0e-5
) -> np.ndarray:
    return _optimise_triangles(points[triangles], point, tolerance)


@njit(cache=True)
def _knupp_metric(
    point: np.ndarray, normals: np.ndarray, centres: np.ndarray, beta: float
) -> float:
    result = 0.0
    for i in range(len(normals)):
        offset = point - centres[i]
        normal = normals[i]
        distance = (normal[0] * offset[0] + normal[1] * offset[1] + normal[2] * offset[2]) - beta
        value = abs(distance) - distance
        result += value * value
    return result


@njit(cache=True)
def _knupp_triangles(values: np.ndarray, point: np.ndarray) -> np.ndarray:
    lower = values[0, 0].copy()
    upper = lower.copy()
    normals = np.empty((len(values), 3))
    centres = np.empty_like(normals)
    count = 0
    for triangle in values:
        a, b, c = triangle
        for vertex in triangle:
            lower = np.minimum(lower, vertex)
            upper = np.maximum(upper, vertex)
        normal = 0.5 * np.cross(b - a, c - a)
        length = math.sqrt(_squared_length(normal))
        if length > 1.0e-300:
            normals[count] = normal / length
            centres[count] = (1.0 / 3.0) * (a + b + c)
            count += 1
    normals = normals[:count]
    centres = centres[:count]
    candidate = point.copy()
    if np.any(candidate < lower) or np.any(candidate > upper):
        candidate = 0.5 * (lower + upper)
    beta = 0.01 * math.sqrt(_squared_length(upper - lower))
    tolerance = (2.0e-15 * 2.0e-15) * _squared_length(lower - upper)
    for _outer in range(5):
        previous = _knupp_metric(candidate, normals, centres, beta)
        for _iteration in range(10):
            original = candidate.copy()
            gradient = np.zeros(3)
            hessian = np.zeros((3, 3))
            for i in range(count):
                offset = candidate - centres[i]
                normal = normals[i]
                distance = (
                    normal[0] * offset[0] + normal[1] * offset[1] + normal[2] * offset[2]
                ) - beta
                # Foam::sign(0) is +1, whereas numpy.sign(0) is zero.
                metric_gradient = (0.0 if distance >= 0.0 else -2.0) * normal
                gradient += (abs(distance) - distance) * metric_gradient
                for row in range(3):
                    for column in range(3):
                        hessian[row, column] += metric_gradient[row] * metric_gradient[column]
            determinant = _tensor_determinant(hessian)
            displacement = np.zeros(3)
            if determinant > 1.0e-15:
                displacement = _tensor_solve(hessian, determinant, gradient)
                if not np.isfinite(displacement).all():
                    displacement = np.zeros(3)
                candidate -= displacement
                current = _knupp_metric(candidate, normals, centres, beta)
                relaxation = 0.8
                loops = 0
                while current > previous:
                    candidate = original - relaxation * displacement
                    relaxation *= 0.5
                    current = _knupp_metric(candidate, normals, centres, beta)
                    if current < previous:
                        continue
                    loops += 1
                    if loops == 5:
                        candidate = original
                        displacement = np.zeros(3)
                        current = 0.0
                previous = current
            if _squared_length(displacement) <= tolerance:
                break
        if not (previous < 1.0e-300 and _knupp_metric(candidate, normals, centres, 0.0) > 1.0e-300):
            break
        beta /= 2.0
    return candidate


def _knupp_point(points: np.ndarray, triangles: np.ndarray, point: np.ndarray) -> np.ndarray:
    return _knupp_triangles(points[triangles], point)
