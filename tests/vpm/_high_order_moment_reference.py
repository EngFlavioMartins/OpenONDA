"""UNWIRED NumPy oracle for compact, scaled Cartesian source moments.

Matches self-FMM's stored positive (source-centre) offsets; field contraction
supplies (-1)**degree. This validates algebra, not GPU performance or a complete
floating-point error certificate. It authorizes no new production admission.
"""

from dataclasses import dataclass
from functools import cache
import math

import numpy as np


@cache
def multi_indices(order):
    return tuple(
        (a, b, degree - a - b)
        for degree in range(order + 1)
        for a in range(degree + 1)
        for b in range(degree - a + 1)
    )


def factorial(alpha):
    return math.prod(math.factorial(x) for x in alpha)


def direct_moments(position, strength, centre, scale, order):
    """M_alpha = sum Gamma*((source-centre)/scale)**alpha / alpha!."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("cell scale must be finite and positive, including singleton cells")
    offset = (np.asarray(position, np.float64) - centre) / scale
    strength = np.asarray(strength, np.float64)
    values, absolute = [], []
    for alpha in multi_indices(order):
        terms = strength * (np.prod(offset ** np.array(alpha), axis=1) / factorial(alpha))[:, None]
        values.append(np.sum(terms, axis=0))
        absolute.append(np.sum(np.abs(terms), axis=0))
    return np.asarray(values), np.asarray(absolute)


def translate_moments(values, child_centre, child_scale, parent_centre, parent_scale, order):
    """Exact polynomial M2M algebra; scales need not be equal or nested."""
    if not all(math.isfinite(s) and s > 0 for s in (child_scale, parent_scale)):
        raise ValueError("moment scales must be finite and positive")
    indices = multi_indices(order)
    delta = (np.asarray(child_centre) - parent_centre) / parent_scale
    ratio = child_scale / parent_scale
    translated = np.zeros_like(values, dtype=np.float64)
    for row, alpha in enumerate(indices):
        for column, beta in enumerate(indices):
            difference = np.asarray(alpha) - beta
            if np.all(difference >= 0):
                weight = (
                    ratio ** sum(beta) * np.prod(delta**difference) / factorial(difference)
                )
                translated[row] += weight * values[column]
    return translated


@dataclass
class CompactMoments:
    order: int
    node_ids: np.ndarray
    centres: np.ndarray
    scales: np.ndarray
    values: np.ndarray
    slots: dict

    def fields(self, node, target):
        slot = self.slots[node]
        return moment_fields(
            self.values[slot], self.centres[slot], self.scales[slot], target, self.order
        )


def build_compact_moments(position, strength, *, centres, scales, leaves, children, order):
    """Store only caller-selected active cells, not every fine LBVH node.

centres/scales map node IDs to metadata; leaves maps active terminal IDs to
source indices; children maps active internal IDs to their two active children.
The caller must supply node_centre (not strength COM), exactly as self-FMM does.
No implicit N/32 size assumption or production allocation policy is made.
"""
    nodes = tuple(centres)
    if set(nodes) != set(scales) or set(leaves) & set(children):
        raise ValueError("inconsistent active-cell metadata")
    if set(leaves) | set(children) != set(nodes):
        raise ValueError("every active cell must be a leaf or a complete internal cell")
    owned = np.concatenate([np.asarray(indices, np.int64) for indices in leaves.values()])
    if not np.array_equal(np.sort(owned), np.arange(len(position))):
        raise ValueError("active leaves must partition the source prefix exactly once")
    descendants = [child for pair in children.values() for child in pair]
    if any(len(pair) != 2 for pair in children.values()) or len(set(descendants)) != len(descendants):
        raise ValueError("active hierarchy must be a binary tree/forest without shared children")
    if not set(descendants) <= set(nodes):
        raise ValueError("all active children must have moment storage")
    slots = {node: slot for slot, node in enumerate(nodes)}
    values = np.zeros((len(nodes), len(multi_indices(order)), 3))
    ready, visiting = set(), set()

    def build(node):
        if node in ready:
            return
        if node in visiting:
            raise ValueError("active hierarchy contains a cycle")
        visiting.add(node)
        if node in leaves:
            selected = np.asarray(leaves[node], np.int64)
            values[slots[node]] = direct_moments(
                np.asarray(position)[selected], np.asarray(strength)[selected],
                centres[node], scales[node], order,
            )[0]
        else:
            for child in children[node]:
                build(child)
                values[slots[node]] += translate_moments(
                    values[slots[child]], centres[child], scales[child],
                    centres[node], scales[node], order,
                )
        visiting.remove(node)
        ready.add(node)

    for node in nodes:
        build(node)
    return CompactMoments(
        order, np.asarray(nodes, np.int64), np.asarray([centres[n] for n in nodes]),
        np.asarray([scales[n] for n in nodes]), values, slots,
    )


def _green_taylor(direction, order):
    """Independent binomial-series coefficients of 1/|direction+h|.

Coefficients are derivatives divided by alpha!, dimensionless. Expanding
(1+2*n.h+h.h)^(-1/2) avoids the production Cartesian derivative recurrence.
"""
    zero = (0, 0, 0)
    polynomial = {}
    for axis in range(3):
        linear, square = [0, 0, 0], [0, 0, 0]
        linear[axis], square[axis] = 1, 2
        polynomial[tuple(linear)] = 2 * direction[axis]
        polynomial[tuple(square)] = 1.0
    result, power, coefficient = {zero: 1.0}, {zero: 1.0}, 1.0
    for exponent in range(1, order + 1):
        product = {}
        for alpha, value in power.items():
            for beta, term in polynomial.items():
                gamma = tuple(a + b for a, b in zip(alpha, beta, strict=True))
                if sum(gamma) <= order:
                    product[gamma] = product.get(gamma, 0.0) + value * term
        power = product
        coefficient *= (-0.5 - exponent + 1) / exponent
        for alpha, value in power.items():
            result[alpha] = result.get(alpha, 0.0) + coefficient * value
    return result


def moment_fields(values, centre, scale, target, order):
    """Singular velocity/Jacobian oracle with scale-normalized contractions."""
    displacement = np.asarray(target) - centre
    distance = float(np.linalg.norm(displacement))
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("target must be separated from the expansion centre")
    green = _green_taylor(displacement / distance, order + 2)
    first, second = np.zeros((3, 3)), np.zeros((3, 3, 3))
    for alpha, moment in zip(multi_indices(order), values, strict=True):
        scaled = moment * (-scale / distance) ** sum(alpha)
        for axis in range(3):
            beta = list(alpha)
            beta[axis] += 1
            first[axis] += scaled * green[tuple(beta)] * factorial(beta) / distance**2
            for column in range(3):
                gamma = beta.copy()
                gamma[column] += 1
                second[axis, column] += scaled * green[tuple(gamma)] * factorial(gamma) / distance**3
    velocity = np.array([first[1, 2] - first[2, 1], first[2, 0] - first[0, 2], first[0, 1] - first[1, 0]])
    gradient = np.array([
        second[1, :, 2] - second[2, :, 1],
        second[2, :, 0] - second[0, :, 2],
        second[0, :, 1] - second[1, :, 0],
    ])
    return velocity / (4 * math.pi), gradient / (4 * math.pi)


def storage_and_work(order, active_cells, source_count, target_order=7, scalar_bytes=4):
    """Exact coefficient counts; work terms are not hardware-time predictions."""
    coefficients = math.comb(order + 3, 3)
    return {
        "source_coefficients": coefficients,
        "moment_bytes": active_cells * coefficients * 3 * scalar_bytes,
        "dense_node_map_bytes": max(0, 2 * source_count - 1) * 4,
        "p2m_vector_terms": source_count * coefficients,
        "m2m_vector_terms_per_binary_parent": 2 * math.comb(order + 6, 6),
        "m2l_vector_terms_per_pair": coefficients * math.comb(target_order + 3, 3),
        "m2l_derivative_coefficients": math.comb(order + target_order + 3, 3),
        "runtime_qualified": False,
        "complete_arithmetic_error_certificate": False,
    }
