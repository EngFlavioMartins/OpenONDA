"""One-level far-field expansion for constant-strength source panels.

Far from a closed body the whole surface looks like a single dipole.  Each
body is therefore compressed into a scalar monopole and vector dipole.  The
monopole is retained so a legitimate non-zero net flux remains correct.  On a
320-panel sphere, dipole-only relative velocity errors measured at 4, 8, and
16 body radii were ``7.2e-6``, ``4.8e-7``, and ``3.0e-8`` respectively.
"""

from dataclasses import dataclass

import numpy as np
import taichi as ti

from ....config.constants import PANEL_EPSILON
from ..solver.lattice import PanelBody
from .source_velocity import compute_source_velocity


@dataclass(frozen=True)
class PanelFarFieldBody:
    """Source moments and acceptance geometry for one contiguous body."""

    uid: str
    start_idx: int
    count: int
    centre: np.ndarray
    radius: float
    monopole: float
    dipole: np.ndarray


def build_far_field_bodies(
    panel_centre: np.ndarray,
    panel_area: np.ndarray,
    source_strength: np.ndarray,
    bodies: list[PanelBody],
) -> list[PanelFarFieldBody]:
    """Build one monopole/dipole expansion per body in ``O(N_panels)``."""
    expansions = []
    for body in bodies:
        panel_slice = slice(body.start_idx, body.start_idx + body.count)
        centres = np.asarray(panel_centre[panel_slice], dtype=np.float64)
        area = np.asarray(panel_area[panel_slice], dtype=np.float64)
        strength = np.asarray(source_strength[panel_slice], dtype=np.float64)
        area_total = float(np.sum(area))
        centre = (
            np.sum(centres * area[:, None], axis=0) / area_total
            if area_total > 0.0
            else np.mean(centres, axis=0)
        )
        radius = float(np.max(np.linalg.norm(centres - centre, axis=1))) if body.count else 0.0
        weighted_source = strength * area
        expansions.append(
            PanelFarFieldBody(
                uid=body.uid,
                start_idx=body.start_idx,
                count=body.count,
                centre=centre,
                radius=radius,
                monopole=float(np.sum(weighted_source)),
                dipole=np.sum(weighted_source[:, None] * (centres - centre), axis=0),
            )
        )
    return expansions


def far_field_interaction_fraction(
    points: np.ndarray,
    bodies: list[PanelFarFieldBody],
    acceptance: float,
    min_panels: int,
) -> float:
    """Return the fraction of eligible body-target interactions expanded."""
    eligible = [body for body in bodies if body.count >= min_panels and body.radius > 0.0]
    denominator = len(points) * len(eligible)
    if denominator == 0:
        return 0.0
    far = 0
    for body in eligible:
        distances = np.linalg.norm(points - body.centre[None, :], axis=1)
        far += int(np.count_nonzero(distances > acceptance * body.radius))
    return float(far / denominator)


def evaluate_source_far_field(point: np.ndarray, body: PanelFarFieldBody) -> np.ndarray:
    """Evaluate one body's monopole-plus-dipole velocity at ``point``."""
    r = np.asarray(point, dtype=np.float64) - body.centre
    distance = float(np.linalg.norm(r))
    if distance <= 0.0:
        return np.zeros(3, dtype=np.float64)
    inv_r = 1.0 / distance
    inv_r3 = inv_r**3
    inv_r5 = inv_r**5
    return (
        body.monopole * r * inv_r3
        + 3.0 * r * np.dot(body.dipole, r) * inv_r5
        - body.dipole * inv_r3
    ) / (4.0 * np.pi)


@ti.kernel
def accumulate_source_panel_velocity_with_far_field_on_field(
    vertex_position: ti.template(),
    normal: ti.template(),
    source_strength: ti.template(),
    target_position: ti.template(),
    target_velocity: ti.template(),
    body_start: ti.types.ndarray(ndim=1),
    body_count: ti.types.ndarray(ndim=1),
    body_centre: ti.types.ndarray(ndim=2),
    body_radius: ti.types.ndarray(ndim=1),
    body_monopole: ti.types.ndarray(ndim=1),
    body_dipole: ti.types.ndarray(ndim=2),
    n_bodies: ti.i32,
    n_targets: ti.i32,
    acceptance: ti.f64,
    min_panels: ti.i32,
) -> ti.i32:
    """Accumulate source-panel velocity, expanding accepted body ranges."""
    far_interactions = 0
    for i in range(n_targets):
        point = target_position[i]
        value = point * 0.0
        for body_idx in range(n_bodies):
            centre = ti.Vector(
                [body_centre[body_idx, 0], body_centre[body_idx, 1], body_centre[body_idx, 2]]
            )
            r = point - centre
            distance = r.norm()
            use_far = (
                body_count[body_idx] >= min_panels
                and body_radius[body_idx] > 0.0
                and distance > acceptance * body_radius[body_idx]
            )
            if use_far:
                inv_r = 1.0 / ti.max(distance, PANEL_EPSILON)
                inv_r3 = inv_r * inv_r * inv_r
                inv_r5 = inv_r3 * inv_r * inv_r
                dipole = ti.Vector(
                    [
                        body_dipole[body_idx, 0],
                        body_dipole[body_idx, 1],
                        body_dipole[body_idx, 2],
                    ]
                )
                value += (
                    body_monopole[body_idx] * r * inv_r3
                    + 3.0 * r * dipole.dot(r) * inv_r5
                    - dipole * inv_r3
                ) / (4.0 * 3.141592653589793)
                far_interactions += 1
            else:
                start = body_start[body_idx]
                for offset in range(body_count[body_idx]):
                    j = start + offset
                    value += source_strength[j] * compute_source_velocity(
                        point,
                        vertex_position[j, 0],
                        vertex_position[j, 1],
                        vertex_position[j, 2],
                        normal[j],
                    )
        target_velocity[i] += value
    return far_interactions


@ti.kernel
def compute_source_panel_velocity_with_far_field(
    vertex_position: ti.types.ndarray(ndim=3),
    normal: ti.types.ndarray(ndim=2),
    source_strength: ti.types.ndarray(ndim=1),
    target_position: ti.types.ndarray(ndim=2),
    target_velocity: ti.types.ndarray(ndim=2),
    body_start: ti.types.ndarray(ndim=1),
    body_count: ti.types.ndarray(ndim=1),
    body_centre: ti.types.ndarray(ndim=2),
    body_radius: ti.types.ndarray(ndim=1),
    body_monopole: ti.types.ndarray(ndim=1),
    body_dipole: ti.types.ndarray(ndim=2),
    n_bodies: ti.i32,
    acceptance: ti.f64,
    min_panels: ti.i32,
) -> ti.i32:
    """Numpy-target counterpart of the device-resident evaluator."""
    far_interactions = 0
    for i in range(target_position.shape[0]):
        point = ti.Vector([target_position[i, 0], target_position[i, 1], target_position[i, 2]])
        value = point * 0.0
        for body_idx in range(n_bodies):
            centre = ti.Vector(
                [body_centre[body_idx, 0], body_centre[body_idx, 1], body_centre[body_idx, 2]]
            )
            r = point - centre
            distance = r.norm()
            use_far = (
                body_count[body_idx] >= min_panels
                and body_radius[body_idx] > 0.0
                and distance > acceptance * body_radius[body_idx]
            )
            if use_far:
                inv_r = 1.0 / ti.max(distance, PANEL_EPSILON)
                inv_r3 = inv_r * inv_r * inv_r
                inv_r5 = inv_r3 * inv_r * inv_r
                dipole = ti.Vector(
                    [
                        body_dipole[body_idx, 0],
                        body_dipole[body_idx, 1],
                        body_dipole[body_idx, 2],
                    ]
                )
                value += (
                    body_monopole[body_idx] * r * inv_r3
                    + 3.0 * r * dipole.dot(r) * inv_r5
                    - dipole * inv_r3
                ) / (4.0 * 3.141592653589793)
                far_interactions += 1
            else:
                start = body_start[body_idx]
                for offset in range(body_count[body_idx]):
                    j = start + offset
                    v0 = ti.Vector(
                        [
                            vertex_position[j, 0, 0],
                            vertex_position[j, 0, 1],
                            vertex_position[j, 0, 2],
                        ]
                    )
                    v1 = ti.Vector(
                        [
                            vertex_position[j, 1, 0],
                            vertex_position[j, 1, 1],
                            vertex_position[j, 1, 2],
                        ]
                    )
                    v2 = ti.Vector(
                        [
                            vertex_position[j, 2, 0],
                            vertex_position[j, 2, 1],
                            vertex_position[j, 2, 2],
                        ]
                    )
                    panel_normal = ti.Vector([normal[j, 0], normal[j, 1], normal[j, 2]])
                    value += source_strength[j] * compute_source_velocity(
                        point, v0, v1, v2, panel_normal
                    )
        target_velocity[i, 0] = value[0]
        target_velocity[i, 1] = value[1]
        target_velocity[i, 2] = value[2]
    return far_interactions
