"""Conservative refinement of stretched vortex-line elements.

The particle vortex-strength vector is a material line element in a three-
dimensional vortex method.  A particle whose strength has grown too large is
therefore refined along its own vortex strength direction:

* two children receive half the parent vortex strength and particle_volume;
* their position are displaced symmetrically along the vortex-strength vector;
* their Gaussian core radius and material properties are unchanged.

For a displacement parallel to ``vortex_strength`` this construction preserves total
vector vortex strength, total strength variation, linear impulse, and the
kernel-corrected angular impulse algebraically.  It also leaves molecular
diffusion in the core radius untouched.  Unlike a strength limiter, refinement
does not remove stretched vorticity; it adds the Lagrangian degrees of freedom
needed to represent it.

This is the local bisection of Winckelmans (1989), Caltech thesis, p. 89,
using the fixed-core branch. It is distinct from the 1995 A2 grid remeshing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openonda.core.slip_slab_images import fold_into_slab


class FilamentRefinementError(RuntimeError):
    """A requested conservative refinement cannot fit its declared budget."""


@dataclass(frozen=True)
class FilamentRefinementResult:
    """Particle arrays and transfer diagnostics produced by one refinement."""

    position: np.ndarray
    vortex_strength: np.ndarray
    core_radius: np.ndarray
    particle_volume: np.ndarray
    reference_vortex_strength: np.ndarray
    reference_length: np.ndarray
    source_index: np.ndarray
    refined_particles: int
    max_stretch_ratio: float
    vortex_strength_error: float
    linear_impulse_error: float
    angular_impulse_error: float


def particle_moments(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    *,
    angular_core_coefficient: float,
    slip_slab: bool = False,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Return ``(sum vortex_strength, sum|vortex_strength|, I, A)`` for a supported blob kernel.

    Between free-slip planes z = const only the spanwise strength and its
    in-plane moments are invariant; in-plane strength is reversed by the
    images across the planes. ``slip_slab`` keeps the moments of the spanwise
    strength, sum Gamma_z, I_x, I_y and A_z, and returns the other components
    as zero. The variation stays the total strength magnitude.
    """

    position = np.asarray(position, dtype=np.float64)
    vortex_strength = np.asarray(vortex_strength, dtype=np.float64)
    core_radius = np.asarray(core_radius, dtype=np.float64)
    if position.shape != vortex_strength.shape or position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position and vortex strength must both have shape (N, 3)")
    if core_radius.shape != (len(position),):
        raise ValueError("core_radius must have shape (N,)")

    variation = float(np.linalg.norm(vortex_strength, axis=1).sum(dtype=np.float64))
    if slip_slab:
        vortex_strength = vortex_strength * np.array([0.0, 0.0, 1.0])
    total = vortex_strength.sum(axis=0, dtype=np.float64)
    impulse = 0.5 * np.cross(position, vortex_strength).sum(axis=0, dtype=np.float64)
    angular = np.cross(position, np.cross(position, vortex_strength)).sum(
        axis=0, dtype=np.float64
    ) / 3.0 - angular_core_coefficient * (core_radius[:, None] ** 2 * vortex_strength).sum(
        axis=0, dtype=np.float64
    )
    if slip_slab:
        # A_x and A_y weight the spanwise strength by z, which the planes do not preserve.
        angular[:2] = 0.0
    return total, variation, impulse, angular


def gaussian_particle_moments(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    *,
    slip_slab: bool = False,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Return ``(sum vortex_strength, sum|vortex_strength|, I, A)`` for Gaussian vortex blobs."""
    return particle_moments(
        position,
        vortex_strength,
        core_radius,
        angular_core_coefficient=1.0 / 3.0,
        slip_slab=slip_slab,
    )


def _assert_transform_is_exact(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    before: tuple[np.ndarray, float, np.ndarray, np.ndarray],
    after: tuple[np.ndarray, float, np.ndarray, np.ndarray],
) -> None:
    """Verify the bisection preserved its four invariants to roundoff.

    Bisection is exactly moment-preserving by construction, so any measurable
    drift here is an implementation error rather than a modelling choice; it is
    checked where the transform is built instead of being reported upward.
    """

    impulse_scale = max(
        0.5
        * float(np.linalg.norm(np.cross(position, vortex_strength), axis=1).sum(dtype=np.float64)),
        np.finfo(float).tiny,
    )
    angular_terms = (
        np.cross(position, np.cross(position, vortex_strength)) / 3.0
        - core_radius[:, None] ** 2 * vortex_strength / 3.0
    )
    angular_scale = max(
        float(np.linalg.norm(angular_terms, axis=1).sum(dtype=np.float64)),
        np.finfo(float).tiny,
    )
    allowance = 512.0 * np.finfo(float).eps
    checks = (
        ("vector vortex strength", float(np.linalg.norm(after[0] - before[0])), before[1]),
        ("total strength variation", abs(after[1] - before[1]), before[1]),
        ("linear impulse", float(np.linalg.norm(after[2] - before[2])), impulse_scale),
        ("angular impulse", float(np.linalg.norm(after[3] - before[3])), angular_scale),
    )
    for name, error, scale in checks:
        limit = allowance * max(float(scale), np.finfo(float).tiny)
        if error > limit:
            raise FilamentRefinementError(
                f"filament refinement changed {name} by {error:.3e}, beyond its "
                f"roundoff allowance {limit:.3e}"
            )


def split_stretched_filaments(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    particle_volume: np.ndarray,
    *,
    reference_vortex_strength: np.ndarray,
    reference_length: np.ndarray,
    max_stretch_factor: float,
    offset_fraction: float = 0.25,
    max_n_particles: int | None = None,
    max_absolute_vortex_strength: float | None = None,
    slip_slab: tuple[float, float] | None = None,
) -> FilamentRefinementResult:
    """Bisect over-stretched particles along their vortex-line direction.

    ``reference_vortex_strength`` and ``reference_length`` describe each particle at
    its most recent reference time (initialization, or the time its parent was
    split).  Since a vortex-particle strength vector is a material line
    element, ``|vortex_strength| / |vortex_strength_ref|`` estimates its line-stretch ratio.  The
    children are placed at the quarter points of that estimated material line
    by the default ``offset_fraction=0.25``.  Each child then receives its own
    current strength and half-line length as the next reference state.

    A call performs one bisection per selected parent. Delayed checks can leave
    long under-resolved segments; splitting is not a substitute for a resolved
    time step. Every eligible parent must fit within the hard particle capacity.

    Between ``slip_slab`` planes a child displaced across a plane is replaced
    by its image inside the slab, which carries the same field. The reported
    moment errors are those of the bisection itself, before that exact fold.
    """

    position = np.asarray(position, dtype=np.float64)
    vortex_strength = np.asarray(vortex_strength, dtype=np.float64)
    core_radius = np.asarray(core_radius, dtype=np.float64)
    particle_volume = np.asarray(particle_volume, dtype=np.float64)
    reference_vortex_strength = np.asarray(reference_vortex_strength, dtype=np.float64)
    reference_length = np.asarray(reference_length, dtype=np.float64)
    if position.shape != vortex_strength.shape or position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position and vortex strength must both have shape (N, 3)")
    if core_radius.shape != (len(position),) or particle_volume.shape != (len(position),):
        raise ValueError("core_radius and particle_volume must both have shape (N,)")
    if reference_vortex_strength.shape != (len(position),) or reference_length.shape != (
        len(position),
    ):
        raise ValueError("reference_vortex_strength and reference_length must both have shape (N,)")
    if np.isnan(max_stretch_factor) or max_stretch_factor <= 1.0:
        raise ValueError("max_stretch_factor must be greater than one")
    if max_absolute_vortex_strength is not None and (
        not np.isfinite(max_absolute_vortex_strength) or max_absolute_vortex_strength <= 0.0
    ):
        raise ValueError("max_absolute_vortex_strength must be finite and positive")
    if not 0.0 <= offset_fraction <= 0.5:
        raise ValueError("offset_fraction must be in [0, 0.5]")
    if (
        np.any(core_radius <= 0.0)
        or np.any(particle_volume <= 0.0)
        or np.any(reference_vortex_strength <= 0.0)
        or np.any(reference_length <= 0.0)
    ):
        raise ValueError(
            "core_radius, particle_volume, reference_vortex_strength, and "
            "reference_length must be positive"
        )
    if not (
        np.isfinite(position).all()
        and np.isfinite(vortex_strength).all()
        and np.isfinite(core_radius).all()
        and np.isfinite(particle_volume).all()
        and np.isfinite(reference_vortex_strength).all()
        and np.isfinite(reference_length).all()
    ):
        raise ValueError("particle arrays must be finite")

    magnitude = np.linalg.norm(vortex_strength, axis=1)
    stretch_ratio = magnitude / reference_vortex_strength
    risk = stretch_ratio / max_stretch_factor
    if max_absolute_vortex_strength is not None:
        risk = np.maximum(risk, magnitude / max_absolute_vortex_strength)
    eligible = np.flatnonzero(risk >= 1.0)
    if max_n_particles is not None:
        required = len(position) + len(eligible)
        if required > max_n_particles:
            raise FilamentRefinementError(
                f"Filament refinement requires {required} particles, exceeding "
                f"the hard capacity {max_n_particles}"
            )
    selected = eligible
    refined_count = len(eligible)

    if refined_count == 0:
        source = np.arange(len(position), dtype=np.int64)
        return FilamentRefinementResult(
            position=position.copy(),
            vortex_strength=vortex_strength.copy(),
            core_radius=core_radius.copy(),
            particle_volume=particle_volume.copy(),
            reference_vortex_strength=reference_vortex_strength.copy(),
            reference_length=reference_length.copy(),
            source_index=source,
            refined_particles=0,
            max_stretch_ratio=(float(stretch_ratio.max()) if len(stretch_ratio) else 0.0),
            vortex_strength_error=0.0,
            linear_impulse_error=0.0,
            angular_impulse_error=0.0,
        )

    retained_mask = np.ones(len(position), dtype=bool)
    retained_mask[selected] = False
    retained = np.flatnonzero(retained_mask)
    direction = vortex_strength[selected] / magnitude[selected, None]
    current_line_length = reference_length[selected] * stretch_ratio[selected]
    displacement_magnitude = offset_fraction * current_line_length
    displacement = displacement_magnitude[:, None] * direction

    new_position = np.concatenate(
        (
            position[retained],
            position[selected] + displacement,
            position[selected] - displacement,
        ),
        axis=0,
    )
    new_vortex_strength = np.concatenate(
        (
            vortex_strength[retained],
            0.5 * vortex_strength[selected],
            0.5 * vortex_strength[selected],
        ),
        axis=0,
    )
    new_core_radius = np.concatenate(
        (core_radius[retained], core_radius[selected], core_radius[selected])
    )
    new_particle_volume = np.concatenate(
        (
            particle_volume[retained],
            0.5 * particle_volume[selected],
            0.5 * particle_volume[selected],
        )
    )
    child_reference_vortex_strength = 0.5 * magnitude[selected]
    child_reference_length = 0.5 * current_line_length
    new_reference_vortex_strength = np.concatenate(
        (
            reference_vortex_strength[retained],
            child_reference_vortex_strength,
            child_reference_vortex_strength,
        )
    )
    new_reference_length = np.concatenate(
        (
            reference_length[retained],
            child_reference_length,
            child_reference_length,
        )
    )
    source = np.concatenate((retained, selected, selected)).astype(np.int64, copy=False)

    before = gaussian_particle_moments(position, vortex_strength, core_radius)
    after = gaussian_particle_moments(new_position, new_vortex_strength, new_core_radius)
    _assert_transform_is_exact(position, vortex_strength, core_radius, before, after)
    if slip_slab is not None:
        new_position, new_vortex_strength = fold_into_slab(
            new_position, new_vortex_strength, slip_slab
        )
    return FilamentRefinementResult(
        position=new_position,
        vortex_strength=new_vortex_strength,
        core_radius=new_core_radius,
        particle_volume=new_particle_volume,
        reference_vortex_strength=new_reference_vortex_strength,
        reference_length=new_reference_length,
        source_index=source,
        refined_particles=refined_count,
        max_stretch_ratio=float(stretch_ratio.max()),
        vortex_strength_error=float(np.linalg.norm(after[0] - before[0])),
        linear_impulse_error=float(np.linalg.norm(after[2] - before[2])),
        angular_impulse_error=float(np.linalg.norm(after[3] - before[3])),
    )


__all__ = [
    "FilamentRefinementError",
    "FilamentRefinementResult",
    "gaussian_particle_moments",
    "split_stretched_filaments",
]
