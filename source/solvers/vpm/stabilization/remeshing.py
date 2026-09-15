"""Gaussian core reset by convolution, without advancing physical diffusion.

For the kernel exp(-r²/sigma²), zeta_sigma = zeta_s *
zeta_sqrt(sigma²-s²). Deposit strengths with M4', convolve with the missing
Gaussian variance, and retain cores s. This is a particle representation
change, not an additional viscous time step. Width interpolation is linear
in sigma² between ``core_bins`` convolution grids. The sampled Gaussian
convolution adds grid error, especially for sub-grid transfer widths; the
field audit must qualify the resulting representation change. Global moment
restoration belongs to the caller.
"""

from __future__ import annotations

import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from ..numerics.fourier_integrals import _grid_for_particles, _scatter_vortex_strength_m4


def project_grid_strength(field: np.ndarray, spacing: float) -> np.ndarray:
    """Periodic Helmholtz projection of a Cartesian strength field.

    Gaussian convolution commutes with this projection. The curl and hence
    induced velocity are unchanged on the complete grid; finite padding,
    particle pruning and subsequent moment correction remain approximations.
    """
    from .divergence_relaxation import _wave_numbers

    spectrum = [fft.rfftn(field[..., axis], workers=-1) for axis in range(3)]
    wave = _wave_numbers(field.shape[:3], spacing)
    norm = sum(k * k for k in wave)
    dot = sum(k * a for k, a in zip(wave, spectrum, strict=True))
    factor = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
    return np.stack(
        [
            fft.irfftn(a - k * factor, s=field.shape[:3], workers=-1)
            for k, a in zip(wave, spectrum, strict=True)
        ],
        axis=-1,
    )


def gaussian_core_remesh(
    particles,
    *,
    spacing: float,
    core_radius: float,
    tail_budget: float,
    max_particles: int | None,
    core_bins: int = 16,
    max_grid_nodes: int = 8_000_000,
    solenoidal: bool = False,
    preserve_groups: bool = False,
) -> dict[str, np.ndarray]:
    """Return a bounded Gaussian remap, supporting LES-dependent source cores.

    The discarded fraction of total node-strength magnitude is bounded by tail_budget.
    A particle cap never silently overrides that accuracy budget. Labels and
    material properties use nearest-source inheritance; labels cease to be
    material tracers once different rings occupy the same remeshing cell.
    With ``preserve_groups``, each (group_id, zone_id) contribution is remapped
    separately, including coincident particles where contributions overlap.
    The tail budget applies to each contribution and the cap to their total.
    Material properties then use nearest-source inheritance within that label.
    """
    if preserve_groups:
        from types import SimpleNamespace

        names = (
            "position",
            "vortex_strength",
            "core_radius",
            "kinematic_viscosity",
            "eddy_viscosity",
            "group_id",
            "zone_id",
        )
        arrays = {name: getattr(particles, name + "_cpu")() for name in names}
        labels = np.column_stack((arrays["group_id"], arrays["zone_id"]))
        proposals = []
        count = 0
        for label in np.unique(labels, axis=0):
            selected = np.all(labels == label, axis=1)
            if not np.any(arrays["vortex_strength"][selected]):
                continue
            subset = SimpleNamespace(
                **{
                    name + "_cpu": lambda values=values[selected]: values
                    for name, values in arrays.items()
                }
            )
            proposal = gaussian_core_remesh(
                subset,
                spacing=spacing,
                core_radius=core_radius,
                tail_budget=tail_budget,
                max_particles=max_particles,
                core_bins=core_bins,
                max_grid_nodes=max_grid_nodes,
                solenoidal=solenoidal,
            )
            count += len(proposal["position"])
            if max_particles is not None and count > max_particles:
                raise ValueError(
                    f"Group-preserving Gaussian remeshing requires at least {count} particles "
                    f"for tail budget {tail_budget:g}; capacity is {max_particles}"
                )
            proposals.append(proposal)
        if not proposals:
            raise ValueError("Gaussian remeshing produced an empty field")
        return {name: np.concatenate([item[name] for item in proposals]) for name in proposals[0]}

    position = particles.position_cpu().astype(np.float64)
    strength = particles.vortex_strength_cpu().astype(np.float64)
    stored_sigma = particles.core_radius_cpu()
    sigma = stored_sigma.astype(np.float64)
    if spacing <= 0 or core_radius <= 0 or core_bins < 2 or not 0 < tail_budget < 1:
        raise ValueError("remeshing requires positive spacing/core, >=2 bins and 0<tail_budget<1")
    variance = sigma**2 - core_radius**2
    roundoff = 8 * np.finfo(stored_sigma.dtype).eps * max(float(np.max(sigma**2)), core_radius**2)
    if np.any(variance < -roundoff):
        raise ValueError("Gaussian core reset cannot enlarge a source core without filtering")
    variance = np.maximum(variance, 0.0)
    padding = 3 + int(np.ceil(4.0 * np.sqrt(variance.max(initial=0.0)) / spacing))
    grid = _grid_for_particles(position, spacing, padding=padding)
    if np.prod(grid.shape, dtype=np.int64) > max_grid_nodes:
        raise ValueError(f"Gaussian remeshing grid {grid.shape} exceeds {max_grid_nodes} nodes")
    field = np.zeros((*grid.shape, 3), dtype=np.float64)
    lo, hi = float(variance.min()), float(variance.max())
    widths = np.linspace(lo, hi, core_bins) if hi > lo + 1e-14 else np.array([lo])
    if len(widths) > 1:
        coordinate = (variance - lo) / (hi - lo) * (core_bins - 1)
        lower = np.minimum(np.floor(coordinate).astype(int), core_bins - 2)
        fraction = coordinate - lower
    for index, width in enumerate(widths):
        weight = (
            np.ones(len(position))
            if len(widths) == 1
            else np.where(lower == index, 1.0 - fraction, 0.0)
            + np.where(lower + 1 == index, fraction, 0.0)
        )
        selected = weight > 0
        if not np.any(selected):
            continue
        deposited = _scatter_vortex_strength_m4(
            position[selected], strength[selected] * weight[selected, None], grid
        )
        # scipy uses standard deviation; the vortex Gaussian uses sqrt(2)*std.
        std = np.sqrt(width / 2.0) / spacing
        if std > 1e-12:
            for component in range(3):
                deposited[..., component] = gaussian_filter(
                    deposited[..., component], std, mode="constant", truncate=5.0
                )
        field += deposited
    if solenoidal:
        from ..numerics.fourier_integrals import CartesianGrid

        padding = tuple((size // 2, size - size // 2) for size in grid.shape)
        shape = tuple(2 * size for size in grid.shape)
        if np.prod(shape, dtype=np.int64) > max_grid_nodes:
            raise ValueError("solenoidal remeshing padding exceeds the grid-node limit")
        field = project_grid_strength(np.pad(field, (*padding, (0, 0))), spacing)
        grid = CartesianGrid(
            grid.origin - spacing * np.array([p[0] for p in padding]), spacing, shape
        )
    flat = field.reshape(-1, 3)
    magnitude = np.linalg.norm(flat, axis=1)
    nonzero = np.flatnonzero(magnitude > 0)
    order = nonzero[np.argsort(magnitude[nonzero])[::-1]]
    cumulative = np.cumsum(magnitude[order])
    if not len(order) or cumulative[-1] <= 0:
        raise ValueError("Gaussian remeshing produced an empty field")
    count = int(np.searchsorted(cumulative, (1.0 - tail_budget) * cumulative[-1])) + 1
    if max_particles is not None and count > max_particles:
        raise ValueError(
            f"Gaussian remeshing requires {count} particles for tail budget {tail_budget:g}; "
            f"capacity is {max_particles}"
        )
    keep = np.sort(order[:count])
    coordinates = np.column_stack(np.unravel_index(keep, grid.shape))
    new_position = grid.origin + spacing * coordinates
    nearest = cKDTree(position, compact_nodes=False).query(new_position)[1]
    return {
        "position": new_position,
        "vortex_strength": flat[keep],
        "core_radius": np.full(count, core_radius),
        "particle_volume": np.full(count, spacing**3),
        "velocity": np.zeros((count, 3)),
        "kinematic_viscosity": particles.kinematic_viscosity_cpu()[nearest],
        "eddy_viscosity": particles.eddy_viscosity_cpu()[nearest],
        "group_id": particles.group_id_cpu()[nearest],
        "zone_id": particles.zone_id_cpu()[nearest],
    }
