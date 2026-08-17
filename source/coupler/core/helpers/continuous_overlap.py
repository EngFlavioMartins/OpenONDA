"""Conservative overlap transport from FVM cells to VPM particles."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np

from source.coupler.core.helpers.fvm_velocity_trace import CachedVelocityTrace
from source.coupler.diagnostics.injection_correction import recover_invariants
from source.coupler.remesh import remesh_to_grid

logger = logging.getLogger("coupler")

# Particle core radius relative to the regular overlap lattice spacing.
RADIUS_RATIO = 1.0

# M4-prime kernel support half-width in lattice cells.
_M4P_SUPPORT = 2.0

# Cap on the inverse-mollification amplification |Gamma| / |Gamma_raw|.
DEFAULT_TRANSFER_AMPLIFICATION_CAP = 2.0

# Cosine taper width, in cells, at the outer lattice faces: the transform is
# periodic, so the field must decay rather than wrap.
_FFT_EDGE_TAPER_CELLS = 2


def _invariants(pos: np.ndarray, circ: np.ndarray) -> dict[str, np.ndarray]:
    """Circulation, linear impulse and raw angular impulse (Winckelmans 1993).

    Kernel-correction-free: the sigma^2 term cancels across the prune.
    """
    if len(pos) == 0:
        z = np.zeros(3)
        return {"circulation": z, "linear_impulse": z, "angular_impulse": z}
    r_x_g = np.cross(pos, circ)
    return {
        "circulation": np.sum(circ, axis=0),
        "linear_impulse": 0.5 * np.sum(r_x_g, axis=0),
        "angular_impulse": (1.0 / 3.0) * np.sum(np.cross(pos, r_x_g), axis=0),
    }


def required_buffer_length(u_max: float, dt: float, h: float, safety: float = 1.5) -> float:
    """Minimum downstream buffer length for a dt-robust hand-off:
    ``L_buf >= safety * u_max * dt + 2h`` (M4' stencil must stay interior)."""
    return float(safety * abs(u_max) * abs(dt) + _M4P_SUPPORT * h)


def max_stable_dt(u_max: float, l_buf: float, h: float, safety: float = 1.5) -> float:
    """Largest ``dt`` for which the given buffer keeps the hand-off exact
    (inverse of :func:`required_buffer_length`)."""
    u = abs(u_max)
    if u < 1e-30:
        return float("inf")
    return float(max(l_buf - _M4P_SUPPORT * h, 0.0) / (safety * u))


def circulation_from_velocity_trace(
    positions: np.ndarray,
    h: float,
    velocity_at,
) -> np.ndarray:
    """Integrate ``n x u`` over each cubic particle control volume."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    circulation = np.zeros_like(pos)
    offset = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        offset.fill(0.0)
        offset[axis] = 0.5 * h
        delta_u = np.asarray(velocity_at(pos + offset), dtype=np.float64) - np.asarray(
            velocity_at(pos - offset), dtype=np.float64
        )
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
        circulation += h**2 * np.cross(normal, delta_u)
    return circulation


def smoothstep(x: np.ndarray | float, lo: float, hi: float) -> np.ndarray:
    """C1 Hermite ramp, zero slope at both ends.

    Adds no grid-scale content to whatever it multiplies.
    """
    span = float(hi) - float(lo)
    if abs(span) < 1e-300:
        return np.where(np.asarray(x, dtype=np.float64) >= hi, 1.0, 0.0)
    t = np.clip((np.asarray(x, dtype=np.float64) - float(lo)) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _outflow_axis_sign(u_inf=None) -> tuple[int, float]:
    """Return the box axis/sign most aligned with the freestream."""
    axis, sign = 0, +1.0
    if u_inf is not None:
        u = np.asarray(u_inf, dtype=np.float64).reshape(-1)
        if u.size == 3 and np.any(u != 0.0):
            axis = int(np.argmax(np.abs(u)))
            sign = float(np.sign(u[axis]))
    return axis, sign


def _outflow_band_mask(
    grid_pos: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    h: float,
    u_inf=None,
) -> np.ndarray:
    """Boolean mask of the downstream-most h-layer of the box."""
    axis, sign = _outflow_axis_sign(u_inf)
    if sign >= 0:
        return grid_pos[:, axis] >= hi[axis] - h
    return grid_pos[:, axis] <= lo[axis] + h


def cosine_eta(
    grid_pos: np.ndarray,
    box: np.ndarray,
    ramp_width: float,
    dead_zone: float,
) -> np.ndarray:
    """C1 partition-of-unity FVM-authority weight in [0, 1].

    One at ``dist >= ramp_width``, zero at ``dist <= dead_zone`` and outside
    the box, cosine ramp between.
    """
    pos = np.asarray(grid_pos, dtype=np.float64)
    b = np.asarray(box, dtype=np.float64)
    dist = np.minimum.reduce(
        [
            pos[:, 0] - b[0],
            b[1] - pos[:, 0],
            pos[:, 1] - b[2],
            b[3] - pos[:, 1],
            pos[:, 2] - b[4],
            b[5] - pos[:, 2],
        ]
    )
    eta = np.zeros(len(pos), dtype=np.float64)
    if ramp_width <= 0.0:
        eta[dist > 0.0] = 1.0
        return eta

    eta[dist >= ramp_width] = 1.0
    lo = max(dead_zone, 0.0)
    width = max(ramp_width - lo, 1e-30)
    ramp = (dist > lo) & (dist < ramp_width)
    if ramp.any():
        eta[ramp] = 0.5 * (1.0 - np.cos(np.pi * (dist[ramp] - lo) / width))
    return eta


# =========================================================
# Band-limited transfer (replaces the Beale/Picard deconvolution)
# =========================================================
def _wavenumber_grid(shape: tuple[int, int, int], h: float) -> np.ndarray:
    """Squared wavenumber magnitude on the rfftn lattice of ``shape``."""
    kx = 2.0 * np.pi * np.fft.fftfreq(shape[0], d=h)
    ky = 2.0 * np.pi * np.fft.fftfreq(shape[1], d=h)
    kz = 2.0 * np.pi * np.fft.rfftfreq(shape[2], d=h)
    return kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2


def transfer_symbols(
    shape: tuple[int, int, int],
    h: float,
    sigma: float,
    amplification_cap: float = DEFAULT_TRANSFER_AMPLIFICATION_CAP,
) -> tuple[np.ndarray, np.ndarray]:
    """``(W, Phi)`` for the regularised inverse mollification.

    ``Phi(0) = 1`` exactly, else hand-offs leak circulation. ``max W = cap``.
    """
    if not np.isfinite(amplification_cap) or amplification_cap < 1.0:
        raise ValueError("amplification_cap must be finite and at least one")
    cap = float(amplification_cap)
    lam = cap - np.sqrt(cap * cap - 1.0)
    k_sq = _wavenumber_grid(shape, h)
    g = np.exp(-(float(sigma) ** 2) * k_sq / 4.0)
    w = (1.0 + lam * lam) * g / (g * g + lam * lam)
    return w, g * w


def _edge_taper(shape: tuple[int, int, int], cells: int = _FFT_EDGE_TAPER_CELLS) -> np.ndarray:
    """Separable cosine taper over the outermost ``cells`` lattice layers."""
    axes = []
    for n in shape:
        w = np.ones(n, dtype=np.float64)
        m = int(min(cells, max(n // 2, 0)))
        if m > 0:
            ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(m) + 0.5) / m))
            w[:m] = ramp
            w[n - m :] = ramp[::-1]
        axes.append(w)
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]


def bandlimited_transfer(
    target_circ: np.ndarray,
    shape: tuple[int, int, int],
    h: float,
    *,
    sigma: float,
    amplification_cap: float = DEFAULT_TRANSFER_AMPLIFICATION_CAP,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert an FVM lattice circulation into particle strengths.

    ``target_inband`` is the band the lattice carries. ``out_of_band_fraction``
    is a resolution diagnostic: reduce ``h``, not the regularisation.
    """
    grid = np.asarray(target_circ, dtype=np.float64).reshape(*shape, 3)
    taper = _edge_taper(shape)[..., None]
    tapered = grid * taper

    w, phi = transfer_symbols(shape, h, sigma, amplification_cap)
    gamma = np.empty_like(tapered)
    inband = np.empty_like(tapered)
    axes = (0, 1, 2)
    for component in range(3):
        spectrum = np.fft.rfftn(tapered[..., component], axes=axes)
        gamma[..., component] = np.fft.irfftn(spectrum * w, s=shape, axes=axes)
        inband[..., component] = np.fft.irfftn(spectrum * phi, s=shape, axes=axes)

    denominator = float(np.linalg.norm(tapered)) + 1e-30
    out_of_band = float(np.linalg.norm(tapered - inband)) / denominator
    return gamma.reshape(-1, 3), inband.reshape(-1, 3), out_of_band


#: Wavelength bands, in lattice cells, used by the spectral hand-off diagnostic.
SPECTRAL_BANDS: tuple[tuple[float, float], ...] = (
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, 16.0),
    (16.0, 64.0),
)


def spectral_band_ratio(
    particle_field: np.ndarray,
    reference_field: np.ndarray,
    shape: tuple[int, int, int],
    h: float,
    bands: tuple[tuple[float, float], ...] = SPECTRAL_BANDS,
) -> dict[str, float]:
    """Banded ``|omega_VPM(k)| / |omega_FVM(k)|`` on the lattice.

    Both inputs must already be band-limited to the same band.
    """
    left = np.asarray(particle_field, dtype=np.float64).reshape(*shape, 3)
    right = np.asarray(reference_field, dtype=np.float64).reshape(*shape, 3)
    k_sq = _wavenumber_grid(shape, h)
    k = np.sqrt(k_sq)

    numerator = np.zeros(k.shape)
    denominator = np.zeros(k.shape)
    for component in range(3):
        numerator += np.abs(np.fft.rfftn(left[..., component])) ** 2
        denominator += np.abs(np.fft.rfftn(right[..., component])) ** 2

    out: dict[str, float] = {}
    for lo_cells, hi_cells in bands:
        k_lo = 2.0 * np.pi / (hi_cells * h)
        k_hi = 2.0 * np.pi / (lo_cells * h)
        sel = (k >= k_lo) & (k < k_hi)
        if not sel.any():
            continue
        den = float(denominator[sel].sum())
        out[f"{lo_cells:g}-{hi_cells:g}h"] = float(
            np.sqrt(float(numerator[sel].sum()) / den) if den > 0.0 else 0.0
        )
    return out


def _gaussian_mollified_circulation(
    circ_grid: np.ndarray,
    shape: tuple[int, int, int],
    h: float,
    *,
    sigma: float,
) -> np.ndarray:
    """Return circulation represented by Gaussian particles on their lattice."""
    from scipy.ndimage import gaussian_filter

    # s = sigma/sqrt(2) in cells. gaussian_filter normalises to sum(w) = 1,
    # matching the sampled zeta_sigma * h^3 weights for s >~ 1 cell.
    s_cells = float(sigma) / (np.sqrt(2.0) * float(h))
    grid = np.asarray(circ_grid, dtype=np.float64).reshape(*shape, 3)
    return np.stack(
        [
            gaussian_filter(grid[..., component], s_cells, mode="constant", truncate=5.0)
            for component in range(3)
        ],
        axis=-1,
    )


def bounded_local_transfer(
    particle_strength: np.ndarray,
    fvm_target: np.ndarray,
    authority: np.ndarray,
    shape: tuple[int, int, int],
    h: float,
    *,
    sigma: float,
    amplification_cap: float = DEFAULT_TRANSFER_AMPLIFICATION_CAP,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Blend and correct strengths without creating a global FFT tail.

    The FVM/VPM partition is formed in physical-vorticity space. A single
    compact Gaussian residual correction then improves the represented field.
    Its gain is at most two for the production cap, while an exactly zero far
    field remains exactly zero outside the Gaussian stencil.
    """
    if not np.isfinite(amplification_cap) or amplification_cap < 1.0:
        raise ValueError("amplification_cap must be finite and at least one")

    particle_strength = np.asarray(particle_strength, dtype=np.float64).reshape(-1, 3)
    fvm_target = np.asarray(fvm_target, dtype=np.float64).reshape(-1, 3)
    authority = np.asarray(authority, dtype=np.float64).reshape(-1)
    if particle_strength.shape != fvm_target.shape or authority.shape != (len(particle_strength),):
        raise ValueError("local transfer inputs do not share one lattice shape")

    particle_field = _gaussian_mollified_circulation(
        particle_strength, shape, h, sigma=sigma
    ).reshape(-1, 3)
    physical_target = particle_field + authority[:, None] * (fvm_target - particle_field)

    strength = particle_strength + authority[:, None] * (fvm_target - particle_strength)
    represented = _gaussian_mollified_circulation(strength, shape, h, sigma=sigma).reshape(-1, 3)
    residual = physical_target - represented
    denominator = float(np.linalg.norm(physical_target)) + 1.0e-30
    residual_pre = float(np.linalg.norm(residual)) / denominator

    correction_gain = min(float(amplification_cap) - 1.0, 1.0)
    strength = strength + correction_gain * residual
    represented = _gaussian_mollified_circulation(strength, shape, h, sigma=sigma).reshape(-1, 3)
    residual_post = float(np.linalg.norm(physical_target - represented)) / denominator
    raw_norm = float(np.linalg.norm(physical_target, axis=1).max()) + 1.0e-30
    max_amplification = float(np.linalg.norm(strength, axis=1).max()) / raw_norm
    return strength, physical_target, residual_pre, residual_post, max_amplification


# =========================================================
# Continuous (soft) prune with local moment redistribution
# =========================================================
def soft_prune(
    circ: np.ndarray,
    threshold: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative garrote shrinkage of weak lattice strengths.

    ``Gamma * max(0, 1 - (threshold/|Gamma|)^2)``: continuous, and zero below
    the threshold. Returns ``(shrunk, removed)``.
    """
    circ = np.asarray(circ, dtype=np.float64).reshape(-1, 3)
    threshold = np.broadcast_to(np.asarray(threshold, dtype=np.float64), (len(circ),))
    if np.any(~np.isfinite(threshold)) or np.any(threshold < 0.0):
        raise ValueError("prune threshold must be finite and non-negative")
    if not np.any(threshold > 0.0):
        return circ.copy(), np.zeros_like(circ)
    magnitude = np.linalg.norm(circ, axis=1)
    scale = np.zeros_like(magnitude)
    active = magnitude > threshold
    scale[active] = 1.0 - (threshold[active] / magnitude[active]) ** 2
    shrunk = circ * scale[:, None]
    return shrunk, circ - shrunk


def redistribute_locally(
    removed: np.ndarray,
    shrunk: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Push ``removed`` onto surviving face neighbours.

    Equal weights sum to one with zero first moment, so circulation and
    impulse are conserved locally.
    """
    removed = np.asarray(removed, dtype=np.float64).reshape(*shape, 3)
    if not np.any(removed):
        return np.asarray(shrunk, dtype=np.float64).reshape(-1, 3)

    out = np.asarray(shrunk, dtype=np.float64).reshape(*shape, 3).copy()
    alive = np.linalg.norm(out, axis=-1) > 0.0

    shifts = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
    neighbour_alive = [np.roll(alive, -s, axis=a) for a, s in shifts]
    # Do not donate across the periodic seam.
    for index, (axis, step) in enumerate(shifts):
        sl: list[slice | int] = [slice(None)] * 3
        sl[axis] = -1 if step == 1 else 0
        neighbour_alive[index][tuple(sl)] = False

    count = np.sum(neighbour_alive, axis=0).astype(np.float64)
    donatable = count > 0.0
    share = np.zeros_like(removed)
    np.divide(removed, count[..., None], out=share, where=donatable[..., None])

    for index, (axis, step) in enumerate(shifts):
        contribution = np.where(neighbour_alive[index][..., None], share, 0.0)
        out += np.roll(contribution, step, axis=axis)

    # Do not resurrect a broad weak region merely because its nodes have no
    # already-surviving neighbour. The following global invariant recovery
    # moves the small undonated remainder onto the retained physical vortices.
    return out.reshape(-1, 3)


# =========================================================
# Result container
# =========================================================
@dataclass
class HandoffResult:
    """New particle field + per-step diagnostics from one hand-off."""

    pos: np.ndarray
    circ: np.ndarray
    vol: np.ndarray
    rad: np.ndarray

    n_remesh_in: int = 0  # particles fed into the lattice
    n_remesh_out: int = 0  # particles produced from the lattice
    n_free: int = 0  # untouched free-exterior particles
    n_excluded: int = 0  # input particles removed from a physical solid
    n_pruned: int = 0
    pruned_circulation_l1: float = 0.0
    pruned_circulation_fraction: float = 0.0
    n_population_pruned: int = 0
    population_pruned_circulation_fraction: float = 0.0
    population_pruned_velocity_bound: float = 0.0
    cfl: float = 0.0  # U_max*dt / L_buf  (should stay < ~0.7)
    conservation_drift: dict[str, float] = field(default_factory=dict)
    conservation_raw_mismatch: dict[str, float] = field(default_factory=dict)
    conservation_applied_correction: dict[str, float] = field(default_factory=dict)
    conservation_corrected_mismatch: dict[str, float] = field(default_factory=dict)

    # Sum|Gamma_sigma|_VPM / Sum|Gamma|_target over the outflow band.
    # 1.0 means agreement.
    flux_ratio: float = 0.0

    # Historical field names retained for diagnostics-file compatibility.
    # They store the post- and pre-correction representation residuals.
    transfer_in_band_residual: float = 0.0
    transfer_pre_prune_residual: float = 0.0
    transfer_out_of_band_fraction: float = 0.0
    transfer_max_amplification: float = 0.0
    #: Banded |omega_VPM(k)| / |omega_FVM(k)| on the hand-off lattice, keyed by
    #: wavelength range in lattice cells.  All entries should be ~1.
    spectral_band_ratio: dict[str, float] = field(default_factory=dict)

    # Body-mask audit.  L1 sums of |Gamma| removed from input particles, from
    # the VPM remesh support, and from the FVM target.
    excluded_input_circulation_l1: float = 0.0
    excluded_remesh_circulation_l1: float = 0.0
    excluded_target_circulation_l1: float = 0.0

    @property
    def n_total(self) -> int:
        return len(self.pos)


# =========================================================
# Pure-NumPy numerical core
# =========================================================
def continuous_handoff(
    pos: np.ndarray,
    circ: np.ndarray,
    box: np.ndarray | list[float],
    h: float,
    *,
    circulation_at_node,
    u_inf=None,
    mesh_weight_at_node=None,
    fluid_weight_at_node=None,
    interior_at_node=None,
    ramp_width: float | None = None,
    dead_zone: float = 0.0,
    buffer_length: float = 0.0,
    threshold_abs: float = 0.0,
    radius_ratio: float = RADIUS_RATIO,
    amplification_cap: float = DEFAULT_TRANSFER_AMPLIFICATION_CAP,
    boundary_prune_multiplier: float = 1.0,
    u_max: float = 0.0,
    dt: float = 0.0,
    lattice_anchor=None,
    max_output_particles: int | None = None,
) -> HandoffResult:
    """One continuous, conservative, dt-robust FVM->VPM hand-off.

    Parameters
    ----------
    pos, circ : ndarray (N, 3)
        Current particle positions and circulations.
    box : (6,) ``[x0, x1, y0, y1, z0, z1]``
        Hand-off bounds; the interface is the box surface.
    h : float
        Lattice spacing, equal to the VPM particle spacing.
    circulation_at_node : callable
        FVM circulation from the weighted velocity trace.
    u_inf : ndarray (3,) or None
        Freestream, used to orient the outflow diagnostic.
    mesh_weight_at_node : callable ``grid_pos -> (M,) in [0, 1]`` or None
        Smooth confidence that a node has usable FVM data. Must be C1.
    fluid_weight_at_node : callable ``positions -> (M,) in [0, 1]`` or None
        Smooth solid taper: one in the fluid, zero inside a body.
    interior_at_node : callable ``positions -> bool (M,)`` or None
        Exact solid test, a placement guard only.
    ramp_width, dead_zone : float
        Width of the eta ramp and of the eta = 0 band at each face.
    buffer_length : float
        Outward extension of the remesh lattice beyond the box.
    threshold_abs : float
        Soft-prune scale; see :func:`soft_prune`.
    radius_ratio : float
        Particle core radius sigma / h.
    amplification_cap : float
        See :func:`transfer_symbols`.
    boundary_prune_multiplier : float
        Smooth multiplier on the prune scale where FVM authority approaches
        zero at the handoff boundary.
    u_max, dt : float
        CFL diagnostic only.
    max_output_particles : int or None
        Post-handoff population cap.

    Returns
    -------
    HandoffResult
    """
    box = np.asarray(box, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    circ = np.asarray(circ, dtype=np.float64).reshape(-1, 3)
    n = len(pos)

    if ramp_width is None:
        ramp_width = max(2.0 * dead_zone, 4.0 * h)

    lo = np.array([box[0], box[2], box[4]], dtype=np.float64)
    hi = np.array([box[1], box[3], box[5]], dtype=np.float64)

    # Active region = box (+) buffer_length. Particles here are remeshed;
    # beyond it they are free Lagrangian. A 2h guard keeps M4' stencils inside.
    guard = _M4P_SUPPORT * h
    lo_active = lo - buffer_length
    hi_active = hi + buffer_length
    lo_lat = lo_active - guard
    hi_lat = hi_active + guard

    # Keep the remesh lattice aligned with the FVM lattice.
    if lattice_anchor is not None:
        anchor = np.asarray(lattice_anchor, dtype=np.float64).reshape(3)
        shift = np.floor((lo_lat - anchor) / float(h))
        lo_lat = anchor + shift * float(h)

    # ---- Input particles: taper by solid weight, drop true interior ----
    if n > 0:
        input_fluid = (
            np.ones(n)
            if fluid_weight_at_node is None
            else np.clip(
                np.asarray(fluid_weight_at_node(pos), dtype=np.float64).reshape(-1), 0.0, 1.0
            )
        )
        if input_fluid.shape != (n,):
            raise ValueError(f"fluid_weight_at_node returned {input_fluid.shape}, expected ({n},)")
        excluded_input_l1 = float(np.linalg.norm(circ * (1.0 - input_fluid)[:, None], axis=1).sum())
        circ = circ * input_fluid[:, None]
        deep_solid = (
            np.zeros(n, dtype=bool)
            if interior_at_node is None
            else np.asarray(interior_at_node(pos), dtype=bool).reshape(-1)
        )
        valid_input = ~deep_solid
        in_region = valid_input & np.all((pos >= lo_active) & (pos <= hi_active), axis=1)
    else:
        excluded_input_l1 = 0.0
        deep_solid = np.zeros(0, dtype=bool)
        valid_input = np.zeros(0, dtype=bool)
        in_region = np.zeros(0, dtype=bool)
    free_mask = valid_input & ~in_region

    shape = (
        int(np.ceil((hi_lat[0] - lo_lat[0]) / h)) + 1,
        int(np.ceil((hi_lat[1] - lo_lat[1]) / h)) + 1,
        int(np.ceil((hi_lat[2] - lo_lat[2]) / h)) + 1,
    )

    # ---- P2M: conservative M4' scatter of the in-region wake onto the lattice
    grid_pos, grid_circ = remesh_to_grid(pos[in_region], circ[in_region], lo_lat, h, shape)

    # ---- Smooth weights on the lattice --------------------------------------
    node_fluid = (
        np.ones(len(grid_pos))
        if fluid_weight_at_node is None
        else np.clip(
            np.asarray(fluid_weight_at_node(grid_pos), dtype=np.float64).reshape(-1), 0.0, 1.0
        )
    )
    if node_fluid.shape != (len(grid_pos),):
        raise ValueError(
            f"fluid_weight_at_node returned {node_fluid.shape}, expected ({len(grid_pos)},)"
        )
    excluded_remesh_l1 = float(
        np.linalg.norm(grid_circ * (1.0 - node_fluid)[:, None], axis=1).sum()
    )
    grid_circ = grid_circ * node_fluid[:, None]

    node_mesh = (
        np.ones(len(grid_pos))
        if mesh_weight_at_node is None
        else np.clip(
            np.asarray(mesh_weight_at_node(grid_pos), dtype=np.float64).reshape(-1), 0.0, 1.0
        )
    )
    if node_mesh.shape != (len(grid_pos),):
        raise ValueError(
            f"mesh_weight_at_node returned {node_mesh.shape}, expected ({len(grid_pos)},)"
        )

    # ---- FVM target, tapered smoothly to zero before any spectral work ------
    target_raw = np.asarray(circulation_at_node(grid_pos), dtype=np.float64).reshape(-1, 3)
    excluded_target_l1 = float(
        np.linalg.norm(target_raw * (1.0 - node_fluid)[:, None], axis=1).sum()
    )
    # Both tapers are C1, so the target has no step for the transfer to ring on.
    target = target_raw * (node_fluid * node_mesh)[:, None]

    # ---- eta partition of unity (FVM authority), smoothly gated -------------
    eta = cosine_eta(grid_pos, box, ramp_width, dead_zone) * node_mesh

    # Blend the physical field, then apply one bounded local correction. A
    # global inverse is mathematically nonlocal and filled the complete guarded
    # cubeFlow lattice with weak particles (678k after the first handoff).
    sigma = float(radius_ratio) * float(h)
    grid_blended, physical_target, transfer_pre, _transfer_post, max_amplification = (
        bounded_local_transfer(
            grid_circ,
            target,
            eta,
            shape,
            h,
            sigma=sigma,
            amplification_cap=amplification_cap,
        )
    )
    grid_blended = grid_blended * node_fluid[:, None]

    # Representation before pruning.  This is useful while constructing the
    # field, but it is not the state that is ultimately handed to the VPM.
    mollified_pre_prune = _gaussian_mollified_circulation(
        grid_blended, shape, h, sigma=sigma
    ).reshape(-1, 3)
    reference = physical_target
    comparison_weight = node_fluid * node_mesh
    pre_prune_residual = float(
        np.linalg.norm((mollified_pre_prune - reference) * comparison_weight[:, None])
    ) / (float(np.linalg.norm(reference * comparison_weight[:, None])) + 1e-30)
    out_of_band = transfer_pre

    # ---- Continuous prune with local moment redistribution ------------------
    target_inv = _invariants(grid_pos, grid_blended)
    magnitude_before = np.linalg.norm(grid_blended, axis=1)
    local_threshold = float(threshold_abs) * (
        1.0 + (float(boundary_prune_multiplier) - 1.0) * (1.0 - eta)
    )
    shrunk, removed = soft_prune(grid_blended, local_threshold)
    shrunk = redistribute_locally(removed, shrunk, shape)
    if interior_at_node is not None:
        deep_nodes = np.asarray(interior_at_node(grid_pos), dtype=bool).reshape(-1)
        shrunk[deep_nodes] = 0.0

    keep = np.linalg.norm(shrunk, axis=1) > 0.0
    pruned = (magnitude_before > 0.0) & ~keep
    active_l1 = float(magnitude_before[magnitude_before > 0.0].sum())
    pruned_l1 = float(magnitude_before[pruned].sum())

    new_pos = grid_pos[keep]
    new_circ = shrunk[keep]

    pre_correction = _invariants(new_pos, new_circ)
    post_correction = pre_correction

    def _invariant_norms(left: dict[str, np.ndarray], right: dict[str, np.ndarray]):
        return {
            name: float(np.linalg.norm(left[name] - right[name]))
            for name in ("circulation", "linear_impulse", "angular_impulse")
        }

    raw_mismatch = _invariant_norms(target_inv, pre_correction)
    applied_correction = _invariant_norms(pre_correction, post_correction)
    corrected_mismatch = _invariant_norms(target_inv, post_correction)
    drift: dict[str, float] = {}
    if len(new_pos) > 1:
        # Final global closure. A large correction means the local scatter failed.
        new_circ = recover_invariants(
            new_pos, new_circ, target_inv, volumes=np.full(len(new_pos), h**3)
        )
        post_correction = _invariants(new_pos, new_circ)
        applied_correction = _invariant_norms(pre_correction, post_correction)
        corrected_mismatch = _invariant_norms(target_inv, post_correction)
        ref = float(np.linalg.norm(target_inv["circulation"])) + 1e-30
        correction_scale = active_l1 + 1e-30
        drift = {
            **corrected_mismatch,
            "circulation_rel": float(corrected_mismatch["circulation"] / ref),
        }
        if applied_correction["circulation"] > 1.0e-2 * correction_scale:
            logger.warning(
                "[Handoff] global invariant closure had to move %.3e of circulation "
                "(%.2f%% of Sum|Gamma|); the local prune redistribution is not "
                "absorbing the pruned moments.",
                applied_correction["circulation"],
                100.0 * applied_correction["circulation"] / correction_scale,
            )

    # Audit the field that is actually returned.  The previous diagnostic was
    # evaluated before soft pruning and global invariant recovery, so it could
    # report an acceptable handoff even when those operations had subsequently
    # removed or redistributed the resolved vorticity.
    final_strength = np.zeros_like(grid_blended)
    final_strength[keep] = new_circ
    mollified = _gaussian_mollified_circulation(final_strength, shape, h, sigma=sigma).reshape(
        -1, 3
    )
    in_band_residual = float(
        np.linalg.norm((mollified - reference) * comparison_weight[:, None])
    ) / (float(np.linalg.norm(reference * comparison_weight[:, None])) + 1e-30)

    new_vol = np.full(len(new_pos), h**3)
    new_rad = np.full(len(new_pos), h * radius_ratio)

    # ---- Re-attach the free-exterior particles unchanged --------------------
    if free_mask.any():
        out_pos = np.vstack([new_pos, pos[free_mask]])
        out_circ = np.vstack([new_circ, circ[free_mask]])
        out_vol = np.concatenate([new_vol, np.full(int(free_mask.sum()), h**3)])
        out_rad = np.concatenate([new_rad, np.full(int(free_mask.sum()), h * radius_ratio)])
    else:
        out_pos, out_circ, out_vol, out_rad = new_pos, new_circ, new_vol, new_rad

    population_pruned = 0
    population_pruned_fraction = 0.0
    population_pruned_velocity_bound = 0.0
    if max_output_particles is not None and len(out_pos) > max_output_particles:
        target_count = int(max_output_particles)
        combined_target = _invariants(out_pos, out_circ)
        combined_mag = np.linalg.norm(out_circ, axis=1)

        n_remesh = len(new_pos)
        n_free = len(out_pos) - n_remesh
        if n_free < target_count:
            remesh_budget = target_count - n_free
            remesh_keep = np.argpartition(combined_mag[:n_remesh], -remesh_budget)[-remesh_budget:]
            keep_indices = np.concatenate(
                [remesh_keep, np.arange(n_remesh, len(out_pos), dtype=np.int64)]
            )
        else:
            keep_indices = np.argpartition(combined_mag[n_remesh:], -target_count)[-target_count:]
            keep_indices = keep_indices + n_remesh

        keep_mask = np.zeros(len(out_pos), dtype=bool)
        keep_mask[keep_indices] = True
        discarded_l1 = float(combined_mag[~keep_mask].sum())
        delta = np.maximum(np.maximum(lo - out_pos, out_pos - hi), 0.0)
        distance_sq = np.einsum("ij,ij->i", delta, delta)
        population_pruned_velocity_bound = float(
            np.sum(
                combined_mag[~keep_mask]
                / (
                    4.0
                    * np.pi
                    * np.maximum(distance_sq[~keep_mask] + out_rad[~keep_mask] ** 2, 1.0e-30)
                )
            )
        )
        population_pruned_fraction = discarded_l1 / (float(combined_mag.sum()) + 1e-30)
        population_pruned = int((~keep_mask).sum())

        out_pos = out_pos[keep_indices]
        out_circ = out_circ[keep_indices]
        out_vol = out_vol[keep_indices]
        out_rad = out_rad[keep_indices]
        out_circ = recover_invariants(out_pos, out_circ, combined_target, volumes=out_vol)

    # ---- Diagnostics --------------------------------------------------------
    cfl = float(abs(u_max) * abs(dt) / (buffer_length + 1e-30))

    # Outflow-band physical-field content ratio; 1.0 means agreement.
    flux_ratio = 0.0
    if len(grid_pos) > 0:
        band = _outflow_band_mask(grid_pos, lo, hi, h, u_inf)
        if band.any():
            g_vpm = float(np.linalg.norm(mollified[band], axis=1).sum())
            g_fvm = float(np.linalg.norm(reference[band], axis=1).sum())
            flux_ratio = float(g_vpm / (g_fvm + 1e-30))

    bands = spectral_band_ratio(mollified, reference, shape, h)

    return HandoffResult(
        pos=out_pos,
        circ=out_circ,
        vol=out_vol,
        rad=out_rad,
        n_remesh_in=int(in_region.sum()),
        n_remesh_out=int(len(new_pos)),
        n_free=int(free_mask.sum()),
        n_excluded=int(deep_solid.sum()),
        n_pruned=int(pruned.sum()),
        pruned_circulation_l1=pruned_l1,
        pruned_circulation_fraction=pruned_l1 / (active_l1 + 1e-30),
        n_population_pruned=population_pruned,
        population_pruned_circulation_fraction=population_pruned_fraction,
        population_pruned_velocity_bound=population_pruned_velocity_bound,
        cfl=cfl,
        conservation_drift=drift,
        conservation_raw_mismatch=raw_mismatch,
        conservation_applied_correction=applied_correction,
        conservation_corrected_mismatch=corrected_mismatch,
        flux_ratio=flux_ratio,
        transfer_in_band_residual=in_band_residual,
        transfer_pre_prune_residual=pre_prune_residual,
        transfer_out_of_band_fraction=out_of_band,
        transfer_max_amplification=max_amplification,
        spectral_band_ratio=bands,
        excluded_input_circulation_l1=excluded_input_l1,
        excluded_remesh_circulation_l1=excluded_remesh_l1,
        excluded_target_circulation_l1=excluded_target_l1,
    )


# =========================================================
# Solver-facing wrapper (reads native FVM fields, writes the VPM field)
# =========================================================
class ContinuousOverlapInjector:
    """Thin FVM/VPM wrapper around :func:`continuous_handoff`."""

    def __init__(self, coupler):
        cfg = coupler.config
        self.config = cfg
        self.h = float(cfg.h)
        self.nu = float(cfg.nu)
        self.threshold_abs = float(cfg.prune_vorticity_min) * self.h**3

        self.ramp_width = float(cfg.buffer_thickness)
        self.dead_zone = float(cfg.dead_zone_h) * self.h
        self.radius_ratio = float(cfg.overlap_radius_ratio)
        self.amplification_cap = float(
            getattr(cfg, "transfer_amplification_cap", DEFAULT_TRANSFER_AMPLIFICATION_CAP)
        )
        self.boundary_prune_multiplier = float(getattr(cfg, "boundary_prune_multiplier", 1.0))
        self.u_inf = float(np.linalg.norm(cfg.u_inf))
        self.dt = float(coupler.dt_vpm)

        kernel = getattr(getattr(coupler, "vpm", None), "config", None)
        kernel = getattr(kernel, "particles_kernel", None)
        if kernel is not None and kernel != "GAUSSIAN":
            raise ValueError("The FVM-VPM handoff requires the GAUSSIAN particle kernel")

        self._box: np.ndarray | None = None
        self._cell_tree = None
        self._cell_centers: np.ndarray | None = None
        self._body_bounds: np.ndarray | None = None
        self._solid_bodies: tuple = ()
        self._lattice_anchor: np.ndarray | None = None
        self._velocity_trace: CachedVelocityTrace | None = None
        self._face_cells: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._reversed_flow_warned = False
        self._open_vortex_lines_warned = False
        self._transfer_resolution_warned = False
        self.step = 0
        self.last_transfer_diagnostics: dict[str, float] = {}
        self.last_interface_flow: dict[str, float] = {}
        self.last_vortex_line_closure: dict[str, float] = {}

    # ---- interface placement guard -----------------------------------------
    def _build_face_cell_index(self) -> None:
        """Cache the cells within one lattice cell of each box face.

        Lets the outflow assumption be checked.
        """
        self._face_cells = {}
        if self._cell_centers is None or len(self._cell_centers) == 0 or self._box is None:
            return
        centres = self._cell_centers
        box = self._box
        for axis in range(3):
            for side, (bound, sign) in enumerate(
                ((box[2 * axis], -1.0), (box[2 * axis + 1], +1.0))
            ):
                inside_others = np.ones(len(centres), dtype=bool)
                for other in range(3):
                    if other == axis:
                        continue
                    inside_others &= (centres[:, other] >= box[2 * other]) & (
                        centres[:, other] <= box[2 * other + 1]
                    )
                near = np.abs(centres[:, axis] - bound) <= self.h
                index = np.flatnonzero(near & inside_others)
                if index.size:
                    normal = np.zeros(3)
                    normal[axis] = sign
                    name = f"{'xyz'[axis]}{'min' if side == 0 else 'max'}"
                    self._face_cells[name] = (index, normal)

    def check_interface_flow(self, velocity: np.ndarray) -> dict[str, float]:
        """Return the mean outward normal velocity on each hand-off box face."""
        if not self._face_cells:
            return {}
        u = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        report: dict[str, float] = {}
        for name, (index, normal) in self._face_cells.items():
            if index.max(initial=-1) >= len(u):
                continue
            report[name] = float(np.mean(u[index] @ normal))
        return report

    @staticmethod
    def _vorticity_from_gradient(gradient: np.ndarray) -> np.ndarray:
        """omega from the coupler's gradient layout ``G[i][j] = du_j / dx_i``."""
        g = np.asarray(gradient, dtype=np.float64).reshape(-1, 3, 3)
        return np.stack(
            [
                g[:, 1, 2] - g[:, 2, 1],
                g[:, 2, 0] - g[:, 0, 2],
                g[:, 0, 1] - g[:, 1, 0],
            ],
            axis=1,
        )

    def check_vortex_line_closure(self, velocity_gradient: np.ndarray) -> dict[str, float]:
        """Mean ``|omega . n| / |omega|`` on each hand-off box face.

        Lines leaving a face become open tubes, inducing only
        ``a/sqrt(a^2+r^2)`` of the 2-D value.
        """
        if not self._face_cells:
            return {}
        omega = self._vorticity_from_gradient(velocity_gradient)
        scale = float(np.linalg.norm(omega, axis=1).mean()) + 1e-30
        report: dict[str, float] = {}
        for name, (index, normal) in self._face_cells.items():
            if index.max(initial=-1) >= len(omega):
                continue
            report[name] = float(np.mean(np.abs(omega[index] @ normal)) / scale)
        return report

    # ---- setup --------------------------------------------------------------
    def setup(self, fvm):
        transfer_box = getattr(self.config, "handoff_box", None) or self.config.fvm_box
        self._box = np.asarray(transfer_box, dtype=np.float64)
        from scipy.spatial import cKDTree

        self._cell_centers = np.asarray(fvm.get_cell_center_coordinates(), dtype=np.float64)
        # Non-master MPI ranks receive empty arrays from the gather; skip tree.
        if self._cell_centers.shape[0] > 0:
            self._cell_tree = cKDTree(self._cell_centers)
            self._velocity_trace = CachedVelocityTrace(
                self._cell_centers,
                self._cell_tree,
                neighbours=4,
            )

        wall = self.config.wall_patch_name
        if wall:
            wf = np.asarray(
                fvm.get_boundary_face_center_coordinates(wall), dtype=np.float64
            ).reshape(-1, 3)
            if wf.shape[0] > 0:
                bounds = np.array(
                    [
                        wf[:, 0].min(),
                        wf[:, 0].max(),
                        wf[:, 1].min(),
                        wf[:, 1].max(),
                        wf[:, 2].min(),
                        wf[:, 2].max(),
                    ]
                )
                on_planes = np.zeros(len(wf), dtype=bool)
                for ax in range(3):
                    on_planes |= np.isclose(wf[:, ax], bounds[2 * ax], atol=1e-9)
                    on_planes |= np.isclose(wf[:, ax], bounds[2 * ax + 1], atol=1e-9)
                if on_planes.all():
                    self._body_bounds = bounds
                    self._lattice_anchor = bounds[[0, 2, 4]] - 0.5 * self.h
                else:
                    logger.warning(
                        "[Handoff] wall patch %r is not an axis-aligned box; no "
                        "exact particle exclusion mask is available",
                        wall,
                    )

        ibm = getattr(fvm, "ibm", None)
        bodies = tuple(getattr(ibm, "bodies", ()))
        self._solid_bodies = tuple(
            body for body in bodies if bool(getattr(body, "has_solid_geometry", False))
        )
        if self._solid_bodies:
            self._body_bounds = None
            if self._cell_centers is not None and len(self._cell_centers):
                self._lattice_anchor = self._cell_centers[0].copy()
            logger.info(
                "[Handoff] exact immersed-solid exclusion enabled for: %s",
                ", ".join(str(body.name) for body in self._solid_bodies),
            )

        self._build_face_cell_index()

        l_buf = self.buffer_length
        logger.info(
            "[Handoff] ready: box x in [%.2f,%.2f]  h=%.3f  sigma=%.2fh  ramp=%.3f  "
            "dead_zone=%.3f  L_buf=%.3f  (CFL max_dt=%.3e s)  "
            "soft prune |omega|<%.3g 1/s (|Gamma|<%.3g m3/s)  amplification cap=%.2f  "
            "boundary prune=%.1fx",
            self._box[0],
            self._box[1],
            self.h,
            self.radius_ratio,
            self.ramp_width,
            self.dead_zone,
            l_buf,
            max_stable_dt(self.u_inf, l_buf, self.h),
            self.config.prune_vorticity_min,
            self.threshold_abs,
            self.amplification_cap,
            self.boundary_prune_multiplier,
        )
        logger.info("[Handoff] weighted trace (k=4), aligned remesh, bounded local transfer")

    @property
    def buffer_length(self) -> float:
        return required_buffer_length(self.u_inf, self.dt, self.h)

    def _signed_solid_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance to the nearest solid surface, positive in the fluid.

        Only feeds a C1 taper, so a first-order estimate suffices.
        """
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        distance = np.full(len(query), np.inf)
        for body in self._solid_bodies:
            sdf = getattr(body, "signed_distance", None)
            if callable(sdf):
                distance = np.minimum(
                    distance, np.asarray(sdf(query), dtype=np.float64).reshape(-1)
                )
            else:
                inside = np.asarray(
                    body.contains(query, include_boundary=False), dtype=bool
                ).reshape(-1)
                distance = np.minimum(distance, np.where(inside, -self.h, self.h))
        if self._body_bounds is not None:
            bounds = self._body_bounds
            lo_body = bounds[[0, 2, 4]]
            hi_body = bounds[[1, 3, 5]]
            outward = np.maximum(lo_body - query, query - hi_body)
            outside = np.linalg.norm(np.maximum(outward, 0.0), axis=1)
            inside_depth = np.max(outward, axis=1)
            box_distance = np.where(outside > 0.0, outside, inside_depth)
            distance = np.minimum(distance, box_distance)
        return distance

    def _points_in_solid(self, points, *, include_boundary: bool) -> np.ndarray:
        """Union of native IBM solids and the fitted-box solid (exact test)."""
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        inside = np.zeros(len(query), dtype=bool)
        for body in self._solid_bodies:
            inside |= np.asarray(
                body.contains(query, include_boundary=include_boundary), dtype=bool
            ).reshape(-1)
        if self._body_bounds is not None:
            bounds = self._body_bounds
            lo_body = bounds[[0, 2, 4]]
            hi_body = bounds[[1, 3, 5]]
            if include_boundary:
                inside |= np.all((query >= lo_body) & (query <= hi_body), axis=1)
            else:
                inside |= np.all((query > lo_body) & (query < hi_body), axis=1)
        return inside

    # ---- inject -------------------------------------------------------------
    def inject(self, vpm, velocity, velocity_gradient):
        """Execute one continuous overlap hand-off and write the VPM field."""
        self.step += 1
        assert self._box is not None and self._cell_tree is not None

        n = vpm.particles.number_of_particles
        if n > 0:
            pos = np.asarray(vpm.particles_positions, dtype=np.float64).reshape(-1, 3)
            circ = np.asarray(vpm.particles_circulation, dtype=np.float64).reshape(-1, 3)
        else:
            pos = np.zeros((0, 3))
            circ = np.zeros((0, 3))

        tree = self._cell_tree
        h = self.h
        cell_pos = self._cell_centers
        assert cell_pos is not None

        velocity_values = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        gradient_values = np.asarray(velocity_gradient, dtype=np.float64).reshape(-1, 3, 3)
        if len(velocity_values) != len(cell_pos) or len(gradient_values) != len(cell_pos):
            raise ValueError("FVM velocity, gradient, and cell-centre counts must match")

        has_solid = bool(self._solid_bodies) or self._body_bounds is not None

        self.last_interface_flow = self.check_interface_flow(velocity_values)
        self.last_vortex_line_closure = self.check_vortex_line_closure(gradient_values)
        open_faces = {
            name: value for name, value in self.last_vortex_line_closure.items() if value > 0.25
        }
        if open_faces and not self._open_vortex_lines_warned:
            self._open_vortex_lines_warned = True
            logger.warning(
                "[Handoff] vortex lines leave the hand-off box through %s "
                "(mean |omega.n|/|omega| = %s). The particle method has no images "
                "and no periodicity, so those lines become open-ended tubes whose "
                "induced velocity is too weak by a/sqrt(a^2+r^2) at distance r "
                "(0.71 at r = half-span, 0.32 at r = 3x half-span). A "
                "spanwise-uniform set-up violates this by construction and cannot "
                "be matched against a fully meshed reference.",
                ", ".join(sorted(open_faces)),
                ", ".join(f"{k}={v:.2f}" for k, v in sorted(open_faces.items())),
            )

        outflow_axis, outflow_sign = _outflow_axis_sign(self.config.U_inf)
        outflow_name = f"{'xyz'[outflow_axis]}{'max' if outflow_sign >= 0 else 'min'}"
        outflow_un = self.last_interface_flow.get(outflow_name)
        if outflow_un is not None and outflow_un <= 0.0 and not self._reversed_flow_warned:
            self._reversed_flow_warned = True
            logger.warning(
                "[Handoff] mean outward normal velocity on the outflow face %r is "
                "%.3f m/s (<= 0): the hand-off interface is inside a recirculation "
                "region. required_buffer_length, the CFL bound and the outflow-band "
                "diagnostic all assume vorticity convects out through this face; "
                "move the hand-off box downstream of the separation bubble.",
                outflow_name,
                outflow_un,
            )

        def mesh_weight_at_node(grid_pos):
            # C1 confidence the node sits inside the FVM mesh: one within a cell
            # of a cell centre, zero beyond two.
            d, _ = tree.query(grid_pos, workers=-1)
            return 1.0 - smoothstep(d, 1.0 * h, 2.0 * h)

        def fluid_weight_at_node(points):
            if not has_solid:
                return np.ones(len(np.atleast_2d(points)))
            # Preserve the exterior boundary-layer vorticity. Smooth only
            # inside the solid, where particle centres are subsequently
            # excluded exactly.
            return smoothstep(self._signed_solid_distance(points), -h, 0.0)

        def interior_at_node(points):
            return self._points_in_solid(points, include_boundary=False)

        def velocity_at(points):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            assert self._velocity_trace is not None
            sampled = self._velocity_trace.sample(points, velocity_values, gradient_values)
            if has_solid:
                # Taper to the no-slip wall value instead of clipping to it.
                sampled = sampled * smoothstep(self._signed_solid_distance(points), 0.0, h)[:, None]
            return sampled

        def circulation_at_node(grid_pos):
            return circulation_from_velocity_trace(grid_pos, h, velocity_at)

        res = continuous_handoff(
            pos,
            circ,
            self._box,
            h,
            circulation_at_node=circulation_at_node,
            u_inf=self.config.U_inf,
            mesh_weight_at_node=mesh_weight_at_node,
            fluid_weight_at_node=fluid_weight_at_node if has_solid else None,
            interior_at_node=interior_at_node if has_solid else None,
            ramp_width=self.ramp_width,
            dead_zone=self.dead_zone,
            buffer_length=self.buffer_length,
            threshold_abs=self.threshold_abs,
            radius_ratio=self.radius_ratio,
            amplification_cap=self.amplification_cap,
            boundary_prune_multiplier=self.boundary_prune_multiplier,
            u_max=self.u_inf,
            dt=self.dt,
            lattice_anchor=self._lattice_anchor,
            max_output_particles=self.config.handoff_max_particles,
        )

        vpm_dt = getattr(vpm.particles, "_np_float_dtype", np.float32)
        k = res.n_total
        vpm.replace_vortex_particles(
            position=res.pos.astype(vpm_dt),
            velocity=np.zeros((k, 3), dtype=vpm_dt),
            circulation=res.circ.astype(vpm_dt),
            radius=res.rad.astype(vpm_dt),
            volume=res.vol.astype(vpm_dt),
            viscosity=np.full(k, self.nu, dtype=vpm_dt),
            viscosity_turbulent=np.zeros(k, dtype=vpm_dt),
            zone_id=np.zeros(k, dtype=np.int32),
        )

        self.last_transfer_diagnostics = {
            "in_band_residual": res.transfer_in_band_residual,
            "out_of_band_fraction": res.transfer_out_of_band_fraction,
            "max_amplification": res.transfer_max_amplification,
        }

        logger.info(
            "[Handoff step=%d] in=%d -> out=%d  free=%d  solid_removed=%d  "
            "pruned=%d (%.3f%% Sum|Gamma|)  cap_pruned=%d (%.3f%% Sum|Gamma|, "
            "du_bound=%.3e m/s)  CFL=%.2f  |dGamma|/|Gamma|=%.2e  flux_ratio=%.3f",
            self.step,
            res.n_remesh_in,
            res.n_remesh_out,
            res.n_free,
            res.n_excluded,
            res.n_pruned,
            100.0 * res.pruned_circulation_fraction,
            res.n_population_pruned,
            100.0 * res.population_pruned_circulation_fraction,
            res.population_pruned_velocity_bound,
            res.cfl,
            res.conservation_drift.get("circulation_rel", 0.0),
            res.flux_ratio,
        )
        logger.info(
            "     [Transfer] residual final=%.2e  before-prune=%.2e  raw=%.1f%%  "
            "max amplification=%.2f (cap %.2f)",
            res.transfer_in_band_residual,
            res.transfer_pre_prune_residual,
            100.0 * res.transfer_out_of_band_fraction,
            res.transfer_max_amplification,
            self.amplification_cap,
        )
        if res.spectral_band_ratio:
            logger.info(
                "     [Spectrum] |omega_VPM|/|omega_FVM| by wavelength: %s",
                "  ".join(f"{name}={value:.3f}" for name, value in res.spectral_band_ratio.items()),
            )
        if self.last_interface_flow:
            logger.info(
                "     [Interface] mean outward u.n per face: %s",
                "  ".join(
                    f"{name}={value:+.3f}" for name, value in self.last_interface_flow.items()
                ),
            )
        if self.last_vortex_line_closure:
            logger.info(
                "     [Vortex lines] |omega.n|/|omega| per face: %s",
                "  ".join(
                    f"{name}={value:.2f}" for name, value in self.last_vortex_line_closure.items()
                ),
            )
        if res.transfer_in_band_residual > 0.10 and not self._transfer_resolution_warned:
            self._transfer_resolution_warned = True
            logger.warning(
                "[Handoff] post-correction representation residual %.2e exceeds 10%%; "
                "refine h or move the handoff away from unresolved wall vorticity.",
                res.transfer_in_band_residual,
            )
        if (
            res.excluded_input_circulation_l1 > 0.0
            or res.excluded_remesh_circulation_l1 > 0.0
            or res.excluded_target_circulation_l1 > 0.0
        ):
            logger.info(
                "     [Body taper] removed Sum|Gamma|: input=%.3e  remesh=%.3e  target=%.3e",
                res.excluded_input_circulation_l1,
                res.excluded_remesh_circulation_l1,
                res.excluded_target_circulation_l1,
            )
        if res.cfl > 0.7:
            logger.warning(
                "[Handoff] CFL=%.2f > 0.7 - buffer too short for this dt; "
                "reduce dt (max_dt~%.3e s).",
                res.cfl,
                max_stable_dt(self.u_inf, self.buffer_length, self.h),
            )

        return res
