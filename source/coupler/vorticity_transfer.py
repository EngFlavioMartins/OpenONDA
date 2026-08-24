"""Solenoidal velocity-defect transfer from FVM to VPM."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging

import numpy as np

from source.coupler.interpolation import FVMVelocityInterpolator
from source.coupler.reporting import format_coupler_log

logger = logging.getLogger("coupler")


VelocityEvaluator = Callable[[np.ndarray], np.ndarray]

_BLOB_SECOND_MOMENT = {
    "GAUSSIAN": 1.5,
    "WINCKELMANS": 1.5,
    "HIGH_ORDER_GAUSSIAN": 0.0,
    "SUPER_GAUSSIAN": 0.0,
}


def vortex_strength_from_velocity_trace(
    position: np.ndarray,
    particle_spacing: float,
    velocity_at: VelocityEvaluator,
) -> np.ndarray:
    """Integrate ``normal x velocity`` over each cubic particle control volume."""
    position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    vortex_strength = np.zeros_like(position)
    offset = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        offset.fill(0.0)
        offset[axis] = 0.5 * particle_spacing
        velocity_difference = np.asarray(
            velocity_at(position + offset), dtype=np.float64
        ) - np.asarray(velocity_at(position - offset), dtype=np.float64)
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
        vortex_strength += particle_spacing**2 * np.cross(normal, velocity_difference)
    return vortex_strength


def cosine_eta(
    points: np.ndarray,
    box: np.ndarray,
    authority_ramp_width: float,
    vpm_only_width: float,
) -> np.ndarray:
    """Return the C1 FVM-authority partition on points inside ``box``."""
    position = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    bounds = np.asarray(box, dtype=np.float64).reshape(6)
    distance = np.minimum.reduce(
        [
            position[:, 0] - bounds[0],
            bounds[1] - position[:, 0],
            position[:, 1] - bounds[2],
            bounds[3] - position[:, 1],
            position[:, 2] - bounds[4],
            bounds[5] - position[:, 2],
        ]
    )
    authority = np.zeros(len(position), dtype=np.float64)
    dead_zone = max(float(vpm_only_width), 0.0)
    ramp_end = float(authority_ramp_width)
    if ramp_end <= dead_zone:
        authority[distance > dead_zone] = 1.0
        return authority
    authority[distance >= ramp_end] = 1.0
    ramp = (distance > dead_zone) & (distance < ramp_end)
    phase = (distance[ramp] - dead_zone) / (ramp_end - dead_zone)
    authority[ramp] = 0.5 * (1.0 - np.cos(np.pi * phase))
    return authority


def _aligned_axis(
    lower: float,
    upper: float,
    particle_spacing: float,
    anchor: float | None,
) -> np.ndarray:
    """Cell centres whose half-cell faces remain inside one transfer bound."""
    h = float(particle_spacing)
    first_allowed = float(lower) + 0.5 * h
    last_allowed = float(upper) - 0.5 * h
    if last_allowed < first_allowed:
        raise ValueError("transfer region is narrower than one particle cell")
    if anchor is None:
        first = first_allowed
    else:
        first = float(anchor) + np.ceil((first_allowed - float(anchor)) / h) * h
    count = int(np.floor((last_allowed - first) / h + 64.0 * np.finfo(float).eps)) + 1
    if count < 1:
        raise ValueError("transfer lattice contains no cells")
    return first + h * np.arange(count, dtype=np.float64)


@dataclass(frozen=True)
class TransferLattice:
    """Regular correction lattice contained by the FVM transfer box."""

    origin: np.ndarray
    shape: tuple[int, int, int]
    position: np.ndarray
    interior_nodes: np.ndarray


def build_transfer_lattice(
    box: np.ndarray | list[float],
    particle_spacing: float,
    *,
    lattice_anchor: np.ndarray | None = None,
    interior_at_node: Callable[[np.ndarray], np.ndarray] | None = None,
) -> TransferLattice:
    """Build the fixed lattice used by the compatible velocity-curl transfer."""
    bounds = np.asarray(box, dtype=np.float64).reshape(6)
    anchor = None if lattice_anchor is None else np.asarray(lattice_anchor, dtype=np.float64)
    axes = tuple(
        _aligned_axis(
            bounds[2 * axis],
            bounds[2 * axis + 1],
            particle_spacing,
            None if anchor is None else float(anchor[axis]),
        )
        for axis in range(3)
    )
    grid = np.meshgrid(*axes, indexing="ij")
    position = np.column_stack([component.ravel() for component in grid])
    shape = (len(axes[0]), len(axes[1]), len(axes[2]))
    interior = (
        np.zeros(len(position), dtype=bool)
        if interior_at_node is None
        else np.asarray(interior_at_node(position), dtype=bool).reshape(-1)
    )
    if interior.shape != (len(position),):
        raise ValueError("interior_at_node returned the wrong number of values")
    return TransferLattice(
        origin=np.array([axis[0] for axis in axes], dtype=np.float64),
        shape=shape,
        position=position,
        interior_nodes=interior,
    )


def _extended_positions(
    lattice: TransferLattice,
    particle_spacing: float,
    *,
    guard_layers: int = 1,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return the collocated lattice with the requested guard layers."""
    h = float(particle_spacing)
    if guard_layers < 0:
        raise ValueError("guard_layers must be non-negative")
    axes: list[np.ndarray] = []
    for component, count in enumerate(lattice.shape):
        axes.append(
            lattice.origin[component] - guard_layers * h + h * np.arange(count + 2 * guard_layers)
        )
    extended_shape = (len(axes[0]), len(axes[1]), len(axes[2]))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([component.ravel() for component in mesh]), extended_shape


def discrete_divergence(
    vortex_strength: np.ndarray,
    shape: tuple[int, int, int],
    particle_spacing: float,
) -> np.ndarray:
    """Return centred ``div_h(vorticity)`` on lattice interior nodes."""
    field = np.asarray(vortex_strength, dtype=np.float64).reshape(*shape, 3)
    h = float(particle_spacing)
    if min(shape) < 3:
        return np.zeros((0, 0, 0), dtype=np.float64)
    vorticity = field / h**3
    return (
        (vorticity[2:, 1:-1, 1:-1, 0] - vorticity[:-2, 1:-1, 1:-1, 0])
        + (vorticity[1:-1, 2:, 1:-1, 1] - vorticity[1:-1, :-2, 1:-1, 1])
        + (vorticity[1:-1, 1:-1, 2:, 2] - vorticity[1:-1, 1:-1, :-2, 2])
    ) / (2.0 * h)


def normalized_divergence(
    vortex_strength: np.ndarray,
    shape: tuple[int, int, int],
    particle_spacing: float,
) -> tuple[float, float]:
    """Return dimensionless L2 and Linf divergence of a lattice field."""
    lattice_vortex_strength = np.asarray(vortex_strength, dtype=np.float64).reshape(*shape, 3)
    divergence = discrete_divergence(lattice_vortex_strength, shape, particle_spacing)
    if divergence.size == 0:
        return 0.0, 0.0
    vorticity = lattice_vortex_strength[1:-1, 1:-1, 1:-1] / float(particle_spacing) ** 3
    scale_l2 = float(np.linalg.norm(vorticity)) / np.sqrt(max(vorticity.size // 3, 1))
    scale_max = float(np.max(np.linalg.norm(vorticity, axis=-1), initial=0.0))
    h = float(particle_spacing)
    l2 = h * float(np.linalg.norm(divergence)) / np.sqrt(divergence.size)
    linf = h * float(np.max(np.abs(divergence), initial=0.0))
    return l2 / (scale_l2 + np.finfo(float).tiny), linf / (scale_max + np.finfo(float).tiny)


@dataclass
class TransferResult:
    """Correction particles and diagnostics from one FVM-to-VPM handoff."""

    position: np.ndarray
    vortex_strength: np.ndarray
    particle_volume: np.ndarray
    core_radius: np.ndarray
    updated_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    updated_vortex_strength: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    n_existing_particles: int = 0
    n_support_nodes: int = 0
    correction_vortex_strength_l1: float = 0.0
    correction_vortex_strength_net: np.ndarray = field(default_factory=lambda: np.zeros(3))
    divergence_correction_l2: float = 0.0
    divergence_correction_linf: float = 0.0
    diagnostics_evaluated: bool = True

    @property
    def n_added_particles(self) -> int:
        return len(self.position)

    @property
    def n_updated_particles(self) -> int:
        return len(self.updated_indices)

    @property
    def n_total_particles(self) -> int:
        return self.n_existing_particles + self.n_added_particles


def _transfer_log_record(step: int, result: TransferResult) -> str:
    """Return one factual record for an FVM-to-VPM transfer."""
    divergence = (
        f"{result.divergence_correction_l2:.3e}"
        if result.diagnostics_evaluated
        else "not evaluated"
    )
    return format_coupler_log(
        "Transfer",
        f"step {step:,}",
        f"particles   existing {result.n_existing_particles:,}"
        f" | updated {result.n_updated_particles:,}"
        f" | added {result.n_added_particles:,}"
        f" | total {result.n_total_particles:,}",
        f"support     {result.n_support_nodes:,} lattice nodes",
        "correction  vortex-strength magnitude sum "
        f"{result.correction_vortex_strength_l1:.3e} m^3/s"
        f" | net magnitude {float(np.linalg.norm(result.correction_vortex_strength_net)):.3e} m^3/s",
        f"divergence  relative vorticity L2 {divergence}",
    )


def solenoidal_velocity_correction(
    lattice: TransferLattice,
    particle_spacing: float,
    *,
    fvm_velocity_at: VelocityEvaluator,
    vpm_velocity_at: VelocityEvaluator,
    authority_at: Callable[[np.ndarray], np.ndarray],
    identity_authority_at: Callable[[np.ndarray], np.ndarray] | None = None,
    core_radius_ratio: float,
    blob_second_moment: float = 1.5,
    n_existing_particles: int = 0,
    compute_diagnostics: bool = True,
) -> TransferResult:
    r"""Return a blob-consistent compatible curl as correction particles.

    The leading convolution error of an isotropic blob with second moment
    ``m2`` is removed from the velocity defect before taking the curl:
    ``curl_h[(I - m2*sigma**2/6 Laplacian_h) eta (u_F-u_V)]``.  The correction
    is used only where the complete Laplacian stencil exists.
    """
    h = float(particle_spacing)
    shape = lattice.shape
    sigma = h * float(core_radius_ratio)
    m2 = float(blob_second_moment)
    if not np.isfinite(m2) or m2 < 0.0:
        raise ValueError("blob_second_moment must be finite and non-negative")
    extended_position, extended_shape = _extended_positions(lattice, h, guard_layers=2)
    authority = np.asarray(authority_at(extended_position), dtype=np.float64).reshape(-1)
    if authority.shape != (len(extended_position),):
        raise ValueError("authority_at returned the wrong number of values")
    if np.any(~np.isfinite(authority)) or np.any((authority < 0.0) | (authority > 1.0)):
        raise ValueError("authority_at must return finite values in [0, 1]")

    defect = np.zeros((len(extended_position), 3), dtype=np.float64)
    active = authority > 0.0
    if active.any():
        fvm_velocity = np.asarray(
            fvm_velocity_at(extended_position[active]), dtype=np.float64
        ).reshape(-1, 3)
        vpm_velocity = np.asarray(
            vpm_velocity_at(extended_position[active]), dtype=np.float64
        ).reshape(-1, 3)
        if fvm_velocity.shape != vpm_velocity.shape or fvm_velocity.shape != (
            int(active.sum()),
            3,
        ):
            raise ValueError("velocity evaluators returned incompatible shapes")
        if not np.all(np.isfinite(fvm_velocity)) or not np.all(np.isfinite(vpm_velocity)):
            raise RuntimeError("velocity-defect evaluation returned non-finite values")
        defect[active] = authority[active, None] * (fvm_velocity - vpm_velocity)

    twice_guarded_defect = defect.reshape(*extended_shape, 3)
    velocity_defect = twice_guarded_defect[1:-1, 1:-1, 1:-1].copy()
    if m2 > 0.0:
        laplacian = (
            twice_guarded_defect[2:, 1:-1, 1:-1]
            + twice_guarded_defect[:-2, 1:-1, 1:-1]
            + twice_guarded_defect[1:-1, 2:, 1:-1]
            + twice_guarded_defect[1:-1, :-2, 1:-1]
            + twice_guarded_defect[1:-1, 1:-1, 2:]
            + twice_guarded_defect[1:-1, 1:-1, :-2]
            - 6.0 * velocity_defect
        ) / h**2
        active_grid = active.reshape(extended_shape)
        complete_stencil = active_grid[1:-1, 1:-1, 1:-1].copy()
        complete_stencil &= active_grid[2:, 1:-1, 1:-1]
        complete_stencil &= active_grid[:-2, 1:-1, 1:-1]
        complete_stencil &= active_grid[1:-1, 2:, 1:-1]
        complete_stencil &= active_grid[1:-1, :-2, 1:-1]
        complete_stencil &= active_grid[1:-1, 1:-1, 2:]
        complete_stencil &= active_grid[1:-1, 1:-1, :-2]
        velocity_defect[complete_stencil] -= (m2 * sigma**2 / 6.0) * laplacian[complete_stencil]
    d_dx = (velocity_defect[2:, 1:-1, 1:-1] - velocity_defect[:-2, 1:-1, 1:-1]) / (2.0 * h)
    d_dy = (velocity_defect[1:-1, 2:, 1:-1] - velocity_defect[1:-1, :-2, 1:-1]) / (2.0 * h)
    d_dz = (velocity_defect[1:-1, 1:-1, 2:] - velocity_defect[1:-1, 1:-1, :-2]) / (2.0 * h)
    correction = h**3 * np.stack(
        [
            d_dy[..., 2] - d_dz[..., 1],
            d_dz[..., 0] - d_dx[..., 2],
            d_dx[..., 1] - d_dy[..., 0],
        ],
        axis=-1,
    )

    identity_authority = np.asarray(
        (authority_at if identity_authority_at is None else identity_authority_at)(
            lattice.position
        ),
        dtype=np.float64,
    ).reshape(-1)
    if identity_authority.shape != (len(lattice.position),):
        raise ValueError("identity_authority_at returned the wrong number of values")
    identity_active = identity_authority.reshape(shape) > 0.0
    outside_identity = ~identity_active & np.any(correction != 0.0, axis=-1)
    if np.any(outside_identity):
        raise RuntimeError(
            "compatible curl reached an eta=0 particle node; reserve one velocity-stencil "
            "guard layer inside the declared correction authority"
        )

    authority_grid = authority.reshape(extended_shape)[1:-1, 1:-1, 1:-1]
    support = authority_grid[1:-1, 1:-1, 1:-1] > 0.0
    support |= authority_grid[2:, 1:-1, 1:-1] > 0.0
    support |= authority_grid[:-2, 1:-1, 1:-1] > 0.0
    support |= authority_grid[1:-1, 2:, 1:-1] > 0.0
    support |= authority_grid[1:-1, :-2, 1:-1] > 0.0
    support |= authority_grid[1:-1, 1:-1, 2:] > 0.0
    support |= authority_grid[1:-1, 1:-1, :-2] > 0.0

    support &= ~lattice.interior_nodes.reshape(shape)
    support &= identity_active
    correction[~support] = 0.0
    flat = correction.reshape(-1, 3)
    nonzero = support.reshape(-1) & np.any(flat != 0.0, axis=1)
    position = lattice.position[nonzero]
    vortex_strength = flat[nonzero]
    particle_volume = np.full(len(position), h**3, dtype=np.float64)
    core_radius = np.full(len(position), h * float(core_radius_ratio), dtype=np.float64)
    divergence_l2, divergence_linf = (0.0, 0.0)
    if compute_diagnostics:
        divergence_l2, divergence_linf = normalized_divergence(flat, shape, h)
    return TransferResult(
        position=position,
        vortex_strength=vortex_strength,
        particle_volume=particle_volume,
        core_radius=core_radius,
        n_existing_particles=int(n_existing_particles),
        n_support_nodes=int(support.sum()),
        correction_vortex_strength_l1=float(np.linalg.norm(vortex_strength, axis=1).sum()),
        correction_vortex_strength_net=np.sum(vortex_strength, axis=0),
        divergence_correction_l2=divergence_l2,
        divergence_correction_linf=divergence_linf,
        diagnostics_evaluated=compute_diagnostics,
    )


def coalesce_lattice_corrections(
    result: TransferResult,
    existing_position: np.ndarray,
    existing_core_radius: np.ndarray,
    lattice: TransferLattice,
    particle_spacing: float,
    correction_radius: float,
) -> TransferResult:
    """Add coincident same-kernel corrections through existing particles.

    Only vortex strength is updated. Position, core radius, particle volume,
    kinematic viscosity, IDs, and all other persistent fields remain untouched.
    """
    if result.n_added_particles == 0 or len(existing_position) == 0:
        return result

    position = np.asarray(existing_position, dtype=np.float64).reshape(-1, 3)
    core_radius = np.asarray(existing_core_radius, dtype=np.float64).reshape(-1)
    if len(position) != len(core_radius):
        raise ValueError("existing particle position and core_radius counts must match")

    h = float(particle_spacing)
    lattice_index = np.rint((position - lattice.origin) / h).astype(np.int64)
    reconstructed = lattice.origin + h * lattice_index
    on_lattice = np.max(np.abs(position - reconstructed), axis=1) <= 1.0e-4 * h
    on_lattice &= np.all(lattice_index >= 0, axis=1)
    on_lattice &= np.all(lattice_index < np.asarray(lattice.shape), axis=1)
    on_lattice &= np.isclose(core_radius, correction_radius, rtol=1.0e-5, atol=1.0e-7 * h)

    existing_by_node: dict[tuple[int, int, int], int] = {}
    for particle_index in np.flatnonzero(on_lattice):
        existing_by_node.setdefault(tuple(lattice_index[particle_index]), int(particle_index))

    correction_index = np.rint((result.position - lattice.origin) / h).astype(np.int64)
    matched_particle = np.array(
        [existing_by_node.get(tuple(index), -1) for index in correction_index], dtype=np.int64
    )
    matched = matched_particle >= 0
    if not matched.any():
        return result

    order = np.argsort(matched_particle[matched])
    result.updated_indices = matched_particle[matched][order]
    result.updated_vortex_strength = result.vortex_strength[matched][order]
    result.position = result.position[~matched]
    result.vortex_strength = result.vortex_strength[~matched]
    result.particle_volume = result.particle_volume[~matched]
    result.core_radius = result.core_radius[~matched]
    return result


class VorticityTransfer:
    """Apply one compatible velocity-defect curl without rebuilding the VPM."""

    def __init__(self, coupler):
        cfg = coupler.setup
        if (
            coupler.kinematic_viscosity is None
            or coupler.vpm_time_step_size is None
            or coupler.fvm_box is None
        ):
            raise RuntimeError("VorticityTransfer requires initialized FVM and VPM state")
        self.config = cfg
        self.particle_spacing = float(cfg.vpm_particle_spacing)
        self.kinematic_viscosity = float(coupler.kinematic_viscosity)
        self.authority_ramp_width = float(cfg.authority_ramp_width)
        self.vpm_only_width = float(cfg.vpm_only_width)
        self.core_radius_ratio = float(cfg.vpm_core_radius_ratio)
        self.diagnostic_interval = int(cfg.transfer_diagnostic_interval_steps)
        self._fvm_box = np.asarray(coupler.fvm_box, dtype=np.float64)
        self._box: np.ndarray | None = None
        self._cell_centre: np.ndarray | None = None
        self._cell_tree = None
        self._velocity_trace: FVMVelocityInterpolator | None = None
        self._body_bounds: np.ndarray | None = None
        self._solid_bodies: tuple = ()
        self._lattice_anchor: np.ndarray | None = None
        self._lattice: TransferLattice | None = None
        self._face_cells: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.step = 0
        self.last_transfer_diagnostics: dict[str, float] = {}
        self.last_interface_flow: dict[str, float] = {}
        self.last_vortex_line_closure: dict[str, float] = {}

    def _build_face_cell_index(self) -> None:
        self._face_cells = {}
        if self._cell_centre is None or self._box is None:
            return
        centres = self._cell_centre
        for axis in range(3):
            for side, (bound, sign) in enumerate(
                ((self._box[2 * axis], -1.0), (self._box[2 * axis + 1], 1.0))
            ):
                inside = np.ones(len(centres), dtype=bool)
                for other in range(3):
                    if other != axis:
                        inside &= (centres[:, other] >= self._box[2 * other]) & (
                            centres[:, other] <= self._box[2 * other + 1]
                        )
                index = np.flatnonzero(
                    inside & (np.abs(centres[:, axis] - bound) <= self.particle_spacing)
                )
                if index.size:
                    normal = np.zeros(3)
                    normal[axis] = sign
                    name = f"{'xyz'[axis]}{'min' if side == 0 else 'max'}"
                    self._face_cells[name] = (index, normal)

    def check_interface_flow(self, velocity: np.ndarray) -> dict[str, float]:
        values = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        return {
            name: float(np.mean(values[index] @ normal))
            for name, (index, normal) in self._face_cells.items()
            if index.max(initial=-1) < len(values)
        }

    @staticmethod
    def _vorticity_from_gradient(gradient: np.ndarray) -> np.ndarray:
        """Curl for the FVM layout ``G[i,j] = d(u_j)/d(x_i)``."""
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
        vorticity = self._vorticity_from_gradient(velocity_gradient)
        scale = float(np.linalg.norm(vorticity, axis=1).mean()) + np.finfo(float).tiny
        return {
            name: float(np.mean(np.abs(vorticity[index] @ normal)) / scale)
            for name, (index, normal) in self._face_cells.items()
            if index.max(initial=-1) < len(vorticity)
        }

    def setup(self, fvm) -> None:
        transfer_box = self.config.transfer_region_bounds or self._fvm_box
        self._box = np.asarray(transfer_box, dtype=np.float64)
        from scipy.spatial import cKDTree  # type: ignore[attr-defined]

        self._cell_centre = np.asarray(fvm.get_cell_centre_coordinates(), dtype=np.float64)

        # Partitioned FVM getters are collective even though only rank zero
        # receives assembled arrays. Keep their order identical on all ranks.
        wall_patches = [
            boundary_condition.name
            for boundary_condition in fvm.setup.boundaries
            if boundary_condition.mesh_type == "wall"
        ]
        wall_faces = None
        if len(wall_patches) == 1:
            wall_faces = np.asarray(
                fvm.get_boundary_face_centre_coordinates(wall_patches[0]), dtype=np.float64
            ).reshape(-1, 3)

        if len(self._cell_centre) == 0:
            self._build_face_cell_index()
            return
        self._cell_tree = cKDTree(self._cell_centre)
        self._velocity_trace = FVMVelocityInterpolator(
            self._cell_centre,
            self._cell_tree,
            neighbour_count=4,
        )

        if wall_faces is not None and len(wall_faces):
            bounds = np.array(
                [
                    wall_faces[:, 0].min(),
                    wall_faces[:, 0].max(),
                    wall_faces[:, 1].min(),
                    wall_faces[:, 1].max(),
                    wall_faces[:, 2].min(),
                    wall_faces[:, 2].max(),
                ]
            )
            on_planes = np.zeros(len(wall_faces), dtype=bool)
            for axis in range(3):
                on_planes |= np.isclose(wall_faces[:, axis], bounds[2 * axis], atol=1e-9)
                on_planes |= np.isclose(wall_faces[:, axis], bounds[2 * axis + 1], atol=1e-9)
            if on_planes.all():
                wall_cells = (bounds[[1, 3, 5]] - bounds[[0, 2, 4]]) / self.particle_spacing
                if not np.allclose(wall_cells, np.rint(wall_cells), rtol=0.0, atol=1.0e-10):
                    raise ValueError(
                        "VPM particle spacing must divide every axis-aligned body extent; "
                        f"got body cells {wall_cells.tolist()} at h={self.particle_spacing:.12g}. "
                        "Use a wall-commensurate transfer/GBD lattice."
                    )
                self._body_bounds = bounds
                # Put lattice nodes on the exact wall planes. With a zero
                # velocity defect on/in the solid, curl support then lies
                # on the wall and fluid nodes, never in the open interior.
                self._lattice_anchor = bounds[[0, 2, 4]]

        ibm = getattr(fvm, "ibm", None)
        bodies = () if ibm is None else tuple(ibm.bodies)
        self._solid_bodies = tuple(body for body in bodies if body.has_solid_geometry)
        if self._solid_bodies:
            self._body_bounds = None
            self._lattice_anchor = self._cell_centre[0].copy()

        self._lattice = build_transfer_lattice(
            self._box,
            self.particle_spacing,
            lattice_anchor=self._lattice_anchor,
            interior_at_node=lambda points: self._points_in_solid(points, include_boundary=False),
        )
        self._build_face_cell_index()
        logger.info(
            format_coupler_log(
                "TransferGrid",
                f"{len(self._lattice.position):,} nodes | spacing {self.particle_spacing:.4g} m",
                f"authority  ramp {self.authority_ramp_width:.4g} m"
                f" | VPM-only width {self.vpm_only_width:.4g} m",
            )
        )

    def _points_in_solid(self, points, *, include_boundary: bool) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        inside = np.zeros(len(query), dtype=bool)
        for body in self._solid_bodies:
            inside |= np.asarray(
                body.contains(query, include_boundary=include_boundary), dtype=bool
            ).reshape(-1)
        if self._body_bounds is not None:
            lower = self._body_bounds[[0, 2, 4]]
            upper = self._body_bounds[[1, 3, 5]]
            if include_boundary:
                inside |= np.all((query >= lower) & (query <= upper), axis=1)
            else:
                inside |= np.all((query > lower) & (query < upper), axis=1)
        return inside

    @staticmethod
    def _chunked_evaluate(
        points: np.ndarray,
        evaluator: VelocityEvaluator,
        *,
        chunk_size: int = 100_000,
    ) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        result = np.empty_like(query)
        for start in range(0, len(query), chunk_size):
            stop = min(start + chunk_size, len(query))
            result[start:stop] = np.asarray(evaluator(query[start:stop]), dtype=np.float64).reshape(
                -1, 3
            )
        return result

    def _sample_vpm_velocity(self, vpm, points: np.ndarray) -> np.ndarray:
        return self._chunked_evaluate(
            points,
            lambda query: vpm.compute_velocity_at_points(
                query,
                include_freestream=True,
                zone_mask=None,
                include_body=True,
            ),
        )

    def _fluid_authority(self, points: np.ndarray) -> np.ndarray:
        """Velocity-blend authority with solid and identity guards."""
        assert self._box is not None
        dead_zone = self.vpm_only_width
        ramp_end = self.authority_ramp_width
        if dead_zone > 0.0:
            dead_zone += self.particle_spacing
            ramp_end += self.particle_spacing
        authority = cosine_eta(
            points,
            self._box,
            ramp_end,
            dead_zone,
        )
        authority[self._points_in_solid(points, include_boundary=True)] = 0.0
        return authority

    def _identity_authority(self, points: np.ndarray) -> np.ndarray:
        """Declared correction authority used for pointwise identity checks."""
        assert self._box is not None
        return cosine_eta(
            points,
            self._box,
            self.authority_ramp_width,
            self.vpm_only_width,
        )

    def transfer(self, vpm, velocity, velocity_gradient) -> TransferResult:
        """Add the compatible velocity-defect curl to the existing VPM cloud."""
        self.step += 1
        if self._box is None or self._lattice is None or self._velocity_trace is None:
            raise RuntimeError("VorticityTransfer.setup() has not prepared a transfer lattice")
        cell_position = self._cell_centre
        assert cell_position is not None
        velocity_values = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        gradient_values = np.asarray(velocity_gradient, dtype=np.float64).reshape(-1, 3, 3)
        if len(velocity_values) != len(cell_position) or len(gradient_values) != len(cell_position):
            raise ValueError("FVM velocity, gradient, and cell-centre counts must match")

        self.last_interface_flow = self.check_interface_flow(velocity_values)
        self.last_vortex_line_closure = self.check_vortex_line_closure(gradient_values)
        evaluate_diagnostics = self.step % self.diagnostic_interval == 0
        result = solenoidal_velocity_correction(
            self._lattice,
            self.particle_spacing,
            fvm_velocity_at=lambda points: self._velocity_trace.sample(
                points,
                velocity_values,
                gradient_values,
            ),
            vpm_velocity_at=lambda points: self._sample_vpm_velocity(vpm, points),
            authority_at=self._fluid_authority,
            identity_authority_at=self._identity_authority,
            core_radius_ratio=self.core_radius_ratio,
            blob_second_moment=_BLOB_SECOND_MOMENT[getattr(vpm, "particle_kernel", "GAUSSIAN")],
            n_existing_particles=int(vpm.particles.n_particles_total),
            compute_diagnostics=evaluate_diagnostics,
        )

        can_coalesce = all(
            hasattr(vpm.particles, name) for name in ("position_cpu", "core_radius_cpu")
        )
        if result.n_added_particles and can_coalesce:
            result = coalesce_lattice_corrections(
                result,
                vpm.particles.position_cpu(),
                vpm.particles.core_radius_cpu(),
                self._lattice,
                self.particle_spacing,
                self.particle_spacing * self.core_radius_ratio,
            )

        required = result.n_total_particles
        capacity = int(vpm.particles.capacity)
        if required > capacity:
            raise RuntimeError(
                "FVM-to-VPM correction requires "
                f"{result.n_added_particles} new particles ({result.n_existing_particles} existing, "
                f"required capacity {required}, VPM maximum {capacity}). Increase "
                "VPMSetup.max_n_particles; the coupler will not delete wake particles."
            )

        if result.n_updated_particles:
            mask = np.zeros(result.n_existing_particles, dtype=bool)
            mask[result.updated_indices] = True
            vpm.update_particle_vortex_strength(mask, result.updated_vortex_strength)

        if result.n_added_particles:
            dtype = vpm.np_dtype
            vpm.add_vortex_particles(
                position=result.position.astype(dtype),
                velocity=np.zeros((result.n_added_particles, 3), dtype=dtype),
                vortex_strength=result.vortex_strength.astype(dtype),
                core_radius=result.core_radius.astype(dtype),
                particle_volume=result.particle_volume.astype(dtype),
                kinematic_viscosity=np.full(
                    result.n_added_particles, self.kinematic_viscosity, dtype=dtype
                ),
                eddy_viscosity=np.zeros(result.n_added_particles, dtype=dtype),
                group_id=np.zeros(result.n_added_particles, dtype=np.int32),
                zone_id=np.zeros(result.n_added_particles, dtype=np.int32),
            )

        if result.n_added_particles:
            notify_mutation = getattr(vpm, "notify_external_particle_mutation", None)
            if notify_mutation is not None:
                notify_mutation()

        self.last_transfer_diagnostics = {
            "correction_divergence_l2": result.divergence_correction_l2,
            "correction_divergence_linf": result.divergence_correction_linf,
        }
        logger.info(_transfer_log_record(self.step, result))
        return result


__all__ = [
    "TransferLattice",
    "TransferResult",
    "VorticityTransfer",
    "build_transfer_lattice",
    "coalesce_lattice_corrections",
    "cosine_eta",
    "discrete_divergence",
    "normalized_divergence",
    "solenoidal_velocity_correction",
    "vortex_strength_from_velocity_trace",
]
