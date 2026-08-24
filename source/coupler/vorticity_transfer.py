"""Absolute FVM-state replacement inside the FVM--VPM overlap."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np

from source.coupler.reporting import format_coupler_log

logger = logging.getLogger("coupler")


def replacement_eta(
    points: np.ndarray,
    box: np.ndarray | list[float] | tuple[float, ...],
    blend_width: float,
) -> np.ndarray:
    """Return the FVM state-replacement weight at ``points``.

    ``blend_width == 0`` selects a hard ownership boundary: every point inside
    ``box`` has ``eta = 1`` and every point outside has ``eta = 0``. A positive
    width replaces the hard jump by a C1 cosine ramp measured inward from the
    six box faces. This is a state partition, not an additive correction.
    """
    position = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    bounds = np.asarray(box, dtype=np.float64).reshape(6)
    width = float(blend_width)
    if not np.all(np.isfinite(bounds)) or np.any(bounds[1::2] <= bounds[::2]):
        raise ValueError("replacement box must contain six finite increasing bounds")
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("eta_blend_width must be finite and non-negative")

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
    eta = np.zeros(len(position), dtype=np.float64)
    inside = distance >= 0.0
    if width == 0.0:
        eta[inside] = 1.0
        return eta

    eta[distance >= width] = 1.0
    ramp = inside & (distance < width)
    eta[ramp] = 0.5 * (1.0 - np.cos(np.pi * distance[ramp] / width))
    return eta


@dataclass(frozen=True)
class TransferResult:
    """Particle and circulation budget for one absolute state replacement."""

    n_particles_before: int
    n_particles_retained: int
    n_particles_removed: int
    n_particles_blended: int
    n_particles_injected: int
    n_particles_after: int
    injected_vortex_strength_l1: float
    injected_vortex_strength_net: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    replaced_vortex_strength_l1: float = 0.0
    replaced_vortex_strength_net: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    state_change_vortex_strength_net: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    eta_blending_enabled: bool = False


def _validate_particle_sources(
    position: np.ndarray,
    cell_volume: np.ndarray,
    vorticity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    volume = np.asarray(cell_volume, dtype=np.float64).reshape(-1)
    source_vorticity = np.asarray(vorticity, dtype=np.float64).reshape(-1, 3)
    if len(source_position) != len(volume) or len(source_position) != len(source_vorticity):
        raise ValueError("FVM position, volume, and vorticity counts must match")
    if not np.all(np.isfinite(source_position)):
        raise RuntimeError("FVM cell positions contain non-finite values")
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0.0):
        raise RuntimeError("FVM cell volumes must be finite and positive")
    if not np.all(np.isfinite(source_vorticity)):
        raise RuntimeError("FVM cell vorticity contains non-finite values")
    return source_position, volume, source_vorticity


def replace_particles_from_fvm(
    vpm,
    *,
    transfer_box: np.ndarray | list[float] | tuple[float, ...],
    eta_blend_width: float,
    fvm_position: np.ndarray,
    fvm_cell_volume: np.ndarray,
    fvm_vorticity: np.ndarray,
    core_radius_ratio: float,
    kinematic_viscosity: float,
    fvm_solid_mask: np.ndarray | None = None,
) -> TransferResult:
    r"""Replace the overlap state with literal FVM cell circulation.

    Existing particles are attenuated by ``1 - eta`` and FVM cell particles
    carry ``eta * V_cell * omega_F``. Consequently ``eta = 1`` is a hard
    delete/reinject operation, ``eta = 0`` leaves the VPM state untouched, and
    intermediate values form a partition-of-unity state blend. Particles
    outside ``transfer_box`` are never mutated.
    """
    ratio = float(core_radius_ratio)
    viscosity = float(kinematic_viscosity)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("core_radius_ratio must be finite and positive")
    if not np.isfinite(viscosity) or viscosity < 0.0:
        raise ValueError("kinematic_viscosity must be finite and non-negative")

    source_position, source_volume, source_vorticity = _validate_particle_sources(
        fvm_position,
        fvm_cell_volume,
        fvm_vorticity,
    )
    source_eta = replacement_eta(source_position, transfer_box, eta_blend_width)
    if fvm_solid_mask is not None:
        solid = np.asarray(fvm_solid_mask, dtype=bool).reshape(-1)
        if len(solid) != len(source_position):
            raise ValueError("fvm_solid_mask must match the FVM cell count")
        source_eta[solid] = 0.0

    source_strength = source_volume[:, None] * source_vorticity
    inject = (source_eta > 0.0) & np.any(source_strength != 0.0, axis=1)
    injected_position = source_position[inject]
    injected_volume = source_volume[inject]
    injected_strength = source_eta[inject, None] * source_strength[inject]
    injected_core_radius = ratio * np.cbrt(injected_volume)

    particles = vpm.particles
    n_before = int(particles.n_particles_total)
    existing_position = np.asarray(particles.position_cpu(), dtype=np.float64).reshape(-1, 3)
    existing_strength = np.asarray(particles.vortex_strength_cpu(), dtype=np.float64).reshape(-1, 3)
    if len(existing_position) != n_before or len(existing_strength) != n_before:
        raise RuntimeError("VPM particle arrays do not match the active particle count")
    if not np.all(np.isfinite(existing_position)) or not np.all(np.isfinite(existing_strength)):
        raise RuntimeError("VPM particle state contains non-finite values")

    existing_eta = replacement_eta(existing_position, transfer_box, eta_blend_width)
    tolerance = 32.0 * np.finfo(np.float64).eps
    remove = existing_eta >= 1.0 - tolerance
    blend = (existing_eta > tolerance) & ~remove
    remove_index = np.flatnonzero(remove)
    n_removed = int(len(remove_index))
    n_blended = int(np.count_nonzero(blend))
    n_injected = int(len(injected_position))
    n_after = n_before - n_removed + n_injected
    capacity = int(particles.capacity)
    if n_after > capacity:
        raise RuntimeError(
            "FVM overlap replacement requires "
            f"{n_after:,} particles ({n_before:,} before, {n_removed:,} removed, "
            f"{n_injected:,} injected), exceeding the VPM capacity {capacity:,}."
        )

    replaced_strength = existing_eta[:, None] * existing_strength
    replaced_net = replaced_strength.sum(axis=0)
    injected_net = injected_strength.sum(axis=0)

    # Complete every validation and capacity check before the first mutation.
    if n_blended:
        vpm.update_particle_vortex_strength(
            blend,
            -existing_eta[blend, None] * existing_strength[blend],
        )
    if n_removed:
        vpm.remove_particles(particle_indices=remove_index.tolist())
    if n_injected:
        dtype = vpm.np_dtype
        vpm.add_vortex_particles(
            position=np.ascontiguousarray(injected_position, dtype=dtype),
            velocity=np.zeros((n_injected, 3), dtype=dtype),
            vortex_strength=np.ascontiguousarray(injected_strength, dtype=dtype),
            core_radius=np.ascontiguousarray(injected_core_radius, dtype=dtype),
            particle_volume=np.ascontiguousarray(injected_volume, dtype=dtype),
            kinematic_viscosity=np.full(n_injected, viscosity, dtype=dtype),
            eddy_viscosity=np.zeros(n_injected, dtype=dtype),
            group_id=np.zeros(n_injected, dtype=np.int32),
            zone_id=np.zeros(n_injected, dtype=np.int32),
        )

    actual_after = int(particles.n_particles_total)
    if actual_after != n_after:
        raise RuntimeError(
            f"VPM particle count after replacement is {actual_after}, expected {n_after}"
        )
    return TransferResult(
        n_particles_before=n_before,
        n_particles_retained=n_before - n_removed,
        n_particles_removed=n_removed,
        n_particles_blended=n_blended,
        n_particles_injected=n_injected,
        n_particles_after=n_after,
        injected_vortex_strength_l1=float(np.linalg.norm(injected_strength, axis=1).sum()),
        injected_vortex_strength_net=injected_net,
        replaced_vortex_strength_l1=float(np.linalg.norm(replaced_strength, axis=1).sum()),
        replaced_vortex_strength_net=replaced_net,
        state_change_vortex_strength_net=injected_net - replaced_net,
        eta_blending_enabled=float(eta_blend_width) > 0.0,
    )


def _transfer_log_record(step: int, result: TransferResult) -> str:
    return format_coupler_log(
        "StateReplacement",
        f"step {step:,} | eta blend {'on' if result.eta_blending_enabled else 'off'}",
        "particles  "
        f"before {result.n_particles_before:,} | removed {result.n_particles_removed:,}"
        f" | blended {result.n_particles_blended:,} | injected {result.n_particles_injected:,}"
        f" | after {result.n_particles_after:,}",
        "vortex strength  "
        f"replaced L1 {result.replaced_vortex_strength_l1:.3e}"
        f" | injected L1 {result.injected_vortex_strength_l1:.3e} m^3/s"
        f" | net state change {float(np.linalg.norm(result.state_change_vortex_strength_net)):.3e} m^3/s",
    )


class VorticityTransfer:
    """Synchronize the inner VPM cloud with the absolute FVM vorticity state."""

    def __init__(self, coupler):
        cfg = coupler.setup
        if coupler.kinematic_viscosity is None or coupler.fvm_box is None:
            raise RuntimeError("VorticityTransfer requires initialized FVM and VPM state")
        self.config = cfg
        if not np.isfinite(coupler.vpm_core_radius_ratio):
            raise RuntimeError("VorticityTransfer requires the resolved VPM core-radius ratio")
        self.core_radius_ratio = float(coupler.vpm_core_radius_ratio)
        self.eta_blend_width = float(cfg.eta_blend_width)
        self.kinematic_viscosity = float(coupler.kinematic_viscosity)
        self.diagnostic_interval = int(cfg.transfer_diagnostic_interval_steps)
        self._fvm_box = np.asarray(coupler.fvm_box, dtype=np.float64)
        self._box: np.ndarray | None = None
        self._cell_centre: np.ndarray | None = None
        self._cell_volume: np.ndarray | None = None
        self._fvm_solid_mask: np.ndarray | None = None
        self._body_bounds: np.ndarray | None = None
        self._solid_bodies: tuple = ()
        self._lattice_anchor: np.ndarray | None = None
        self._face_cells: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.step = 0
        self.last_interface_flow: dict[str, float] = {}
        self.last_vortex_line_closure: dict[str, float] = {}

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
            comparison = (query >= lower) & (query <= upper)
            if not include_boundary:
                comparison = (query > lower) & (query < upper)
            inside |= np.all(comparison, axis=1)
        return inside

    def _build_face_cell_index(self) -> None:
        self._face_cells = {}
        if self._cell_centre is None or self._box is None:
            return
        centres = self._cell_centre
        scale = (
            np.cbrt(self._cell_volume) if self._cell_volume is not None else np.zeros(len(centres))
        )
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
                index = np.flatnonzero(inside & (np.abs(centres[:, axis] - bound) <= scale))
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

    def check_vortex_line_closure(self, velocity_gradient: np.ndarray) -> dict[str, float]:
        vorticity = self._vorticity_from_gradient(velocity_gradient)
        scale = float(np.linalg.norm(vorticity, axis=1).mean()) + np.finfo(float).tiny
        return {
            name: float(np.mean(np.abs(vorticity[index] @ normal)) / scale)
            for name, (index, normal) in self._face_cells.items()
            if index.max(initial=-1) < len(vorticity)
        }

    def setup(self, fvm) -> None:
        self._box = np.asarray(
            self.config.transfer_region_bounds or self._fvm_box, dtype=np.float64
        )
        self._cell_centre = np.asarray(fvm.get_cell_centre_coordinates(), dtype=np.float64).reshape(
            -1, 3
        )
        self._cell_volume = np.asarray(fvm.get_cell_volume(), dtype=np.float64).reshape(-1)

        # These partitioned getters are collective, even though only rank zero
        # receives the assembled arrays. Keep their call order identical.
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
        if len(self._cell_volume) != len(self._cell_centre):
            raise RuntimeError("FVM cell-centre and cell-volume counts do not match")
        _validate_particle_sources(
            self._cell_centre,
            self._cell_volume,
            np.zeros_like(self._cell_centre),
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
                on_planes |= np.isclose(wall_faces[:, axis], bounds[2 * axis], atol=1.0e-9)
                on_planes |= np.isclose(wall_faces[:, axis], bounds[2 * axis + 1], atol=1.0e-9)
            if on_planes.all():
                self._body_bounds = bounds
                self._lattice_anchor = bounds[[0, 2, 4]]

        ibm = getattr(fvm, "ibm", None)
        bodies = () if ibm is None else tuple(ibm.bodies)
        self._solid_bodies = tuple(body for body in bodies if body.has_solid_geometry)
        if self._solid_bodies:
            self._body_bounds = None
            self._lattice_anchor = self._cell_centre[0].copy()

        self._fvm_solid_mask = self._points_in_solid(
            self._cell_centre,
            include_boundary=True,
        )
        donor_eta = replacement_eta(self._cell_centre, self._box, self.eta_blend_width)
        donor_count = int(np.count_nonzero((donor_eta > 0.0) & ~self._fvm_solid_mask))
        if donor_count == 0:
            raise ValueError("FVM transfer region contains no fluid cell centres")
        self._build_face_cell_index()
        logger.info(
            format_coupler_log(
                "ReplacementRegion",
                f"{donor_count:,} FVM fluid cells",
                f"eta blend {'off' if self.eta_blend_width == 0.0 else f'{self.eta_blend_width:.4g} m'}",
                "state  Gamma = cell volume * FVM vorticity",
            )
        )

    def transfer(self, vpm, velocity, velocity_gradient) -> TransferResult:
        """Replace the inner particle state and preserve the outer particle cloud."""
        self.step += 1
        if self._box is None or self._cell_centre is None or self._cell_volume is None:
            raise RuntimeError("VorticityTransfer.setup() has not prepared the FVM donor cells")
        velocity_values = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        gradient_values = np.asarray(velocity_gradient, dtype=np.float64).reshape(-1, 3, 3)
        if len(velocity_values) != len(self._cell_centre) or len(gradient_values) != len(
            self._cell_centre
        ):
            raise ValueError("FVM velocity, gradient, and cell-centre counts must match")

        self.last_interface_flow = self.check_interface_flow(velocity_values)
        self.last_vortex_line_closure = self.check_vortex_line_closure(gradient_values)
        result = replace_particles_from_fvm(
            vpm,
            transfer_box=self._box,
            eta_blend_width=self.eta_blend_width,
            fvm_position=self._cell_centre,
            fvm_cell_volume=self._cell_volume,
            fvm_vorticity=self._vorticity_from_gradient(gradient_values),
            core_radius_ratio=self.core_radius_ratio,
            kinematic_viscosity=self.kinematic_viscosity,
            fvm_solid_mask=self._fvm_solid_mask,
        )
        if self.step % self.diagnostic_interval == 0:
            logger.info(_transfer_log_record(self.step, result))
        return result


__all__ = [
    "TransferResult",
    "VorticityTransfer",
    "replace_particles_from_fvm",
    "replacement_eta",
]
