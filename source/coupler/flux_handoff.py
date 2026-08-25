"""Experimental conservative vorticity-flux release for FVM→VPM coupling.

This module deliberately has no dependency on the production volumetric M4'
replacement path and does not mutate a VPM solver.  It owns only flux-budget
accounting and deterministic particle birth geometry so the two contracts can
be tested independently before an experimental solver integration is proposed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

_GEOMETRY_TOLERANCE = 128.0 * np.finfo(np.float64).eps


def vorticity_transport_flux(
    velocity: np.ndarray,
    vorticity: np.ndarray,
    normal: np.ndarray,
    normal_vorticity_gradient: np.ndarray | None = None,
    *,
    kinematic_viscosity: float = 0.0,
) -> np.ndarray:
    r"""Return outward conservative vorticity flux through surface patches.

    For incompressible constant-viscosity flow this is the flux in

    ``d_t omega_i + d_j(u_j omega_i - u_i omega_j - nu d_j omega_i) = 0``.

    Consequently, for an outward unit normal ``n``, the flux is
    ``(u·n) omega - (omega·n) u - nu d_n omega``.  The sign of the final term
    follows directly from the conservative equation above.
    """
    u = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
    omega = np.asarray(vorticity, dtype=np.float64).reshape(-1, 3)
    unit_normal = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    if len(u) != len(omega) or len(u) != len(unit_normal):
        raise ValueError("velocity, vorticity, and normal counts must match")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(omega)):
        raise ValueError("velocity and vorticity must be finite")
    if not np.all(np.isfinite(unit_normal)):
        raise ValueError("surface normals must be finite")
    normal_length = np.linalg.norm(unit_normal, axis=1)
    if not np.allclose(normal_length, 1.0, rtol=0.0, atol=64.0 * np.finfo(np.float64).eps):
        raise ValueError("surface normals must be unit vectors")
    viscosity = float(kinematic_viscosity)
    if not np.isfinite(viscosity) or viscosity < 0.0:
        raise ValueError("kinematic_viscosity must be finite and non-negative")
    if normal_vorticity_gradient is None:
        gradient = np.zeros_like(omega)
    else:
        gradient = np.asarray(normal_vorticity_gradient, dtype=np.float64).reshape(-1, 3)
        if len(gradient) != len(omega) or not np.all(np.isfinite(gradient)):
            raise ValueError("normal_vorticity_gradient must be finite and match vorticity")
    return (
        np.einsum("ij,ij->i", u, unit_normal)[:, None] * omega
        - np.einsum("ij,ij->i", omega, unit_normal)[:, None] * u
        - viscosity * gradient
    )


@dataclass
class _ReleaseSlot:
    """Persistent state for one global deterministic release-lattice slot."""

    position: np.ndarray
    normal: np.ndarray
    pending_strength: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    accumulated_displacement: float = 0.0
    pending_age: float = 0.0
    next_normal_index: int = 1
    emitted_count: int = 0


@dataclass(frozen=True)
class ReleaseSlotStatus:
    """Inspectable circulation-reservoir state for one release slot."""

    slot_id: tuple[int, int, int]
    pending_strength: np.ndarray
    accumulated_displacement: float
    pending_age: float
    next_normal_index: int
    emitted_count: int


@dataclass(frozen=True)
class FluxReleaseBatch:
    """Emission and conservation report for one flux-handoff interval."""

    position: np.ndarray
    vortex_strength: np.ndarray
    core_radius: np.ndarray
    slot_id: np.ndarray
    emitted_vortex_strength_net: np.ndarray
    outward_flux_vortex_strength_increment: np.ndarray
    inward_flux_vortex_strength_increment: np.ndarray
    pending_vortex_strength_net: np.ndarray
    conservation_error: np.ndarray
    outward_area_fraction: float
    inward_area_fraction: float
    min_new_new_separation: float
    min_new_existing_separation: float
    neighbour_count_within_2sigma: np.ndarray
    neighbour_count_within_3sigma: np.ndarray
    held_slot_ids: tuple[tuple[int, int, int], ...]
    trapped_slot_ids: tuple[tuple[int, int, int], ...]


class FluxReleaseHandoff:
    """One-way flux release with a displacement-governed circulation reservoir.

    The caller supplies globally unique integer release-slot identifiers.  A
    slot may receive many surface-patch contributions, but it can emit at most
    one particle at each deterministic normal-lattice location.  A proposed
    particle closer than ``particle_spacing`` to a new or existing particle is
    held in the slot reservoir; it is never discarded or merged.

    This is an isolated experiment.  It does not yet create solver particles,
    estimate FVM gradients, choose a release surface, or establish a physical
    FVM/VPM field-continuity guarantee.
    """

    def __init__(
        self,
        *,
        particle_spacing: float,
        core_radius: float,
        max_pending_strength: float | None = None,
        max_pending_age: float | None = None,
    ):
        spacing = float(particle_spacing)
        radius = float(core_radius)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("particle_spacing must be finite and positive")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("core_radius must be finite and positive")
        for name, value in (
            ("max_pending_strength", max_pending_strength),
            ("max_pending_age", max_pending_age),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when configured")
        self.particle_spacing = spacing
        self.core_radius = radius
        self.max_pending_strength = max_pending_strength
        self.max_pending_age = max_pending_age
        self._slot: dict[tuple[int, int, int], _ReleaseSlot] = {}
        self._outward_flux_total = np.zeros(3, dtype=np.float64)
        self._emitted_total = np.zeros(3, dtype=np.float64)

    @property
    def outward_flux_total(self) -> np.ndarray:
        """Return total outward flux received since construction."""
        return self._outward_flux_total.copy()

    @property
    def emitted_vortex_strength_total(self) -> np.ndarray:
        """Return total strength emitted since construction."""
        return self._emitted_total.copy()

    def slot_status(self) -> tuple[ReleaseSlotStatus, ...]:
        """Return deterministic, copy-safe reservoir state sorted by slot ID."""
        return tuple(
            ReleaseSlotStatus(
                slot_id=slot_id,
                pending_strength=slot.pending_strength.copy(),
                accumulated_displacement=slot.accumulated_displacement,
                pending_age=slot.pending_age,
                next_normal_index=slot.next_normal_index,
                emitted_count=slot.emitted_count,
            )
            for slot_id, slot in sorted(self._slot.items())
        )

    @staticmethod
    def _slot_ids(slot_id: np.ndarray | Iterable[Iterable[int]]) -> np.ndarray:
        index = np.asarray(slot_id, dtype=np.int64).reshape(-1, 3)
        return np.ascontiguousarray(index)

    def _get_or_create_slot(
        self,
        slot_id: tuple[int, int, int],
        position: np.ndarray,
        normal: np.ndarray,
    ) -> _ReleaseSlot:
        existing = self._slot.get(slot_id)
        if existing is None:
            created = _ReleaseSlot(position=position.copy(), normal=normal.copy())
            self._slot[slot_id] = created
            return created
        tolerance = _GEOMETRY_TOLERANCE * max(1.0, self.particle_spacing)
        if (
            np.linalg.norm(existing.position - position) > tolerance
            or np.linalg.norm(existing.normal - normal) > tolerance
        ):
            raise ValueError("a release slot ID must have one deterministic position and normal")
        return existing

    def _minimum_separation(self, candidate: np.ndarray, position: np.ndarray) -> float:
        if not len(position):
            return float("inf")
        return float(np.linalg.norm(position - candidate, axis=1).min())

    def advance(
        self,
        *,
        slot_id: np.ndarray | Iterable[Iterable[int]],
        slot_position: np.ndarray,
        slot_normal: np.ndarray,
        patch_area: np.ndarray,
        vorticity_flux: np.ndarray,
        normal_velocity: np.ndarray,
        time_step_size: float,
        existing_position: np.ndarray | None = None,
    ) -> FluxReleaseBatch:
        """Accumulate outward flux and emit only after one spacing of transport.

        ``vorticity_flux`` has units of vorticity transport per area.  Its
        patch-integrated circulation increment is ``flux * patch_area * dt``.
        Patches with non-positive normal velocity are recorded as inward flux
        and are deliberately not fed to this one-way emitter.
        """
        identifiers = self._slot_ids(slot_id)
        position = np.asarray(slot_position, dtype=np.float64).reshape(-1, 3)
        normal = np.asarray(slot_normal, dtype=np.float64).reshape(-1, 3)
        area = np.asarray(patch_area, dtype=np.float64).reshape(-1)
        flux = np.asarray(vorticity_flux, dtype=np.float64).reshape(-1, 3)
        velocity = np.asarray(normal_velocity, dtype=np.float64).reshape(-1)
        count = len(identifiers)
        if (
            len(position) != count
            or len(normal) != count
            or len(area) != count
            or len(flux) != count
            or len(velocity) != count
        ):
            raise ValueError("all release-patch arrays must have one value per slot_id")
        dt = float(time_step_size)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("time_step_size must be finite and positive")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(flux)):
            raise ValueError("release positions and vorticity flux must be finite")
        if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(velocity)):
            raise ValueError("release normals and normal velocities must be finite")
        if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
            raise ValueError("release patch areas must be finite and positive")
        normal_length = np.linalg.norm(normal, axis=1)
        if not np.allclose(normal_length, 1.0, rtol=0.0, atol=64.0 * np.finfo(np.float64).eps):
            raise ValueError("release normals must be unit vectors")
        existing = (
            np.empty((0, 3), dtype=np.float64)
            if existing_position is None
            else np.asarray(existing_position, dtype=np.float64).reshape(-1, 3)
        )
        if not np.all(np.isfinite(existing)):
            raise ValueError("existing particle positions must be finite")

        grouped: dict[tuple[int, int, int], dict[str, np.ndarray | float]] = {}
        outward_flux_increment = np.zeros(3, dtype=np.float64)
        inward_flux_increment = np.zeros(3, dtype=np.float64)
        outward_area = 0.0
        inward_area = 0.0
        total_area = float(area.sum())
        for patch in range(count):
            identifier: tuple[int, int, int] = (
                int(identifiers[patch, 0]),
                int(identifiers[patch, 1]),
                int(identifiers[patch, 2]),
            )
            contribution = flux[patch] * area[patch] * dt
            entry = grouped.get(identifier)
            if entry is None:
                entry = {
                    "position": position[patch].copy(),
                    "normal": normal[patch].copy(),
                    "outward_strength": np.zeros(3, dtype=np.float64),
                    "outward_area_weighted_speed": 0.0,
                    "area": 0.0,
                }
                grouped[identifier] = entry
            else:
                tolerance = _GEOMETRY_TOLERANCE * max(1.0, self.particle_spacing)
                if (
                    np.linalg.norm(np.asarray(entry["position"]) - position[patch]) > tolerance
                    or np.linalg.norm(np.asarray(entry["normal"]) - normal[patch]) > tolerance
                ):
                    raise ValueError("duplicate release slot IDs need matching position and normal")
            if velocity[patch] > 0.0:
                entry["outward_strength"] = np.asarray(entry["outward_strength"]) + contribution
                entry["outward_area_weighted_speed"] = (
                    float(entry["outward_area_weighted_speed"]) + area[patch] * velocity[patch]
                )
                entry["area"] = float(entry["area"]) + area[patch]
                outward_flux_increment += contribution
                outward_area += area[patch]
            else:
                inward_flux_increment += contribution
                inward_area += area[patch]

        self._outward_flux_total += outward_flux_increment
        emitted_position: list[np.ndarray] = []
        emitted_strength: list[np.ndarray] = []
        emitted_slot_id: list[tuple[int, int, int]] = []
        new_new_separation: list[float] = []
        new_existing_separation: list[float] = []
        held_slot_ids: list[tuple[int, int, int]] = []
        for identifier in sorted(grouped):
            entry = grouped[identifier]
            slot = self._get_or_create_slot(
                identifier,
                np.asarray(entry["position"], dtype=np.float64),
                np.asarray(entry["normal"], dtype=np.float64),
            )
            strength_increment = np.asarray(entry["outward_strength"], dtype=np.float64)
            weighted_area = float(entry["area"])
            speed = (
                float(entry["outward_area_weighted_speed"]) / weighted_area
                if weighted_area > 0.0
                else 0.0
            )
            displacement_increment = speed * dt
            original_pending = slot.pending_strength.copy()
            original_displacement = slot.accumulated_displacement
            original_age = slot.pending_age
            if not np.any(original_pending) and not np.any(strength_increment):
                # Retain release-lattice phase even in a zero-vorticity slot.
                slot.accumulated_displacement = (
                    original_displacement + displacement_increment
                ) % self.particle_spacing
                continue
            slot.pending_age += dt

            def emit(
                candidate: np.ndarray,
                release_strength: np.ndarray,
                *,
                release_slot_id: tuple[int, int, int] = identifier,
            ) -> bool:
                min_existing = self._minimum_separation(candidate, existing)
                min_new = self._minimum_separation(
                    candidate,
                    np.asarray(emitted_position, dtype=np.float64).reshape(-1, 3),
                )
                if (
                    min(min_existing, min_new) + _GEOMETRY_TOLERANCE * self.particle_spacing
                    < self.particle_spacing
                ):
                    return False
                emitted_position.append(candidate)
                emitted_strength.append(release_strength)
                emitted_slot_id.append(release_slot_id)
                new_new_separation.append(min_new)
                new_existing_separation.append(min_existing)
                return True

            # The first threshold mixes a carried reservoir with the current
            # interval. Interpolate the current flux at the exact crossing.
            threshold = self.particle_spacing * (1.0 - _GEOMETRY_TOLERANCE)
            if original_displacement + displacement_increment < threshold:
                slot.pending_strength = original_pending + strength_increment
                slot.accumulated_displacement = original_displacement + displacement_increment
            else:
                if original_displacement >= self.particle_spacing:
                    # A previously held reservoir has no recoverable substep
                    # history. Release the oldest pending state at the surface.
                    release_strength = original_pending.copy()
                    remainder_strength = strength_increment.copy()
                    remainder_displacement = displacement_increment
                    candidate = slot.position.copy()
                else:
                    needed_displacement = self.particle_spacing - original_displacement
                    fraction = (
                        needed_displacement / displacement_increment
                        if displacement_increment > 0.0
                        else 0.0
                    )
                    fraction = float(np.clip(fraction, 0.0, 1.0))
                    release_strength = original_pending + fraction * strength_increment
                    remainder_strength = (1.0 - fraction) * strength_increment
                    remainder_displacement = max(0.0, displacement_increment - needed_displacement)
                    # Existing particles are supplied at the end of this FVM
                    # interval, so advance a just-emitted particle through its
                    # post-emission displacement before checking its geometry.
                    candidate = slot.position + slot.normal * remainder_displacement
                if not np.any(release_strength):
                    slot.pending_strength = remainder_strength
                    slot.accumulated_displacement = remainder_displacement
                elif not emit(candidate, release_strength):
                    slot.pending_strength = original_pending + strength_increment
                    slot.accumulated_displacement = original_displacement + displacement_increment
                    slot.pending_age = original_age + dt
                    held_slot_ids.append(identifier)
                    continue
                else:
                    self._emitted_total += release_strength
                    slot.next_normal_index += 1
                    slot.emitted_count += 1
                    slot.pending_strength = remainder_strength
                    slot.accumulated_displacement = remainder_displacement

                # A large FVM interval can cross multiple spacings. Its
                # remaining flux is uniform over remaining transport, so split
                # it at each exact crossing instead of making one large blob.
                while slot.accumulated_displacement >= threshold:
                    fraction = self.particle_spacing / slot.accumulated_displacement
                    release_strength = fraction * slot.pending_strength
                    remainder_strength = slot.pending_strength - release_strength
                    remainder_displacement = slot.accumulated_displacement - self.particle_spacing
                    candidate = slot.position + slot.normal * remainder_displacement
                    if not np.any(release_strength):
                        slot.pending_strength = remainder_strength
                        slot.accumulated_displacement = remainder_displacement
                        break
                    if not emit(candidate, release_strength):
                        held_slot_ids.append(identifier)
                        break
                    self._emitted_total += release_strength
                    slot.next_normal_index += 1
                    slot.emitted_count += 1
                    slot.pending_strength = remainder_strength
                    slot.accumulated_displacement = remainder_displacement
            if not np.any(slot.pending_strength):
                slot.pending_strength[:] = 0.0
                slot.pending_age = 0.0

        trapped_slot_ids: list[tuple[int, int, int]] = []
        for identifier, slot in self._slot.items():
            if (
                self.max_pending_strength is not None
                and np.linalg.norm(slot.pending_strength) > self.max_pending_strength
            ) or (self.max_pending_age is not None and slot.pending_age > self.max_pending_age):
                trapped_slot_ids.append(identifier)

        emitted_array = (
            np.asarray(emitted_position, dtype=np.float64).reshape(-1, 3)
            if emitted_position
            else np.empty((0, 3), dtype=np.float64)
        )
        strength_array = (
            np.asarray(emitted_strength, dtype=np.float64).reshape(-1, 3)
            if emitted_strength
            else np.empty((0, 3), dtype=np.float64)
        )
        slot_id_array = (
            np.asarray(emitted_slot_id, dtype=np.int64).reshape(-1, 3)
            if emitted_slot_id
            else np.empty((0, 3), dtype=np.int64)
        )
        pending_net = sum(
            (slot.pending_strength for slot in self._slot.values()),
            start=np.zeros(3, dtype=np.float64),
        )
        conservation_error = self._emitted_total + pending_net - self._outward_flux_total
        geometry_position = np.vstack((existing, emitted_array))
        neighbour_count_2sigma = np.array(
            [
                np.count_nonzero(
                    np.linalg.norm(geometry_position - point, axis=1) <= 2.0 * self.core_radius
                )
                - 1
                for point in emitted_array
            ],
            dtype=np.int64,
        )
        neighbour_count_3sigma = np.array(
            [
                np.count_nonzero(
                    np.linalg.norm(geometry_position - point, axis=1) <= 3.0 * self.core_radius
                )
                - 1
                for point in emitted_array
            ],
            dtype=np.int64,
        )
        return FluxReleaseBatch(
            position=emitted_array,
            vortex_strength=strength_array,
            core_radius=np.full(len(emitted_array), self.core_radius),
            slot_id=slot_id_array,
            emitted_vortex_strength_net=strength_array.sum(axis=0, dtype=np.float64),
            outward_flux_vortex_strength_increment=outward_flux_increment,
            inward_flux_vortex_strength_increment=inward_flux_increment,
            pending_vortex_strength_net=pending_net,
            conservation_error=conservation_error,
            outward_area_fraction=outward_area / total_area if total_area else 0.0,
            inward_area_fraction=inward_area / total_area if total_area else 0.0,
            min_new_new_separation=min(new_new_separation, default=float("inf")),
            min_new_existing_separation=min(new_existing_separation, default=float("inf")),
            neighbour_count_within_2sigma=neighbour_count_2sigma,
            neighbour_count_within_3sigma=neighbour_count_3sigma,
            held_slot_ids=tuple(sorted(set(held_slot_ids))),
            trapped_slot_ids=tuple(sorted(trapped_slot_ids)),
        )


__all__ = [
    "FluxReleaseBatch",
    "FluxReleaseHandoff",
    "ReleaseSlotStatus",
    "vorticity_transport_flux",
]
