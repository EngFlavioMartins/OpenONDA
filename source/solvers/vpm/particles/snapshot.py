"""Reusable device-only snapshots for provisional particle transactions.

These are disposable scratch buffers, not restart files or source-field caches.
Every capture copies the current device fields. A monotonically increasing
generation rejects a stale handle if its named scratch slot has been reused.
"""

from dataclasses import dataclass

import numpy as np
import taichi as ti


@ti.kernel
def _copy_fields(source: ti.template(), target: ti.template(), count: int, validate: ti.template()):
    if ti.static(validate):
        target._invalid[None] = 0
    for i in range(count):
        target.position[i] = source.position[i]
        target.velocity[i] = source.velocity[i]
        target.vortex_strength[i] = source.vortex_strength[i]
        target.core_radius[i] = source.core_radius[i]
        target.particle_volume[i] = source.particle_volume[i]
        target.kinematic_viscosity[i] = source.kinematic_viscosity[i]
        target.eddy_viscosity[i] = source.eddy_viscosity[i]
        target.group_id[i] = source.group_id[i]
        target.zone_id[i] = source.zone_id[i]
        target.velocity_gradient[i] = source.velocity_gradient[i]
        target.strain_rate[i] = source.strain_rate[i]
        if ti.static(validate):
            for field in ti.static(
                (
                    source.position,
                    source.velocity,
                    source.vortex_strength,
                    source.velocity_gradient,
                    source.strain_rate,
                )
            ):
                if ti.math.isnan(field[i]).any() or ti.math.isinf(field[i]).any():
                    ti.atomic_or(target._invalid[None], 1)
            for field in ti.static(
                (
                    source.core_radius,
                    source.particle_volume,
                    source.kinematic_viscosity,
                    source.eddy_viscosity,
                )
            ):
                if ti.math.isnan(field[i]) or ti.math.isinf(field[i]):
                    ti.atomic_or(target._invalid[None], 1)


@ti.kernel
def _rebuild_replacement_fields(particles: ti.template(), count: int):
    # Rebuild these two derived fields when publishing the restored cloud.
    for i in range(count):
        particles.vorticity[i] = particles.vortex_strength[i] / particles.particle_volume[i]
        particles.effective_viscosity[i] = (
            particles.kinematic_viscosity[i] + particles.eddy_viscosity[i]
        )


class ParticleSnapshotBuffer:
    """Own a bounded reusable device allocation for the eleven rollback fields."""

    def __init__(self, capacity: int, dtype):
        self.capacity = max(1, int(capacity))
        self.generation = 0
        self._tree = None
        builder = ti.FieldsBuilder()
        self._invalid = ti.field(dtype=ti.i32)
        builder.place(self._invalid)
        fields = []
        for name in ("position", "velocity", "vortex_strength"):
            field = ti.Vector.field(3, dtype=dtype)
            setattr(self, name, field)
            fields.append(field)
        for name in ("core_radius", "particle_volume", "kinematic_viscosity", "eddy_viscosity"):
            field = ti.field(dtype=dtype)
            setattr(self, name, field)
            fields.append(field)
        for name in ("group_id", "zone_id"):
            field = ti.field(dtype=ti.i32)
            setattr(self, name, field)
            fields.append(field)
        for name in ("velocity_gradient", "strain_rate"):
            field = ti.Matrix.field(3, 3, dtype=dtype)
            setattr(self, name, field)
            fields.append(field)
        builder.dense(ti.i, self.capacity).place(*fields)
        self._tree = builder.finalize()

    def capture(self, particles, *, prepare_lineage: bool = False):
        count = int(particles.n_particles_total)
        if self._tree is None or count > self.capacity:
            raise RuntimeError("Particle snapshot storage cannot hold the current cloud")
        _copy_fields(particles, self, count, True)
        self.generation += 1
        snapshot = ParticleSnapshot(self, self.generation, count, particles)
        if prepare_lineage:
            snapshot.prepare_lineage()
        return snapshot

    def destroy(self) -> None:
        if self._tree is not None:
            ti.sync()
            self._tree.destroy()
            self._tree = None


@dataclass
class ParticleSnapshot:
    """Single-generation handle; only its originating particle owner may restore it."""

    buffer: ParticleSnapshotBuffer
    generation: int
    count: int
    owner: object
    lineage: tuple[np.ndarray, np.ndarray] | None = None
    finite_validated: bool = False

    def _validate(self, particles) -> None:
        if particles is not self.owner:
            raise ValueError("Particle snapshot belongs to another solver")
        if self.buffer._tree is None or self.generation != self.buffer.generation:
            raise RuntimeError("Particle snapshot scratch slot has been reused or released")
        if self.count > particles.capacity:
            raise RuntimeError("Particle snapshot exceeds the destination capacity")
        if not self.finite_validated:
            if int(self.buffer._invalid[None]):
                raise ValueError("Particle snapshot contains non-finite replacement fields")
            self.finite_validated = True

    def prepare_lineage(self) -> tuple[np.ndarray, np.ndarray]:
        """Preserve refinement bookkeeping only when lineage is active.

        Normal coupling needs no particle downloads. Optional refinement lineage
        still lives on the host, so its two inputs use the existing float64 norm
        and cube-root path rather than changing reduction precision.
        """
        self._validate(self.owner)
        if self.lineage is None:
            strength = self.buffer.vortex_strength.to_numpy()[: self.count].astype(np.float64)
            volume = self.buffer.particle_volume.to_numpy()[: self.count].copy()
            self.lineage = (np.linalg.norm(strength, axis=1), volume)
        return self.lineage

    def restore(self, particles) -> None:
        self._validate(particles)
        previous = int(particles.n_particles_total)
        _copy_fields(self.buffer, particles, self.count, False)
        _rebuild_replacement_fields(particles, self.count)
        particles.n_particles_total = self.count
        particles.sync_device_counter()
        particles.touch_state()
        particles._log_particles_replaced(previous)
