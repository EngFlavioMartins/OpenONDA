"""Reuse of the self-induced field of an unchanged accepted particle state.

An accepted step evaluates particle induction at its final state for the
particle state checks, and the next step's first RK stage evaluates the same
state again unless a split operator, regeneration or an external particle
update changes the sources in between. Every such change publishes a new
``Particles.state_revision``. This store keeps the self-induced velocity,
velocity gradient and vortex-strength rate of one accepted revision and
publishes them for later requests at that same revision, so the reused
fields are bit-identical to the evaluation that produced them.

Only the autonomous self-induced field is stored. Freestream, VLM bound
vortices, surface sources, body fields and every other stage-time provider
are evaluated again for each request.
"""

from dataclasses import dataclass

import taichi as ti

from .base import InductionMethod, TaichiDataType, TaichiField
from .treecode.lbvh import _DeviceFields


@dataclass
class InductionReuseStatistics:
    """Count backend evaluations and reused accepted-state fields."""

    evaluations: int = 0
    reuses: int = 0


@ti.kernel
def _copy_field(source: ti.template(), target: ti.template(), count: ti.i32):  # type: ignore[invalid-annotation]
    for i in range(count):
        target[i] = source[i]


@ti.kernel
def _zero_vector(target: ti.template(), count: ti.i32):  # type: ignore[invalid-annotation]
    for i in range(count):
        target[i] = ti.Vector([0.0, 0.0, 0.0])


class _InductionStorage:
    """Velocity, gradient and strength-rate fields in one disposable SNode tree."""

    def __init__(self, capacity: int, dtype: TaichiDataType):
        self.capacity = capacity
        self.dtype = dtype
        self._fields = _DeviceFields()
        try:
            self.velocity = self._fields.vector(3, dtype=dtype, shape=capacity)
            self.gradient = self._fields.matrix(3, 3, dtype=dtype, shape=capacity)
            self.strength_rate = self._fields.vector(3, dtype=dtype, shape=capacity)
            self._fields.finalize()
        except BaseException:
            self._fields.destroy()
            raise

    def destroy(self) -> None:
        self._fields.destroy()


class AcceptedStateInduction:
    """Self-induced field of one accepted particle revision.

    The entry is valid for the induction backend that evaluated it, the
    accepted particle revision and the active particle count. It records which
    outputs were evaluated: velocity always, the velocity gradient when it was
    requested and the strength rate when it was requested or retained for the
    next first RK stage. A request reuses the entry only when every output it
    needs is stored; otherwise the backend evaluates the request again and the
    entry is replaced.

    The backend must be autonomous and must not change velocity or strength
    rate when a gradient is requested in addition, which every induction
    backend guarantees. A request without strength rate can receive the
    velocity of an evaluation with strength rate; direct summation computes
    that velocity in its fused velocity and stretching pass, which agrees with
    its velocity-only pass to rounding.
    """

    def __init__(self) -> None:
        self._storage: _InductionStorage | None = None
        self._backend: InductionMethod | None = None
        self._revision: int | None = None
        self._count = 0
        self._has_gradient = False
        self._has_strength_rate = False

    def invalidate(self) -> None:
        """Discard the stored field without releasing its device storage."""
        self._backend = None
        self._revision = None

    def close(self) -> None:
        """Release the device storage."""
        self.invalidate()
        storage, self._storage = self._storage, None
        if storage is not None:
            ti.sync()
            storage.destroy()

    def matches(
        self,
        backend: InductionMethod,
        revision: int,
        count: int,
        gradient: bool,
        strength_rate: bool,
    ) -> bool:
        """Whether the stored field answers this request exactly."""
        return (
            self._backend is backend
            and self._revision == revision
            and self._count == count
            and (self._has_gradient or not gradient)
            and (self._has_strength_rate or not strength_rate)
        )

    def _ensure_storage(self, capacity: int, dtype: TaichiDataType) -> _InductionStorage:
        # Storage spans the full output capacity: reallocating as the particle
        # count grows would compile new copy kernels for every new field.
        storage = self._storage
        if storage is None or storage.dtype != dtype or storage.capacity != capacity:
            self.close()
            storage = self._storage = _InductionStorage(capacity, dtype)
        return storage

    def evaluate(
        self,
        backend: InductionMethod,
        revision: int,
        *,
        position: TaichiField,
        vortex_strength: TaichiField,
        core_radius: TaichiField,
        count: int,
        velocity_out: TaichiField,
        gradient: bool,
        strength_rate: bool,
        stage_time: float,
    ) -> None:
        """Evaluate the requested outputs of one accepted state and store them."""
        self.invalidate()
        storage = self._ensure_storage(velocity_out.shape[0], velocity_out.dtype)
        backend.evaluate_stage(
            position=position,
            vortex_strength=vortex_strength,
            core_radius=core_radius,
            count=count,
            velocity_out=storage.velocity,
            vortex_strength_rate_out=storage.strength_rate,
            velocity_gradient_out=storage.gradient if gradient else None,
            strength_rate_enabled=strength_rate,
            stage_time=stage_time,
        )
        self._backend = backend
        self._revision = revision
        self._count = count
        self._has_gradient = gradient
        self._has_strength_rate = strength_rate

    def publish(
        self,
        *,
        velocity_out: TaichiField,
        vortex_strength_rate_out: TaichiField,
        velocity_gradient_out: TaichiField | None,
        strength_rate_enabled: bool,
    ) -> None:
        """Copy the stored outputs into the caller's fields."""
        storage, count = self._storage, self._count
        if storage is None:
            raise RuntimeError("no stored self-induced field to publish")
        _copy_field(storage.velocity, velocity_out, count)
        if velocity_gradient_out is not None:
            _copy_field(storage.gradient, velocity_gradient_out, count)
        if strength_rate_enabled:
            _copy_field(storage.strength_rate, vortex_strength_rate_out, count)
        else:
            _zero_vector(vortex_strength_rate_out, count)


__all__ = ["AcceptedStateInduction", "InductionReuseStatistics"]
