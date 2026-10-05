"""Opt-in exact-content reuse of pure, autonomous particle induction.

StageRHS installs this only with an explicit standard-backend capability.
The caller executes its position guard before this evaluator and every
external provider afterward, including on hits. No particle state, sampler, or
callback is cached.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import struct
from typing import Any

import taichi as ti

from .treecode.lbvh import _DeviceFields


@dataclass(frozen=True)
class InductionReuseConditions:
    """Explicit backend proof obligations; no capability is inferred.

    ``operator_key`` is an exact, immutable tuple of primitive dependencies,
    NOT a hash of particle data. Include kernel/options, precision, stretching
    formulation, image geometry/tail settings and every other dependency not
    contained in ordered x/Gamma/core arrays. A backend that depends on stage
    time or hidden mutable data must decline the capability.

    ``complete_outputs_are_equivalent`` validates that requesting private
    velocity, Jacobian and enabled strength rate together leaves the requested
    numerical outputs identical to the original output-subset query. External
    external-field and exchange work must not occur inside this pure operation.

    Diagnostic callbacks cover ONLY state-local observations (e.g. image-tail
    validation and last rate defect). They must NEVER capture/restore cumulative
    evaluation/work counters. Counters continue to count actual backend work;
    this evaluator separately counts requests, exact checks, hits and misses.
    Set ``diagnostics_are_complete`` only after covering all such observations,
    including when there are none and both callbacks are omitted.
    """

    operator_key: tuple
    autonomous: bool = False
    sources_are_read_only: bool = False
    complete_outputs_are_equivalent: bool = False
    diagnostics_are_complete: bool = False
    capture_diagnostics: Callable[[], Any] | None = None
    restore_diagnostics: Callable[[Any], None] | None = None
    # Optional backend-specific public-request guards. Invoked before any
    # exact comparison, including would-be hits, with the original call args.
    # It must not mutate particle/output data; exceptions revoke validity.
    # False declines this particular request and preserves the original
    # public call unchanged; None (the default) or True accepts it.
    request_check: Callable[..., bool | None] | None = None


def _primitive_key(value):
    if type(value) in (str, bytes, int, bool, type(None)):
        return True
    if type(value) is float:
        # Reject NaNs, whose equality is not reflexive, and infinities.
        return float("-inf") < value < float("inf")
    return type(value) is tuple and all(_primitive_key(item) for item in value)


def _typed_value_key(value):
    """Preserve primitive type and all float bits, unlike Python == alone."""
    if type(value) is tuple:
        return ("tuple", tuple(_typed_value_key(item) for item in value))
    if type(value) is float:
        return ("float64", struct.pack("!d", value))
    return (type(value).__name__, value)


def _separate_output_fields(sources, outputs):
    """Private staging cannot change aliased-input/output backend semantics.

    Compare every scalar member, not just the outer vector-field identity:
    Taichi permits several field views to refer to the same placed member.
    Unknown storage interfaces conservatively retain the original call.
    """
    try:
        occupied = {
            member.ptr.snode().id for field in sources for member in field._get_field_members()
        }
        for field in outputs:
            members = [member.ptr.snode().id for member in field._get_field_members()]
            if len(set(members)) != len(members) or occupied.intersection(members):
                return False
            occupied.update(members)
    except AttributeError:
        return False
    return True


@dataclass
class InductionReuseStatistics:
    requests: int = 0
    exact_checks: int = 0
    hits: int = 0
    misses: int = 0
    bypasses: int = 0
    successful_publications: int = 0
    storage_bytes: int = 0


class _ReuseStorage:
    """Immutable field bindings: growth creates a new kernel template solver."""

    def __init__(self, capacity, source_dtype, result_dtype):
        self._fields = None
        try:
            fields = self._fields = _DeviceFields()
            self._position = fields.vector(3, dtype=source_dtype, shape=capacity)
            self._strength = fields.vector(3, dtype=source_dtype, shape=capacity)
            self._radius = fields.scalar(dtype=source_dtype, shape=capacity)
            self._velocity = fields.vector(3, dtype=result_dtype, shape=capacity)
            self._gradient = fields.matrix(3, 3, dtype=result_dtype, shape=capacity)
            self._rate = fields.vector(3, dtype=result_dtype, shape=capacity)
            self._different = fields.scalar(dtype=ti.i32, shape=())
            self._nonfinite = fields.scalar(dtype=ti.i32, shape=())
            fields.finalize()
        except BaseException as error:
            try:
                self.destroy()
            except BaseException as cleanup_error:
                error.add_note(f"Induction reuse allocation cleanup also failed: {cleanup_error!r}")
            raise

    def destroy(self):
        fields, self._fields = self._fields, None
        if fields is not None:
            fields.destroy()


@ti.data_oriented
class ExactContentInductionReuse:
    """One-solver, one-entry device cache with exact source comparisons.

    Unsupported operations bypass unchanged. Finite source contents are
    compared bit-for-bit, including order and signed zeros; no identity,
    timestep, solver-generation shortcut, or probabilistic digest grants a hit.
    Private outputs cannot be modified by later provider accumulation or by
    disposal/reuse of a backend's temporary workspace.

    ``conditions_provider`` is explicitly opt-in and called for every request.
    Returning None declines reuse for that request. Without a provider, every
    operation bypasses; arbitrary/custom induction is never assumed pure.
    The caller must not mutate source arrays concurrently with evaluation.
    """

    def __init__(
        self,
        backend,
        *,
        max_particles,
        source_dtype=ti.f32,
        result_dtype=ti.f32,
        conditions_provider: Callable[[], InductionReuseConditions | None] | None = None,
    ):
        if (
            isinstance(max_particles, bool)
            or int(max_particles) != max_particles
            or max_particles < 1
        ):
            raise ValueError("max_particles must be a positive integer")
        if source_dtype not in (ti.f32, ti.f64) or result_dtype not in (ti.f32, ti.f64):
            raise ValueError("Exact induction reuse supports f32/f64 fields only")
        self.backend = backend
        self.max_particles = int(max_particles)
        self.source_dtype = source_dtype
        self.result_dtype = result_dtype
        self._bit_dtype = ti.u32 if source_dtype == ti.f32 else ti.u64
        self.conditions_provider = conditions_provider
        self.statistics = InductionReuseStatistics()
        self._storage = None
        self._capacity = 0
        self._valid = False
        self._count = 0
        self._key = None
        self._diagnostics = None

    def invalidate(self):
        """Discard result validity without changing particle/backend state."""
        self._valid = False
        self._diagnostics = None

    def close(self):
        """Release this cache's private device allocation only."""
        self.invalidate()
        storage, self._storage = self._storage, None
        self._capacity = 0
        self.statistics.storage_bytes = 0
        if storage is not None:
            sync_error = None
            try:
                ti.sync()
            except BaseException as error:
                sync_error = error
            try:
                storage.destroy()
            except BaseException as error:
                if sync_error is None:
                    raise
                sync_error.add_note(f"Induction reuse release also failed: {error!r}")
            if sync_error is not None:
                raise sync_error

    def _ensure_capacity(self, count):
        if count <= self._capacity:
            return
        capacity = min(self.max_particles, max(count, 2 * self._capacity))
        self.close()
        self._storage = _ReuseStorage(capacity, self.source_dtype, self.result_dtype)
        self._capacity = capacity
        source_bytes = 4 if self.source_dtype == ti.f32 else 8
        result_bytes = 4 if self.result_dtype == ti.f32 else 8
        self.statistics.storage_bytes = capacity * (7 * source_bytes + 15 * result_bytes) + 8

    @ti.kernel
    def _check_sources(
        self,
        storage: ti.template(),
        position: ti.template(),
        strength: ti.template(),
        radius: ti.template(),
        count: ti.i32,
    ):
        storage._different[None] = 0
        for i in range(count):
            different = False
            for axis in ti.static(range(3)):
                different = different or (
                    ti.bit_cast(position[i][axis], self._bit_dtype)
                    != ti.bit_cast(storage._position[i][axis], self._bit_dtype)
                )
                different = different or (
                    ti.bit_cast(strength[i][axis], self._bit_dtype)
                    != ti.bit_cast(storage._strength[i][axis], self._bit_dtype)
                )
            different = different or (
                ti.bit_cast(radius[i], self._bit_dtype)
                != ti.bit_cast(storage._radius[i], self._bit_dtype)
            )
            if different:
                ti.atomic_or(storage._different[None], 1)

    @ti.kernel
    def _capture_sources(
        self,
        storage: ti.template(),
        position: ti.template(),
        strength: ti.template(),
        radius: ti.template(),
        count: ti.i32,
    ):
        storage._nonfinite[None] = 0
        for i in range(count):
            storage._position[i] = position[i]
            storage._strength[i] = strength[i]
            storage._radius[i] = radius[i]
            if (
                ti.math.isnan(position[i]).any()
                or ti.math.isinf(position[i]).any()
                or ti.math.isnan(strength[i]).any()
                or ti.math.isinf(strength[i]).any()
                or ti.math.isnan(radius[i])
                or ti.math.isinf(radius[i])
            ):
                ti.atomic_or(storage._nonfinite[None], 1)

    @ti.kernel
    def _check_outputs(self, storage: ti.template(), count: ti.i32):
        for i in range(count):
            if (
                ti.math.isnan(storage._velocity[i]).any()
                or ti.math.isinf(storage._velocity[i]).any()
                or ti.math.isnan(storage._rate[i]).any()
                or ti.math.isinf(storage._rate[i]).any()
                or ti.math.isnan(storage._gradient[i]).any()
                or ti.math.isinf(storage._gradient[i]).any()
            ):
                ti.atomic_or(storage._nonfinite[None], 1)

    @ti.kernel
    def _publish(
        self,
        storage: ti.template(),
        velocity: ti.template(),
        rate: ti.template(),
        gradient: ti.template(),
        count: ti.i32,
        with_gradient: ti.template(),
        with_rate: ti.template(),
    ):
        for i in range(count):
            velocity[i] = storage._velocity[i]
            if ti.static(with_gradient):
                gradient[i] = storage._gradient[i]
            if ti.static(with_rate):
                rate[i] = storage._rate[i]
            else:
                rate[i] = ti.Vector.zero(self.result_dtype, 3)

    def _conditions(self, sources):
        if self.conditions_provider is None:
            return None
        conditions = self.conditions_provider()
        if not isinstance(conditions, InductionReuseConditions):
            return None
        if not (
            conditions.autonomous
            and conditions.sources_are_read_only
            and conditions.complete_outputs_are_equivalent
            and conditions.diagnostics_are_complete
            and type(conditions.operator_key) is tuple
            and _primitive_key(conditions.operator_key)
            and (
                (conditions.capture_diagnostics is None) == (conditions.restore_diagnostics is None)
            )
            and (conditions.request_check is None or callable(conditions.request_check))
            and all(getattr(field, "dtype", None) == self.source_dtype for field in sources)
        ):
            return None
        return conditions

    def evaluate_stage(
        self,
        *,
        position,
        vortex_strength,
        core_radius,
        count,
        velocity_out,
        vortex_strength_rate_out,
        velocity_gradient_out=None,
        strength_rate_enabled=True,
        stage_time=0.0,
    ):
        """Reuse pure fields or evaluate the backend; no external work is skipped."""
        self.statistics.requests += 1
        sources = (position, vortex_strength, core_radius)
        try:
            conditions = self._conditions(sources)
            if conditions is not None and conditions.request_check is not None:
                checked = conditions.request_check(
                    position=position,
                    vortex_strength=vortex_strength,
                    core_radius=core_radius,
                    count=count,
                    velocity_out=velocity_out,
                    vortex_strength_rate_out=vortex_strength_rate_out,
                    velocity_gradient_out=velocity_gradient_out,
                    strength_rate_enabled=strength_rate_enabled,
                    stage_time=stage_time,
                )
                if checked is False:
                    conditions = None
        except BaseException:
            self.invalidate()
            raise
        outputs = (velocity_out, vortex_strength_rate_out) + (
            (velocity_gradient_out,) if velocity_gradient_out is not None else ()
        )
        outputs_compatible = all(
            getattr(field, "dtype", None) == self.result_dtype for field in outputs
        ) and _separate_output_fields(sources, outputs)
        if conditions is None or not outputs_compatible or count <= 0 or count > self.max_particles:
            self.statistics.bypasses += 1
            # A declined/unknown call can mutate operator state. Never retain
            # an earlier valid entry across that unproven operation.
            self.invalidate()
            return self.backend.evaluate_stage(
                position=position,
                vortex_strength=vortex_strength,
                core_radius=core_radius,
                count=count,
                velocity_out=velocity_out,
                vortex_strength_rate_out=vortex_strength_rate_out,
                velocity_gradient_out=velocity_gradient_out,
                strength_rate_enabled=strength_rate_enabled,
                stage_time=stage_time,
            )
        count = int(count)
        key = (id(self.backend), _typed_value_key(conditions.operator_key))
        hit = False
        if self._valid and self._count == count and self._key == key:
            self.statistics.exact_checks += 1
            self._check_sources(self._storage, *sources, count)
            hit = not bool(self._storage._different[None])
        try:
            if hit:
                if conditions.restore_diagnostics is not None:
                    conditions.restore_diagnostics(deepcopy(self._diagnostics))
            else:
                self.statistics.misses += 1
                self.invalidate()
                self._ensure_capacity(count)
                storage = self._storage
                self._capture_sources(storage, *sources, count)
                self.backend.evaluate_stage(
                    position=position,
                    vortex_strength=vortex_strength,
                    core_radius=core_radius,
                    count=count,
                    velocity_out=storage._velocity,
                    vortex_strength_rate_out=storage._rate,
                    velocity_gradient_out=storage._gradient,
                    strength_rate_enabled=True,
                    stage_time=stage_time,
                )
                # Complete the asynchronous operation before granting validity.
                # Tail/validation exceptions leave caller fields unpublished.
                self._check_outputs(storage, count)
                ti.sync()
                self._diagnostics = (
                    deepcopy(conditions.capture_diagnostics())
                    if conditions.capture_diagnostics is not None
                    else None
                )
                self._key = key
                self._count = count
                self._valid = not bool(storage._nonfinite[None])
            self._publish(
                self._storage,
                velocity_out,
                vortex_strength_rate_out,
                velocity_gradient_out
                if velocity_gradient_out is not None
                else self._storage._gradient,
                count,
                velocity_gradient_out is not None,
                bool(strength_rate_enabled),
            )
        except BaseException:
            self.invalidate()
            raise
        if hit:
            self.statistics.hits += 1
        self.statistics.successful_publications += 1


__all__ = ["ExactContentInductionReuse", "InductionReuseConditions", "InductionReuseStatistics"]
