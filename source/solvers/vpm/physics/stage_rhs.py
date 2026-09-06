"""Central stage right-hand side for coupled VPM evolution."""

from typing import Protocol

import numpy as np
import taichi as ti

from .induction.base import InductionMethod, StageRates, StageState


class ExternalStageContribution(Protocol):
    """Add explicitly modelled external rates at one RK stage.

    A provider may add velocity only, velocity plus a gradient, or a direct
    vortex-strength rate. It must use the supplied :class:`StageState`; the
    accepted particle fields are not part of this protocol.
    """

    def add_stage_rates(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        """Accumulate external velocity/rate contributions into stage outputs."""


class CallableStageContribution:
    """Adapt an explicit external stage callback to the provider protocol.

    The callback is called as::

        evaluate(stage_time, stage_position, stage_vortex_strength,
                 velocity_out, strength_rate_out, gradient_out)

    where all arrays are limited to the active stage particle count and the
    three output arrays are writable. The callback may leave unused outputs at
    zero. Its contributions are accumulated into the supplied ``StageRates``.
    Set ``include_external_stretching=True`` when the callback's returned
    gradient represents an external velocity field whose transposed gradient
    contraction must also contribute to the vortex-strength rate.
    """

    def __init__(
        self, evaluate, *, include_external_stretching: bool = False, physics=None
    ) -> None:
        self.evaluate = evaluate
        self.include_external_stretching = bool(include_external_stretching)
        self.physics = physics

    def add_stage_rates(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        count = int(stage_state.count)
        position = _stage_array(stage_state.position, count, self.physics)
        vortex_strength = _stage_array(stage_state.vortex_strength, count, self.physics)
        velocity = np.zeros((count, 3), dtype=np.float64)
        strength_rate = np.zeros((count, 3), dtype=np.float64)
        # Keep the callback contract stable: every output is writable even when
        # the caller did not request a diagnostic gradient destination.
        gradient = np.zeros((count, 3, 3), dtype=np.float64)
        self.evaluate(
            float(stage_time),
            position,
            vortex_strength,
            velocity,
            strength_rate,
            gradient,
        )
        if self.include_external_stretching:
            strength_rate += np.einsum("nji,nj->ni", gradient, vortex_strength)
        _accumulate_stage_array(stage_rates.velocity, velocity, count, self.physics)
        _accumulate_stage_array(
            stage_rates.vortex_strength_rate, strength_rate, count, self.physics
        )
        if stage_rates.velocity_gradient is not None:
            _accumulate_stage_array(stage_rates.velocity_gradient, gradient, count, self.physics)


class VLMStageContribution:
    """Add the latest solved VLM bound-vortex field at RK stage positions.

    VLM circulation and geometry are solved once by the accepted-step coupling
    phase. The particle target positions are still the exact temporary RK
    positions, so the external velocity is stage-state consistent while the
    boundary solve itself remains an explicitly lagged accepted-step field.
    VLM does not provide an external strength rate through this provider.
    """

    def __init__(self, vlm_solver, physics=None) -> None:
        self.vlm_solver = vlm_solver
        self.physics = physics
        self._gradient = None
        if physics is not None:
            self._gradient = ti.Matrix.field(
                3, 3, dtype=physics.accumulator_dtype, shape=(physics.max_n_particles,)
            )

    def add_stage_rates(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        if self._gradient is None or not hasattr(
            self.vlm_solver, "add_stage_velocity_and_gradient"
        ):
            self.vlm_solver.add_stage_velocity(
                stage_state.position,
                stage_rates.velocity,
                stage_state.count,
                stage_time,
            )
            return
        self.vlm_solver.add_stage_velocity_and_gradient(
            stage_state.position,
            stage_rates.velocity,
            self._gradient,
            stage_state.count,
            stage_time,
        )
        gradient = self.physics._download_matrix_field(self._gradient, stage_state.count)
        strength = _stage_array(stage_state.vortex_strength, stage_state.count, self.physics)
        rate = np.einsum("nji,nj->ni", gradient, strength)
        _accumulate_stage_array(
            stage_rates.vortex_strength_rate, rate, stage_state.count, self.physics
        )
        if stage_rates.velocity_gradient is not None:
            _accumulate_stage_array(
                stage_rates.velocity_gradient,
                gradient,
                stage_state.count,
                self.physics,
            )


class StageRHS:
    """Combine selected self-induced induction and external stage providers.

    The object has no accepted-state access.  Every provider receives the exact
    temporary position/strength fields and physical time of the current RK
    stage.  Providers that only model velocity leave the strength-rate field
    untouched; providers that model external stretching or forcing explicitly
    add the corresponding rate.
    """

    def __init__(
        self,
        induction: InductionMethod,
        providers: tuple[ExternalStageContribution, ...] = (),
        *,
        strength_enabled: bool = True,
    ) -> None:
        self.induction = induction
        self.providers = tuple(providers)
        self.strength_enabled = bool(strength_enabled)

    @property
    def requires_velocity_gradient(self) -> bool:
        """Whether a provider needs the self-induced Jacobian at every stage.

        Most direct induction calls only need velocity and stretching, and can
        use their fused velocity/rate path.  A velocity override without an
        explicit total-Jacobian callback is different: its chain rule needs the
        incoming self-induced Jacobian, including for an in-place blender.
        Keep that requirement explicit so RK does not pay for a separate
        gradient traversal on ordinary stages.
        """
        for provider in self.providers:
            if bool(getattr(provider, "requires_velocity_gradient", False)):
                return True
            physics = getattr(provider, "physics", None)
            override = getattr(physics, "velocity_override", None)
            if (
                override is not None
                and getattr(physics, "velocity_override_gradient", None) is None
            ):
                return True
        return False

    def evaluate(self, stage_state: StageState, stage_time: float, stage_rates: StageRates) -> None:
        """Evaluate self-induced and external rates for one common stage state."""
        self.induction.evaluate_stage(
            position=stage_state.position,
            vortex_strength=stage_state.vortex_strength,
            core_radius=stage_state.core_radius,
            count=stage_state.count,
            velocity_out=stage_rates.velocity,
            vortex_strength_rate_out=stage_rates.vortex_strength_rate,
            velocity_gradient_out=stage_rates.velocity_gradient,
            strength_rate_enabled=stage_rates.strength_rate_enabled,
            stage_time=stage_time,
        )
        for provider in self.providers:
            provider.add_stage_rates(stage_state, stage_time, stage_rates)
        if not self.strength_enabled or not stage_rates.strength_rate_enabled:
            _zero_stage_field(stage_rates.vortex_strength_rate, stage_state.count)


@ti.kernel
def _zero_stage_field(field: ti.template(), count: ti.i32):
    for i in range(count):
        field[i] = ti.Vector([0.0, 0.0, 0.0])


def _stage_array(field, count: int, physics=None) -> np.ndarray:
    """Return a bounded NumPy view/copy for host-side external providers."""
    if (
        physics is not None
        and hasattr(field, "to_numpy")
        and hasattr(physics, "_download_vector_field")
    ):
        return physics._download_vector_field(field, count).astype(np.float64, copy=False)
    values = field.to_numpy() if hasattr(field, "to_numpy") else np.asarray(field)
    return np.asarray(values[:count]).copy()


def _stage_matrix_array(field, count: int, physics=None) -> np.ndarray:
    """Return a bounded NumPy copy of a matrix-valued stage field."""
    if (
        physics is not None
        and hasattr(field, "to_numpy")
        and hasattr(physics, "_download_matrix_field")
    ):
        return physics._download_matrix_field(field, count).astype(np.float64, copy=False)
    values = field.to_numpy() if hasattr(field, "to_numpy") else np.asarray(field)
    return np.asarray(values[:count]).copy()


def _accumulate_stage_array(field, values: np.ndarray, count: int, physics=None) -> None:
    """Accumulate a host provider result into NumPy or Taichi output storage."""
    if (
        physics is not None
        and hasattr(field, "to_numpy")
        and hasattr(physics, "_download_vector_field")
    ):
        if values.ndim == 2 and values.shape[1] == 3:
            stored = physics._download_vector_field(field, count)
            stored += values.astype(stored.dtype, copy=False)
            physics._upload_vector_array(stored, field, count)
            return
        if values.ndim == 3 and hasattr(physics, "_download_matrix_field"):
            stored = physics._download_matrix_field(field, count)
            stored += values.astype(stored.dtype, copy=False)
            physics._upload_matrix_array(stored, field, count)
            return
    if hasattr(field, "to_numpy") and hasattr(field, "from_numpy"):
        stored = field.to_numpy()
        stored[:count] += values.astype(stored.dtype, copy=False)
        field.from_numpy(stored)
    else:
        field[:count] += values


def _replace_stage_array(field, values: np.ndarray, count: int, physics=None) -> None:
    """Replace a bounded vector or matrix stage field with host-side values."""
    if (
        physics is not None
        and hasattr(field, "to_numpy")
        and values.ndim == 2
        and hasattr(physics, "_upload_vector_array")
    ):
        physics._upload_vector_array(values, field, count)
        return
    if (
        physics is not None
        and hasattr(field, "to_numpy")
        and values.ndim == 3
        and hasattr(physics, "_upload_matrix_array")
    ):
        physics._upload_matrix_array(values, field, count)
        return
    if hasattr(field, "to_numpy") and hasattr(field, "from_numpy"):
        stored = field.to_numpy()
        stored[:count] = values.astype(stored.dtype, copy=False)
        field.from_numpy(stored)
    else:
        field[:count] = values


@ti.data_oriented
class ParticleExternalStageContribution:
    """Apply stage-time external velocities to a particle RHS.

    The provider is intentionally small: self-induced induction belongs to the
    selected induction object, while freestream, body and overlap corrections
    are explicit stage contributions.  Python callbacks are given the supplied
    stage positions, never the accepted particle field.
    """

    def __init__(self, particles, physics, source_owner=None) -> None:
        self.particles = particles
        self.physics = physics
        self.source_owner = source_owner
        self._source_gradient = ti.Matrix.field(
            3, 3, dtype=physics.accumulator_dtype, shape=(physics.max_n_particles,)
        )

    @ti.kernel
    def _add_background(self, velocity: ti.template(), count: ti.i32):
        for i in range(count):
            velocity[i] += self.particles.velocity_background[None]

    def add_stage_rates(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        count = stage_state.count
        if count == 0:
            return
        self._add_background(stage_rates.velocity, count)

        # Surface-source/blockage particles are an external velocity term, not
        # a diagnostic-only target correction.  Evaluate them against the exact
        # temporary RK positions so every coupled stage sees the same source
        # contribution as arbitrary target queries.
        owner = self.source_owner
        source_count = int(getattr(owner, "n_sources", 0)) if owner is not None else 0
        source_gradient = None
        if source_count:
            self.physics.kernels["compute_target_source_velocity_kernel"](
                stage_state.position,
                owner.source_position,
                owner.source_strength,
                owner.source_core_radius,
                stage_rates.velocity,
                count,
                source_count,
            )
            if stage_rates.strength_rate_enabled or stage_rates.velocity_gradient is not None:
                self.physics.kernels["compute_target_source_velocity_gradient_kernel"](
                    stage_state.position,
                    owner.source_position,
                    owner.source_strength,
                    owner.source_core_radius,
                    self._source_gradient,
                    count,
                    source_count,
                )
                source_gradient = self.physics._download_matrix_field(self._source_gradient, count)

        body_field = getattr(self.physics, "body_velocity_field", None)
        if body_field is not None:
            _call_stage_velocity_field(body_field, stage_state, stage_rates.velocity, count)

        body = getattr(self.physics, "body_velocity", None)
        override = getattr(self.physics, "velocity_override", None)
        body_gradient = getattr(self.physics, "body_velocity_gradient", None)
        body_gradient_field = getattr(self.physics, "body_velocity_gradient_field", None)
        if (
            body is None
            and override is None
            and body_gradient is None
            and body_gradient_field is None
            and source_gradient is None
        ):
            return
        if hasattr(self.physics, "_download_vector_field"):
            position = self.physics._download_vector_field(stage_state.position, count)
            velocity = self.physics._download_vector_field(stage_rates.velocity, count)
        else:
            position = stage_state.position.to_numpy()[:count]
            velocity = stage_rates.velocity.to_numpy()[:count]
        external_gradient = np.zeros((count, 3, 3), dtype=np.float64)
        if source_gradient is not None:
            external_gradient += source_gradient
        if body is not None:
            velocity += np.asarray(
                _call_stage_velocity_callback(body, position, stage_time), dtype=velocity.dtype
            ).reshape(count, 3)
            if body_gradient is None:
                # Preserve old velocity-only callbacks while making their
                # stretching contribution explicit and stage-consistent.
                def body_gradient(points, current_time):
                    return _finite_difference_gradient(body, points, current_time)

            external_gradient += _validate_external_gradient(
                _call_stage_gradient_callback(body_gradient, position, stage_time), count
            )
        elif body_gradient is not None:
            # A device-resident body velocity field may pair with a host-side
            # Jacobian evaluator. The velocity has already been accumulated on
            # ``stage_rates.velocity``; only add its stretching contribution.
            external_gradient += _validate_external_gradient(
                _call_stage_gradient_callback(body_gradient, position, stage_time), count
            )
        elif body_gradient_field is not None:
            _call_stage_gradient_field(
                body_gradient_field,
                stage_state,
                external_gradient,
                count,
            )
        override_replaces_rhs = False
        if override is not None:
            induced_velocity = velocity.copy()
            override_gradient = getattr(self.physics, "velocity_override_gradient", None)
            if hasattr(override, "blend_into"):
                override.blend_into(position, stage_time, induced_velocity, velocity)
                if override_gradient is None:
                    position_gradient = _finite_difference_blend_gradient(
                        override, position, stage_time, induced_velocity
                    )
                    if stage_rates.velocity_gradient is not None:
                        base_gradient = _stage_matrix_array(
                            stage_rates.velocity_gradient, count, self.physics
                        )
                        input_gradient = _finite_difference_blend_input_gradient(
                            override, position, stage_time, induced_velocity
                        )
                        override_gradient_value = position_gradient + np.einsum(
                            "nik,nkj->nij",
                            input_gradient,
                            base_gradient + external_gradient,
                        )
                    else:
                        override_gradient_value = position_gradient
                else:
                    override_gradient_value = _validate_external_gradient(
                        _call_stage_gradient_callback(
                            override_gradient, position, stage_time, velocity
                        ),
                        count,
                    )
                external_gradient += override_gradient_value
            else:
                selected_velocity = np.asarray(
                    _call_stage_velocity_callback(override, position, stage_time, induced_velocity),
                    dtype=velocity.dtype,
                )
                if selected_velocity.shape != (count, 3):
                    raise ValueError(
                        "velocity override must return an array with shape "
                        f"({count}, 3), got {selected_velocity.shape}"
                    )
                if not np.all(np.isfinite(selected_velocity)):
                    raise ValueError("velocity override returned non-finite values")
                velocity[...] = selected_velocity
                override_replaces_rhs = True
                if override_gradient is None:
                    position_gradient = _finite_difference_override_gradient(
                        override, position, stage_time, induced_velocity
                    )
                    if stage_rates.velocity_gradient is not None:
                        base_gradient = _stage_matrix_array(
                            stage_rates.velocity_gradient, count, self.physics
                        )
                        input_gradient = _finite_difference_override_input_gradient(
                            override, position, stage_time, induced_velocity
                        )
                        override_gradient_value = position_gradient + np.einsum(
                            "nik,nkj->nij",
                            input_gradient,
                            base_gradient + external_gradient,
                        )
                    else:
                        override_gradient_value = position_gradient
                else:
                    override_gradient_value = _validate_external_gradient(
                        _call_stage_gradient_callback(
                            override_gradient, position, stage_time, velocity
                        ),
                        count,
                    )
                external_gradient = override_gradient_value
                strength = _stage_array(stage_state.vortex_strength, count, self.physics)
                replacement_rate = np.einsum("nji,nj->ni", override_gradient_value, strength)
                _replace_stage_array(
                    stage_rates.vortex_strength_rate,
                    replacement_rate,
                    count,
                    self.physics,
                )
                if stage_rates.velocity_gradient is not None:
                    _replace_stage_array(
                        stage_rates.velocity_gradient,
                        override_gradient_value,
                        count,
                        self.physics,
                    )
        if hasattr(self.physics, "_upload_vector_array"):
            self.physics._upload_vector_array(velocity, stage_rates.velocity, count)
        else:
            uploaded = np.zeros((self.physics.max_n_particles, 3), dtype=velocity.dtype)
            uploaded[:count] = velocity
            stage_rates.velocity.from_numpy(uploaded)

        # Body/source/override gradients use J[i,j] = d u_i / d x_j.  The
        # vortex-strength stretching convention is J^T Γ, matching the
        # self-induced induction contract.
        if not override_replaces_rhs and np.any(external_gradient):
            strength = _stage_array(stage_state.vortex_strength, count, self.physics)
            external_rate = np.einsum("nji,nj->ni", external_gradient, strength)
            _accumulate_stage_array(
                stage_rates.vortex_strength_rate,
                external_rate,
                count,
                self.physics,
            )
            if stage_rates.velocity_gradient is not None:
                _accumulate_stage_array(
                    stage_rates.velocity_gradient,
                    external_gradient,
                    count,
                    self.physics,
                )


@ti.data_oriented
class AxisymmetricNoSwirlStageProjection:
    """Project stage velocity and strength rates onto the declared symmetry."""

    def __init__(self, physics, orbit_id, axis: int) -> None:
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        self.physics = physics
        self.orbit_id = orbit_id
        self.axis = int(axis)

    def add_stage_rates(
        self, stage_state: StageState, stage_time: float, stage_rates: StageRates
    ) -> None:
        del stage_time
        self.physics.average_axisymmetric_no_swirl_rhs(
            stage_state.position,
            stage_rates.velocity,
            stage_rates.vortex_strength_rate,
            self.orbit_id,
            self.axis,
            stage_state.count,
        )


def _call_stage_velocity_callback(callback, position, stage_time, velocity=None):
    """Call one explicitly stage-aware host callback signature.

    Body callbacks accept ``(stage_position, stage_time)`` and velocity
    overrides accept ``(stage_position, stage_time, current_velocity)``.  A
    callback's own ``TypeError`` is never interpreted as an arity mismatch.
    """
    if velocity is None:
        return callback(position, stage_time)
    return callback(position, stage_time, velocity)


def _call_stage_velocity_field(callback, stage_state, velocity, count):
    """Invoke a device callback with the stage time when it supports it."""
    try:
        callback(stage_state.position, velocity, count, stage_state.time)
    except TypeError:
        callback(stage_state.position, velocity, count)


def _call_stage_gradient_callback(callback, position, stage_time, velocity=None):
    """Call a user gradient callback with an optional blend velocity."""
    if velocity is None:
        return callback(position, stage_time)
    try:
        return callback(position, stage_time, velocity)
    except TypeError:
        return callback(position, stage_time)


def _call_stage_gradient_field(callback, stage_state, gradient, count):
    """Invoke a device gradient hook, accepting both old and time-aware forms."""
    try:
        callback(stage_state.position, gradient, count, stage_state.time)
    except TypeError:
        callback(stage_state.position, gradient, count)


def _validate_external_gradient(value, count: int) -> np.ndarray:
    gradient = np.asarray(value, dtype=np.float64)
    if gradient.shape != (count, 3, 3):
        raise ValueError(
            f"external velocity gradient must have shape ({count}, 3, 3), got {gradient.shape}"
        )
    if not np.all(np.isfinite(gradient)):
        raise ValueError("external velocity gradient contains non-finite values")
    return gradient


def _finite_difference_gradient(callback, position, stage_time) -> np.ndarray:
    """Differentiate a velocity-only callback using a centered stage probe."""
    position = np.asarray(position, dtype=np.float64)
    max_position = float(np.max(np.abs(position))) if position.size else 0.0
    scale = max(1.0, max_position)
    step = max(1.0e-7, 1.0e-5 * scale)
    gradient = np.empty((len(position), 3, 3), dtype=np.float64)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = step
        plus = np.asarray(callback(position + offset, stage_time), dtype=np.float64).reshape(
            len(position), 3
        )
        minus = np.asarray(callback(position - offset, stage_time), dtype=np.float64).reshape(
            len(position), 3
        )
        gradient[:, :, axis] = (plus - minus) / (2.0 * step)
    return gradient


def _finite_difference_override_gradient(
    callback,
    position: np.ndarray,
    stage_time: float,
    induced_velocity: np.ndarray,
) -> np.ndarray:
    """Differentiate a replacement callback with its induced input held fixed."""
    position = np.asarray(position, dtype=np.float64)
    induced_velocity = np.asarray(induced_velocity, dtype=np.float64)
    max_position = float(np.max(np.abs(position))) if position.size else 0.0
    scale = max(1.0, max_position)
    step = max(1.0e-7, 1.0e-5 * scale)
    gradient = np.empty((len(position), 3, 3), dtype=np.float64)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = step
        plus = np.asarray(
            callback(position + offset, stage_time, induced_velocity), dtype=np.float64
        ).reshape(len(position), 3)
        minus = np.asarray(
            callback(position - offset, stage_time, induced_velocity), dtype=np.float64
        ).reshape(len(position), 3)
        gradient[:, :, axis] = (plus - minus) / (2.0 * step)
    return _validate_external_gradient(gradient, len(position))


def _finite_difference_override_input_gradient(
    callback,
    position: np.ndarray,
    stage_time: float,
    induced_velocity: np.ndarray,
) -> np.ndarray:
    """Differentiate a replacement callback with respect to velocity input."""
    position = np.asarray(position, dtype=np.float64)
    induced_velocity = np.asarray(induced_velocity, dtype=np.float64)
    max_velocity = float(np.max(np.abs(induced_velocity))) if induced_velocity.size else 0.0
    scale = max(1.0, max_velocity)
    step = max(1.0e-7, 1.0e-5 * scale)
    gradient = np.empty((len(position), 3, 3), dtype=np.float64)
    for axis in range(3):
        offset = np.zeros_like(induced_velocity)
        offset[:, axis] = step
        plus = np.asarray(
            callback(position, stage_time, induced_velocity + offset), dtype=np.float64
        ).reshape(len(position), 3)
        minus = np.asarray(
            callback(position, stage_time, induced_velocity - offset), dtype=np.float64
        ).reshape(len(position), 3)
        gradient[:, :, axis] = (plus - minus) / (2.0 * step)
    return _validate_external_gradient(gradient, len(position))


def _finite_difference_blend_gradient(
    callback,
    position: np.ndarray,
    stage_time: float,
    induced_velocity: np.ndarray,
) -> np.ndarray:
    """Differentiate the added contribution of an in-place velocity blender."""
    position = np.asarray(position, dtype=np.float64)
    induced_velocity = np.asarray(induced_velocity, dtype=np.float64)
    max_position = float(np.max(np.abs(position))) if position.size else 0.0
    scale = max(1.0, max_position)
    step = max(1.0e-7, 1.0e-5 * scale)
    gradient = np.empty((len(position), 3, 3), dtype=np.float64)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = step
        plus = induced_velocity.copy()
        minus = induced_velocity.copy()
        callback.blend_into(position + offset, stage_time, induced_velocity, plus)
        callback.blend_into(position - offset, stage_time, induced_velocity, minus)
        gradient[:, :, axis] = (plus - minus) / (2.0 * step)
    return _validate_external_gradient(gradient, len(position))


def _finite_difference_blend_input_gradient(
    callback,
    position: np.ndarray,
    stage_time: float,
    induced_velocity: np.ndarray,
) -> np.ndarray:
    """Differentiate a blender's added contribution with respect to its input."""
    position = np.asarray(position, dtype=np.float64)
    induced_velocity = np.asarray(induced_velocity, dtype=np.float64)
    max_velocity = float(np.max(np.abs(induced_velocity))) if induced_velocity.size else 0.0
    scale = max(1.0, max_velocity)
    step = max(1.0e-7, 1.0e-5 * scale)
    gradient = np.empty((len(position), 3, 3), dtype=np.float64)
    for axis in range(3):
        offset = np.zeros_like(induced_velocity)
        offset[:, axis] = step
        plus_input = induced_velocity + offset
        minus_input = induced_velocity - offset
        plus = plus_input.copy()
        minus = minus_input.copy()
        callback.blend_into(position, stage_time, plus_input, plus)
        callback.blend_into(position, stage_time, minus_input, minus)
        gradient[:, :, axis] = (plus - minus) / (2.0 * step)
        gradient[:, axis, axis] -= 1.0
    return _validate_external_gradient(gradient, len(position))


__all__ = [
    "AxisymmetricNoSwirlStageProjection",
    "CallableStageContribution",
    "ExternalStageContribution",
    "ParticleExternalStageContribution",
    "StageRHS",
    "VLMStageContribution",
]
