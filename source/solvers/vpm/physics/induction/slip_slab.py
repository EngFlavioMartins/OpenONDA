"""Full-vector reflected induction for a pair of free-slip span planes.

Only physical particles are advanced. Reflected sources are temporary boundary
representations and are never added to the particle population or diagnostics.
"""

from contextlib import nullcontext
import math
from time import perf_counter

import numpy as np
import taichi as ti

from .base import _STRETCHING_MODES
from .stretching import stretching_rate


@ti.kernel
def _build_reflected_targets(
    original: ti.template(),
    transformed: ti.template(),
    shifts: ti.template(),
    odd_flags: ti.template(),
    start: ti.i32,
    points_per_image: ti.i32,
    image_count: ti.i32,
):
    for i in range(points_per_image * image_count):
        image = i // points_per_image
        point = start + i % points_per_image
        p = original[point]
        z = shifts[image] - p[2] if odd_flags[image] else p[2] - shifts[image]
        transformed[i] = ti.Vector([p[0], p[1], z])


@ti.kernel
def _zero_shell(velocity: ti.template(), gradient: ti.template(), count: ti.i32):
    for i in range(count):
        velocity[i] = ti.Vector.zero(velocity.dtype, 3)
        gradient[i] = ti.Matrix.zero(gradient.dtype, 3, 3)


@ti.kernel
def _shell_maxima(
    velocity: ti.template(),
    gradient: ti.template(),
    max_velocity: ti.template(),
    max_gradient: ti.template(),
    count: ti.i32,
):
    for i in range(count):
        v_sq = ti.cast(0.0, ti.f32)
        g_sq = ti.cast(0.0, ti.f32)
        for a in ti.static(range(3)):
            v_sq += ti.cast(velocity[i][a] * velocity[i][a], ti.f32)
            for b in ti.static(range(3)):
                g_sq += ti.cast(gradient[i][a, b] * gradient[i][a, b], ti.f32)
        ti.atomic_max(max_velocity[None], ti.sqrt(v_sq))
        ti.atomic_max(max_gradient[None], ti.sqrt(g_sq))


@ti.kernel
def _span_violation(
    position: ti.template(),
    violation: ti.template(),
    count: ti.i32,
    z_min: ti.template(),
    z_max: ti.template(),
):
    for i in range(count):
        z = position[i][2]
        excess = ti.max(z_min[None] - z, z - z_max[None], 0.0)
        ti.atomic_max(violation[None], ti.cast(excess, ti.f32))


@ti.kernel
def _add_reflected_results(
    image_velocity: ti.template(),
    image_gradient: ti.template(),
    shifts: ti.template(),
    odd_flags: ti.template(),
    strength: ti.template(),
    velocity: ti.template(),
    rate: ti.template(),
    gradient: ti.template(),
    shell_velocity: ti.template(),
    shell_gradient: ti.template(),
    start: ti.i32,
    points_per_image: ti.i32,
    image_count: ti.i32,
    mode: ti.i32,
    rate_enabled: ti.i32,
    has_velocity: ti.template(),
    has_gradient: ti.template(),
    is_stage: ti.template(),
):
    for local_point in range(points_per_image):
        point = start + local_point
        summed_velocity = ti.Vector.zero(velocity.dtype, 3)
        summed_gradient = ti.Matrix.zero(gradient.dtype, 3, 3)
        for image in range(image_count):
            i = image * points_per_image + local_point
            v = image_velocity[i]
            j = image_gradient[i]
            if odd_flags[image]:
                v[2] = -v[2]
                for a, b in ti.static(ti.ndrange(3, 3)):
                    if (a == 2) != (b == 2):
                        j[a, b] = -j[a, b]
            summed_velocity += v
            summed_gradient += j
        # One target owns each output slot, avoiding atomics across images.
        if ti.static(has_velocity):
            velocity[point] += summed_velocity
        if ti.static(has_gradient):
            gradient[point] += summed_gradient
        shell_velocity[point] += summed_velocity
        shell_gradient[point] += summed_gradient
        if ti.static(is_stage):  # noqa: SIM102 - Taichi static branching
            if rate_enabled == 1:
                rate[point] += stretching_rate(summed_gradient, strength[point], mode)


@ti.data_oriented
class SlipSlabInduction:
    """Wrap a full-3D induction backend with converged free-slip images.

    The planes are ``z_min`` and ``z_max``. For an odd reflection the
    circulation vector, an axial vector, transforms as ``(-Gx,-Gy,Gz)``.
    The image family is evaluated in symmetric translation shells. Each call
    must pass velocity and gradient tests in two consecutive doubling blocks
    of images; otherwise it raises. The block difference is an empirical
    convergence check, not a rigorous bound on the infinite remainder.
    """

    supported_devices = frozenset({"AUTO", "CPU", "VULKAN", "CUDA", "METAL"})
    supports_gradient = supports_variable_core_radius = supports_f64 = True
    supports_target_fields = device_resident = True
    method = "SLIP_SLAB"

    def __init__(
        self,
        base,
        *,
        z_min: float,
        z_max: float,
        tail_tolerance: float = 1e-4,
        max_shells: int = 129,
        velocity_scale: float = 1.0,
        gradient_scale: float = 1.0,
    ):
        if not all(
            math.isfinite(v) for v in (z_min, z_max, tail_tolerance, velocity_scale, gradient_scale)
        ):
            raise ValueError("slab parameters must be finite")
        if (
            z_max <= z_min
            or not 0 < tail_tolerance < 1
            or not isinstance(max_shells, int)
            or max_shells < 3
        ):
            raise ValueError("invalid slab bounds, tail tolerance, or shell count")
        if velocity_scale <= 0 or gradient_scale <= 0:
            raise ValueError("tail reference scales must be positive")
        if (
            getattr(base, "method", None) == "PLANAR"
            or not getattr(base, "supports_target_fields", False)
            or not getattr(base, "supports_gradient", False)
        ):
            raise ValueError("slab requires a full-3D target-capable induction backend")
        self.base = base
        self.z_min, self.z_max = float(z_min), float(z_max)
        self.tail_tolerance = float(tail_tolerance)
        self.max_shells = int(max_shells)
        self.velocity_scale, self.gradient_scale = float(velocity_scale), float(gradient_scale)
        self.stretching_scheme = base.stretching_scheme
        self.supported_kernels = base.supported_kernels
        self.supported_devices = base.supported_devices
        self.supports_f64 = base.supports_f64
        self.supports_variable_core_radius = base.supports_variable_core_radius
        self.device_resident = base.device_resident
        self.kernel = base.kernel
        self.last_tail = None
        self.physics = None

    def build(self):
        return type(self)(
            self.base.build(),
            z_min=self.z_min,
            z_max=self.z_max,
            tail_tolerance=self.tail_tolerance,
            max_shells=self.max_shells,
            velocity_scale=self.velocity_scale,
            gradient_scale=self.gradient_scale,
        )

    def bind(self, physics, *, kernel=None):
        self.base.bind(physics, kernel=kernel)
        self.physics = physics
        self.kernel = self.base.kernel
        capacity = physics.max_n_particles
        targets = max(capacity, physics.max_evaluation_points)
        query_capacity = physics.max_evaluation_points
        dtype = physics.accumulator_dtype
        self._image_velocity = ti.Vector.field(3, dtype=dtype, shape=query_capacity)
        self._image_gradient = ti.Matrix.field(3, 3, dtype=dtype, shape=query_capacity)
        self._query_position = ti.Vector.field(3, dtype=dtype, shape=query_capacity)
        self._image_shifts = ti.field(dtype=dtype, shape=64)
        self._image_odd = ti.field(dtype=ti.i32, shape=64)
        self._block_velocity = ti.Vector.field(3, dtype=dtype, shape=targets)
        self._block_gradient = ti.Matrix.field(3, 3, dtype=dtype, shape=targets)
        self._max_shell_velocity = ti.field(dtype=ti.f32, shape=())
        self._max_shell_gradient = ti.field(dtype=ti.f32, shape=())
        self._span_excess = ti.field(dtype=ti.f32, shape=())
        self._z_min_field = ti.field(dtype=dtype, shape=())
        self._z_max_field = ti.field(dtype=dtype, shape=())
        self._z_min_field[None] = self.z_min
        self._z_max_field[None] = self.z_max
        physics._slip_slab_bounds = (self.z_min, self.z_max)
        return self

    def validate_source_arrays(self, position, strength):
        """Reject physical sources outside the slab before initialization/restart."""
        validate = getattr(self.base, "validate_source_arrays", None)
        if validate is not None:
            validate(position, strength)
        z = np.asarray(position, dtype=np.float64).reshape(-1, 3)[:, 2]
        tol = 64 * np.finfo(np.float64).eps * max(1.0, abs(self.z_min), abs(self.z_max))
        if np.any(z < self.z_min - tol) or np.any(z > self.z_max + tol):
            raise ValueError("physical particle source is outside the slip slab")

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _images(
        self,
        source_position,
        source_strength,
        source_radius,
        target_position,
        source_count,
        target_count,
        velocity,
        gradient,
        *,
        stage_strength=None,
        stage_rate=None,
        rate_enabled=False,
    ):
        if source_count == 0 or target_count == 0:
            return
        started = perf_counter()
        target_evaluations = 0
        target_batches = 0
        length = self.z_max - self.z_min
        consecutive = 0
        previous_relative = math.inf
        block_start = 1
        _zero_shell(self._block_velocity, self._block_gradient, target_count)
        max_queries = self.physics.max_evaluation_points
        fixed_source = getattr(self.base, "fixed_source_targets", None)
        context = (
            fixed_source(
                source_position,
                source_strength,
                source_radius,
                source_count,
                reuse_current_tree=True,
            )
            if fixed_source is not None
            else nullcontext()
        )
        with context:
            shell = 0
            while shell < self.max_shells:
                expected_end = 0 if shell == 0 else (1 if shell == 1 else 2 * shell - 2)
                block_end = min(self.max_shells - 1, expected_end)
                images = []
                for current_shell in range(shell, block_end + 1):
                    for k in (0,) if current_shell == 0 else (-current_shell, current_shell):
                        for odd in (0, 1):
                            if k == 0 and odd == 0:
                                continue
                            # Even images translate by +2kL; odd images reflect
                            # around z_min before the same translation.
                            shift = 2.0 * k * length + (2.0 * self.z_min if odd else 0.0)
                            images.append((shift, odd))
                _zero_shell(self._block_velocity, self._block_gradient, target_count)
                for target_start in range(0, target_count, max_queries):
                    points = min(target_count - target_start, max_queries)
                    images_per_batch = max(1, min(64, max_queries // points))
                    for first in range(0, len(images), images_per_batch):
                        batch = images[first : first + images_per_batch]
                        shifts = np.zeros(64, dtype=np.float64)
                        odds = np.zeros(64, dtype=np.int32)
                        for index, (shift, odd) in enumerate(batch):
                            shifts[index], odds[index] = shift, odd
                        self._image_shifts.from_numpy(
                            shifts.astype(
                                np.float64
                                if self.physics.accumulator_dtype == ti.f64
                                else np.float32
                            )
                        )
                        self._image_odd.from_numpy(odds)
                        _build_reflected_targets(
                            target_position,
                            self._query_position,
                            self._image_shifts,
                            self._image_odd,
                            target_start,
                            points,
                            len(batch),
                        )
                        self.base.evaluate_targets(
                            target_position=self._query_position,
                            source_position=source_position,
                            source_vortex_strength=source_strength,
                            source_core_radius=source_radius,
                            target_velocity=self._image_velocity,
                            target_velocity_gradient=self._image_gradient,
                            target_count=points * len(batch),
                            source_count=source_count,
                            include_freestream=False,
                            background_velocity=self.physics._zero_velocity,
                        )
                        target_evaluations += points * len(batch)
                        target_batches += 1
                        _add_reflected_results(
                            self._image_velocity,
                            self._image_gradient,
                            self._image_shifts,
                            self._image_odd,
                            source_strength if stage_strength is None else stage_strength,
                            velocity if velocity is not None else self._block_velocity,
                            stage_rate if stage_rate is not None else self._block_velocity,
                            gradient if gradient is not None else self._block_gradient,
                            self._block_velocity,
                            self._block_gradient,
                            target_start,
                            points,
                            len(batch),
                            _STRETCHING_MODES[self.stretching_scheme],
                            int(rate_enabled),
                            velocity is not None,
                            gradient is not None,
                            stage_strength is not None,
                        )
                if shell == 0:
                    shell = 1
                    continue
                if block_end != expected_end:
                    break  # An incomplete doubling block cannot certify the tail.
                shell = block_end
                self._max_shell_velocity[None] = 0.0
                self._max_shell_gradient[None] = 0.0
                _shell_maxima(
                    self._block_velocity,
                    self._block_gradient,
                    self._max_shell_velocity,
                    self._max_shell_gradient,
                    target_count,
                )
                velocity_tail = float(self._max_shell_velocity[None])
                gradient_tail = float(self._max_shell_gradient[None])
                relative = max(
                    velocity_tail / self.velocity_scale, gradient_tail / self.gradient_scale
                )
                self.last_tail = {
                    "shell": shell,
                    "block_start": block_start,
                    "relative": relative,
                    "velocity": velocity_tail,
                    "gradient": gradient_tail,
                    "target_evaluations": target_evaluations,
                    "target_batches": target_batches,
                    "seconds": perf_counter() - started,
                }
                decaying = relative <= previous_relative * 1.2
                consecutive = consecutive + 1 if relative <= self.tail_tolerance and decaying else 0
                previous_relative = relative
                if consecutive >= 2:
                    return
                block_start = shell + 1
                shell = block_start
            raise RuntimeError(f"slip-slab image tail did not converge: {self.last_tail}")

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
        self._span_excess[None] = 0.0
        _span_violation(position, self._span_excess, count, self._z_min_field, self._z_max_field)
        if float(self._span_excess[None]) > 1e-6 * (self.z_max - self.z_min):
            raise RuntimeError("RK stage particle escaped the physical slip slab")
        self.base.evaluate_stage(
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
        self._images(
            position,
            vortex_strength,
            core_radius,
            position,
            count,
            count,
            velocity_out,
            velocity_gradient_out,
            stage_strength=vortex_strength,
            stage_rate=vortex_strength_rate_out,
            rate_enabled=strength_rate_enabled,
        )

    def evaluate_targets(
        self,
        *,
        target_position,
        source_position,
        source_vortex_strength,
        source_core_radius,
        target_velocity,
        target_velocity_gradient,
        target_count,
        source_count,
        include_freestream,
        background_velocity,
    ):
        self.base.evaluate_targets(
            target_position=target_position,
            source_position=source_position,
            source_vortex_strength=source_vortex_strength,
            source_core_radius=source_core_radius,
            target_velocity=target_velocity,
            target_velocity_gradient=target_velocity_gradient,
            target_count=target_count,
            source_count=source_count,
            include_freestream=include_freestream,
            background_velocity=background_velocity,
        )
        self._images(
            source_position,
            source_vortex_strength,
            source_core_radius,
            target_position,
            source_count,
            target_count,
            target_velocity,
            target_velocity_gradient,
        )


__all__ = ["SlipSlabInduction"]
