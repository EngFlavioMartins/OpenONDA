"""Full-vector reflected induction for a pair of free-slip span planes.

Only physical particles are advanced. Reflected sources are temporary boundary
representations and are never added to the particle population or diagnostics.
"""

from contextlib import nullcontext
import math
from numbers import Integral
from time import perf_counter

import numpy as np
import taichi as ti

from .base import _STRETCHING_MODES
from .fmm.target_geometry import disjoint_fields, method_bindings, scratch_fields, standard_methods
from .gaussian_mesh.session import GaussianSlabPolicy
from .stretching import stretching_rate


def _mesh_runtime_is_cuda():
    runtime = ti.lang.impl.get_runtime()
    return runtime.prog is not None and ti.lang.impl.current_cfg().arch == ti.cuda


def _new_mesh_session(**kwargs):
    # CUDA dependencies remain lazy; CPU uses the same Gaussian operator.
    from .gaussian_mesh.session import GaussianSlabFieldSession

    return GaussianSlabFieldSession(**kwargs)


def _admit_mesh_installation():
    from .gaussian_mesh.availability import require_gaussian_mesh_runtime

    return require_gaussian_mesh_runtime()


def _admit_host_mesh_installation():
    from .gaussian_mesh.availability import require_gaussian_host_runtime

    return require_gaussian_host_runtime()


@ti.kernel
def _download_vector_prefix(source: ti.template(), result: ti.types.ndarray(ndim=2), count: ti.i32):
    for i in range(count):
        for axis in ti.static(range(3)):
            result[i, axis] = source[i][axis]


@ti.kernel
def _download_scalar_prefix(source: ti.template(), result: ti.types.ndarray(ndim=1), count: ti.i32):
    for i in range(count):
        result[i] = source[i]


def _active_snapshots(requests):
    """Copy each distinct input field once, preserving its active values/dtype."""
    sizes = {}
    for source, count in requests:
        sizes[source] = max(count, sizes.get(source, 0))
    snapshots = {}
    for source, count in sizes.items():
        vector = hasattr(source, "n")
        dtype = np.float32 if source.dtype == ti.f32 else np.float64
        result = np.empty((count, 3) if vector else (count,), dtype=dtype)
        if count:
            if vector:
                _download_vector_prefix(source, result, count)
            else:
                _download_scalar_prefix(source, result, count)
        snapshots[source] = result
    return tuple(snapshots[source][:count] for source, count in requests)


@ti.kernel
def _publish_mesh_results(
    host_velocity: ti.types.ndarray(ndim=2),
    host_gradient: ti.types.ndarray(ndim=3),
    velocity: ti.template(),
    gradient: ti.template(),
    strength: ti.template(),
    rate: ti.template(),
    background: ti.template(),
    count: ti.i32,
    mode: ti.i32,
    rate_enabled: ti.i32,
    is_stage: ti.template(),
    has_velocity: ti.template(),
    has_gradient: ti.template(),
    include_freestream: ti.template(),
):
    """Publish a complete admitted host result; no partial mesh is exposed."""
    for i in range(count):
        v = ti.Vector.zero(velocity.dtype, 3)
        j = ti.Matrix.zero(gradient.dtype, 3, 3)
        for a in ti.static(range(3)):
            v[a] = host_velocity[i, a]
            for b in ti.static(range(3)):
                j[a, b] = host_gradient[i, a, b]
        if ti.static(has_velocity):
            if ti.static(is_stage):
                velocity[i] += v
            else:
                if ti.static(include_freestream):
                    v += background[None]
                velocity[i] = v
        if ti.static(has_gradient):
            if ti.static(is_stage):
                gradient[i] += j
            else:
                gradient[i] = j
        if ti.static(is_stage):
            if rate_enabled == 1:
                rate[i] += stretching_rate(j, strength[i], mode)
            else:
                rate[i] = ti.Vector.zero(rate.dtype, 3)


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


@ti.kernel
def _add_image_block_results(
    image_velocity: ti.template(),
    image_gradient: ti.template(),
    strength: ti.template(),
    velocity: ti.template(),
    rate: ti.template(),
    gradient: ti.template(),
    shell_velocity: ti.template(),
    shell_gradient: ti.template(),
    start: ti.i32,
    count: ti.i32,
    mode: ti.i32,
    rate_enabled: ti.i32,
    has_velocity: ti.template(),
    has_gradient: ti.template(),
    is_stage: ti.template(),
):
    """Accumulate one already-parity-corrected complete image block."""
    for local in range(count):
        point = start + local
        v = image_velocity[local]
        j = image_gradient[local]
        if ti.static(has_velocity):
            velocity[point] += v
        if ti.static(has_gradient):
            gradient[point] += j
        shell_velocity[point] += v
        shell_gradient[point] += j
        if ti.static(is_stage):  # noqa: SIM102 -- Taichi compile-time branch.
            if rate_enabled == 1:
                rate[point] += stretching_rate(j, strength[point], mode)


@ti.data_oriented
class SlipSlabInduction:
    """Full-3D induction between two free-slip span planes.

    The planes are ``z_min`` and ``z_max``. For an odd reflection the
    circulation vector, an axial vector, transforms as ``(-Gx,-Gy,Gz)``.
    Gaussian images use a field mesh with enclosed tail/correction admission.
    Particle primary induction retains pair-mean cores; arbitrary queries use
    one source-only primary-plus-image field. ``gaussian_mesh_policy`` controls
    resolution and resources, and cannot disable that operator. Other radial
    kernels use reflected sources with an empirical image-block tail test.
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
        gaussian_mesh_policy: GaussianSlabPolicy = GaussianSlabPolicy(),
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
        if type(gaussian_mesh_policy) is not GaussianSlabPolicy:
            raise TypeError(
                "gaussian_mesh_policy requires GaussianSlabPolicy; it cannot disable Gaussian induction"
            )
        self.device_resident = False  # Gaussian images require host snapshots.
        self.gaussian_mesh_policy = gaussian_mesh_policy
        self._mesh_binding = None
        self._mesh_session = None
        self._mesh_cleanup_error = None
        self._mesh_execution_backend = None

    def build(self):
        return type(self)(
            self.base.build(),
            z_min=self.z_min,
            z_max=self.z_max,
            tail_tolerance=self.tail_tolerance,
            max_shells=self.max_shells,
            velocity_scale=self.velocity_scale,
            gradient_scale=self.gradient_scale,
            gaussian_mesh_policy=self.gaussian_mesh_policy,
        )

    def bind(self, physics, *, kernel=None):
        mesh = physics.particle_kernel == "GAUSSIAN"
        if mesh:
            selected_kernel = self.base.kernel if kernel is None else kernel
            self._validate_mesh_runtime(physics, selected_kernel)
            # Dependency failure must precede base rebinding or any new
            # Taichi/mesh field allocation. Other kernels stay CuPy-lazy.
            backend = self.gaussian_mesh_policy.backend
            if backend == "cpu" or not _mesh_runtime_is_cuda():
                _admit_host_mesh_installation()
                execution_backend = "cpu"
            else:
                from .gaussian_mesh.availability import GaussianMeshUnavailableError

                try:
                    _admit_mesh_installation()
                except GaussianMeshUnavailableError:
                    if backend != "auto":
                        raise
                    _admit_host_mesh_installation()
                    execution_backend = "cpu"
                else:
                    execution_backend = "cupy_cuda"
        self.close_mesh_session()
        self._mesh_binding = None
        self.base.bind(physics, kernel=kernel)
        self.physics = physics
        self.kernel = self.base.kernel
        capacity = physics.max_n_particles
        targets = max(capacity, physics.max_evaluation_points)
        query_capacity = physics.max_evaluation_points
        dtype = physics.accumulator_dtype
        if mesh:
            # The reflected-kernel walkers are unused. Retain tiny template
            # placeholders for omitted outputs, not another O(N) image buffer.
            targets = query_capacity = 1
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
        if mesh:
            self._mesh_execution_backend = execution_backend
            self._mesh_binding = self._mesh_configuration()
        return self

    def _validate_mesh_runtime(self, physics, kernel):
        from ...kernels.base import make_vortex_kernel

        if type(self.gaussian_mesh_policy) is not GaussianSlabPolicy:
            raise TypeError("gaussian_mesh_policy requires GaussianSlabPolicy")
        if self.max_shells > 1024:
            raise ValueError("Gaussian mesh max_shells exceeds the explicit 1024-shell capacity")
        if kernel != make_vortex_kernel("GAUSSIAN") or physics.particle_kernel != "GAUSSIAN":
            raise ValueError("Gaussian mesh induction requires the standard GAUSSIAN kernel")
        if physics.accumulator_dtype not in (ti.f32, ti.f64):
            raise ValueError("Gaussian mesh induction requires binary32 or binary64 fields")
        if self.gaussian_mesh_policy.backend == "cupy_cuda" and not _mesh_runtime_is_cuda():
            raise RuntimeError("Gaussian mesh induction requires an initialized CUDA runtime")

    def _mesh_configuration(self):
        return (
            id(self.physics),
            id(ti.lang.impl.get_runtime().prog),
            self.gaussian_mesh_policy,
            self.z_min,
            self.z_max,
            self.tail_tolerance,
            self.max_shells,
            self.velocity_scale,
            self.gradient_scale,
            self.stretching_scheme,
            self.base.stretching_scheme,
            str(self.physics.accumulator_dtype),
            self.kernel,
            self.base.kernel,
        )

    def _uses_mesh(self):
        if self.physics is None:
            raise RuntimeError("slab induction is unbound; bind the induction backend")
        if self.physics.particle_kernel != "GAUSSIAN":
            if self._mesh_binding is not None:
                raise RuntimeError("slab mesh controls changed; rebind the induction backend")
            return False
        if self.physics is None or self._mesh_binding != self._mesh_configuration():
            raise RuntimeError(
                "slab mesh controls changed or are unbound; rebind the induction backend"
            )
        self._validate_mesh_runtime(self.physics, self.kernel)
        if self._mesh_cleanup_error is not None:
            raise RuntimeError(
                "Gaussian mesh cleanup remains uncertain"
            ) from self._mesh_cleanup_error
        return True

    @staticmethod
    def _mesh_count(count, maximum):
        if isinstance(count, bool) or not isinstance(count, Integral) or not 0 <= count <= maximum:
            raise ValueError("mesh evaluation count exceeds its explicit capacity")
        return int(count)

    def _mesh_field(self, value, count, components, *, output=False):
        if (
            value is None
            or len(getattr(value, "shape", ())) != 1
            or value.shape[0] < count
            or getattr(value, "dtype", None) not in (ti.f32, ti.f64)
        ):
            raise ValueError("mesh evaluation requires bounded floating-point Taichi fields")
        actual = (
            () if not hasattr(value, "n") else (value.n,) if value.m == 1 else (value.n, value.m)
        )
        if actual != components or (output and value.dtype != self.physics.accumulator_dtype):
            raise ValueError(
                "mesh field components or output precision do not match the bound operator"
            )

    def _mesh_fields(
        self,
        source_position,
        source_strength,
        source_radius,
        targets,
        source_count,
        target_count,
        velocity,
        gradient,
        rate=None,
        background=None,
    ):
        source_count = self._mesh_count(source_count, self.physics.max_n_particles)
        target_count = self._mesh_count(target_count, self.gaussian_mesh_policy.max_query_points)
        for value, components in (
            (source_position, (3,)),
            (source_strength, (3,)),
            (source_radius, ()),
        ):
            self._mesh_field(value, source_count, components)
        self._mesh_field(targets, target_count, (3,))
        outputs = tuple(value for value in (velocity, gradient, rate) if value is not None)
        for value, components in ((velocity, (3,)), (gradient, (3, 3)), (rate, (3,))):
            if value is not None:
                self._mesh_field(value, target_count, components, output=True)
        reads = (source_position, source_strength, source_radius, targets, background)
        if not disjoint_fields(reads, outputs) or any(
            not disjoint_fields((value,), outputs[index + 1 :])
            for index, value in enumerate(outputs)
        ):
            raise ValueError("Gaussian mesh source, target and output storage must not alias")
        return source_count, target_count

    def _mesh_evaluate(
        self, position, strength, radius, targets, source_count, target_count, *, source_only
    ):
        self.last_tail = None
        if self._mesh_session is None:
            self._mesh_session = _new_mesh_session(
                z_min=self.z_min,
                z_max=self.z_max,
                tail_tolerance=self.tail_tolerance,
                max_shells=self.max_shells,
                velocity_scale=self.velocity_scale,
                gradient_scale=self.gradient_scale,
                policy=self.gaussian_mesh_policy,
                execution_backend=self._mesh_execution_backend,
                dtype="float32" if self.physics.accumulator_dtype == ti.f32 else "float64",
            )
        u, j, diagnostics = self._mesh_session.evaluate(
            *_active_snapshots(
                (
                    (position, source_count),
                    (strength, source_count),
                    (radius, source_count),
                    (targets, target_count),
                )
            ),
            source_only=source_only,
        )
        dtype = np.dtype(np.float32 if self.physics.accumulator_dtype == ti.f32 else np.float64)
        if (
            u.shape != (target_count, 3)
            or j.shape != (target_count, 3, 3)
            or u.dtype != dtype
            or j.dtype != dtype
            or not np.isfinite(u).all()
            or not np.isfinite(j).all()
        ):
            raise RuntimeError(
                "Gaussian mesh did not return complete finite fields at bound precision"
            )
        return np.ascontiguousarray(u), np.ascontiguousarray(j), diagnostics

    def _record_mesh_tail(self, diagnostics):
        # Display only; actual admission used outward interval sums and limits
        # inside the session before any field was published.
        u = diagnostics.get("velocity_tail_bound", 0.0) + diagnostics.get(
            "velocity_correction_bound", 0.0
        )
        j = diagnostics.get("gradient_tail_bound", 0.0) + diagnostics.get(
            "gradient_correction_bound", 0.0
        )
        self.last_tail = {
            "shell": diagnostics.get("shell", 0),
            "velocity": u,
            "gradient": j,
            "relative": max(u / self.velocity_scale, j / self.gradient_scale),
            "seconds": diagnostics.get("seconds", 0.0),
            "contract": self.gaussian_mesh_policy.tail_contract,
            "mesh": diagnostics,
        }

    def close_mesh_session(self):
        """Close this wrapper's private Gaussian field owner, never its base."""
        if self._mesh_cleanup_error is not None:
            raise RuntimeError(
                "Gaussian mesh cleanup remains uncertain"
            ) from self._mesh_cleanup_error
        if self._mesh_session is not None:
            try:
                self._mesh_session.close()
                if self._mesh_session.cleanup_uncertain:
                    raise RuntimeError("Gaussian mesh owner cleanup remains uncertain")
            except BaseException as error:
                self._mesh_cleanup_error = error
                raise
            self._mesh_session = None

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

    def _image_geometry_context(
        self, target_position, target_count, tile_capacity, *, read_fields, write_fields
    ):
        """Only the unchanged built-in shell loop can promise target immutability."""
        from .fmm.device import FMMInduction

        if type(self.base) is not FMMInduction or not standard_methods(
            self, SlipSlabInduction, _GEOMETRY_SLAB_METHODS
        ):
            return nullcontext()
        prepare = getattr(self.base, "_fixed_image_targets", None)
        if prepare is None:
            return nullcontext()
        return prepare(
            target_position,
            target_count,
            tile_capacity,
            read_fields=read_fields,
            write_fields=tuple(write_fields) + scratch_fields(self),
        )

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
        if self.physics.particle_kernel == "GAUSSIAN":
            raise RuntimeError("Gaussian slip-slab images use the field mesh operator")
        if source_count == 0 or target_count == 0:
            return
        from .fmm.targets import TargetBlockNotWorthwhile

        started = perf_counter()
        target_evaluations = 0
        target_batches = 0
        target_local_evaluations = 0
        declined_blocks = []
        block_evaluator = (
            getattr(self.base, "evaluate_image_block", None)
            if getattr(self.base, "supports_image_blocks", False)
            else None
        )
        length = self.z_max - self.z_min
        consecutive = 0
        previous_relative = math.inf
        block_start = 1
        _zero_shell(self._block_velocity, self._block_gradient, target_count)
        max_queries = self.physics.max_evaluation_points
        tile_queries = min(max_queries, getattr(self.base, "max_image_block_targets", max_queries))
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
        geometry_context = (
            self._image_geometry_context(
                target_position,
                target_count,
                tile_queries,
                read_fields=(
                    target_position,
                    source_position,
                    source_strength,
                    source_radius,
                    stage_strength,
                ),
                write_fields=(velocity, gradient, stage_rate),
            )
            if block_evaluator is not None
            else nullcontext()
        )
        # The geometry lease precedes source-scope entry: admission must not
        # mistake an active immutable-source context for a complete-field hit.
        with geometry_context, context:
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
                for target_start in range(0, target_count, tile_queries):
                    points = min(target_count - target_start, tile_queries)
                    if block_evaluator is not None:
                        try:
                            block_evaluator(
                                source_position=source_position,
                                source_vortex_strength=source_strength,
                                source_core_radius=source_radius,
                                source_count=source_count,
                                target_position=target_position,
                                target_start=target_start,
                                target_count=points,
                                images=images,
                                target_velocity=self._image_velocity,
                                target_velocity_gradient=self._image_gradient,
                            )
                        except TargetBlockNotWorthwhile as error:
                            declined_blocks.append(
                                {"shell": shell, "target_start": target_start, **error.diagnostics}
                            )
                        else:
                            _add_image_block_results(
                                self._image_velocity,
                                self._image_gradient,
                                source_strength if stage_strength is None else stage_strength,
                                velocity if velocity is not None else self._block_velocity,
                                stage_rate if stage_rate is not None else self._block_velocity,
                                gradient if gradient is not None else self._block_gradient,
                                self._block_velocity,
                                self._block_gradient,
                                target_start,
                                points,
                                _STRETCHING_MODES[self.stretching_scheme],
                                int(rate_enabled),
                                velocity is not None,
                                gradient is not None,
                                stage_strength is not None,
                            )
                            # Logical image-target count remains comparable to
                            # old traversal; physical L2P is counted separately.
                            target_evaluations += points * len(images)
                            target_local_evaluations += points
                            target_batches += 1
                            continue
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
                        target_local_evaluations += points * len(batch)
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
                    "target_local_evaluations": target_local_evaluations,
                    "declined_blocks": declined_blocks,
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
        mesh = self._uses_mesh()
        if mesh:
            if velocity_out is None or vortex_strength_rate_out is None:
                raise ValueError("particle mesh stages require velocity and strength-rate outputs")
            _, count = self._mesh_fields(
                position,
                vortex_strength,
                core_radius,
                position,
                count,
                count,
                velocity_out,
                velocity_gradient_out,
                vortex_strength_rate_out,
            )
            if not count:
                return
        self._span_excess[None] = 0.0
        _span_violation(position, self._span_excess, count, self._z_min_field, self._z_max_field)
        if float(self._span_excess[None]) > 1e-6 * (self.z_max - self.z_min):
            raise RuntimeError("RK stage particle escaped the physical slip slab")
        if mesh:
            # Complete all image/correction/tail work before the primary call
            # can publish anything. The physical pair-mean primary is unchanged.
            host_u, host_j, diagnostics = self._mesh_evaluate(
                position,
                vortex_strength,
                core_radius,
                position,
                count,
                count,
                source_only=False,
            )
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
        if mesh:
            _publish_mesh_results(
                host_u,
                host_j,
                velocity_out,
                self._block_gradient if velocity_gradient_out is None else velocity_gradient_out,
                vortex_strength,
                vortex_strength_rate_out,
                self.physics._zero_velocity,
                count,
                _STRETCHING_MODES[self.stretching_scheme],
                int(strength_rate_enabled),
                True,
                True,
                velocity_gradient_out is not None,
                False,
            )
            self._record_mesh_tail(diagnostics)
            return
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
        if self._uses_mesh():
            if include_freestream and (
                getattr(background_velocity, "shape", None) != ()
                or getattr(background_velocity, "n", None) != 3
                or getattr(background_velocity, "m", None) != 1
                or background_velocity.dtype != self.physics.accumulator_dtype
            ):
                raise ValueError("freestream requires a matching scalar Taichi vector field")
            background = background_velocity if include_freestream else self.physics._zero_velocity
            source_count, target_count = self._mesh_fields(
                source_position,
                source_vortex_strength,
                source_core_radius,
                target_position,
                source_count,
                target_count,
                target_velocity,
                target_velocity_gradient,
                background=background,
            )
            if not target_count or (target_velocity is None and target_velocity_gradient is None):
                return
            host_u, host_j, diagnostics = self._mesh_evaluate(
                source_position,
                source_vortex_strength,
                source_core_radius,
                target_position,
                source_count,
                target_count,
                source_only=True,
            )
            # Coherent SOURCE-ONLY primary+images: deliberately no base target
            # call, which would double count primary and spoil wall cancellation.
            _publish_mesh_results(
                host_u,
                host_j,
                self._block_velocity if target_velocity is None else target_velocity,
                self._block_gradient
                if target_velocity_gradient is None
                else target_velocity_gradient,
                self._block_velocity,
                self._block_velocity,
                background,
                target_count,
                0,
                0,
                False,
                target_velocity is not None,
                target_velocity_gradient is not None,
                bool(include_freestream),
            )
            self._record_mesh_tail(diagnostics)
            return
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


_GEOMETRY_SLAB_METHODS = method_bindings(SlipSlabInduction)

__all__ = ["SlipSlabInduction"]
