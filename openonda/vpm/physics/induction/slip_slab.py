"""Full-vector induction between a pair of free-slip span planes.

Only physical particles are advanced. Their slab images are summed by the
image field mesh and are never added to the particle population or
diagnostics.
"""

from collections.abc import Callable, Mapping
from contextlib import nullcontext
import math
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import taichi as ti
from taichi.lang.field import ScalarField
from taichi.lang.matrix import MatrixField

from ...kernels.base import RadialVortexKernel, make_vortex_kernel
from ...kernels.registry import kernel_definition
from .base import _STRETCHING_MODES, InductionMethod
from .gaussian_mesh.session import GaussianSlabSettings
from .stretching import stretching_rate

if TYPE_CHECKING:
    from ..base import PhysicsBase
    from .gaussian_mesh.session import GaussianSlabFieldSession


def _storage_members(
    field: ti.Field,
) -> tuple | tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | tuple[tuple[int, int]]:
    """Identify every placed component, including aliases via another wrapper."""
    if type(field) not in (ScalarField, MatrixField):
        return None
    try:
        return tuple(
            (member.ptr.snode().get_snode_tree_id(), member.ptr.snode().id)
            for member in field._get_field_members()
        )
    except (AttributeError, RuntimeError, TypeError):
        return None


def _disjoint_fields(
    read_fields: tuple[ti.Field, ti.Field, ti.Field, ti.Field, None]
    | tuple[ti.Field, ti.Field, ti.Field, ti.Field, ti.Field]
    | tuple[ti.Field],
    write_fields: object,
) -> bool:
    read, write = set(), set()
    for fields, result in ((read_fields, read), (write_fields, write)):
        for field in fields:
            if field is not None:
                members = _storage_members(field)
                if not members:
                    return False
                result.update(members)
    return not read.intersection(write)


def _mesh_runtime_is_cuda() -> bool:
    runtime = ti.lang.impl.get_runtime()
    return runtime.prog is not None and ti.lang.impl.current_cfg().arch == ti.cuda


def _new_mesh_session(**kwargs) -> "GaussianSlabFieldSession":
    # CUDA dependencies remain lazy; CPU uses the same Gaussian operator.
    from .gaussian_mesh.session import GaussianSlabFieldSession

    return GaussianSlabFieldSession(**kwargs)


def _check_cuda_mesh_installation() -> dict[str, str] | None:
    from .gaussian_mesh.availability import require_gaussian_mesh_runtime

    return require_gaussian_mesh_runtime()


def _check_cpu_mesh_installation() -> dict[str, str]:
    from .gaussian_mesh.availability import require_gaussian_host_runtime

    return require_gaussian_host_runtime()


def _active_snapshots(
    physics: "PhysicsBase",
    requests: tuple[
        tuple[ti.Field, int], tuple[ti.Field, int], tuple[ti.Field, int], tuple[ti.Field, int]
    ]
    | tuple[tuple[ti.Field, int], tuple[ti.Field, int], tuple[ti.Field, int]]
    | tuple[tuple[ti.Field, int], tuple[ti.Field, int]],
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray]
):
    """Copy each distinct input field once, preserving its active values/dtype."""
    sizes = {}
    for source, count in requests:
        sizes[source] = max(count, sizes.get(source, 0))
    snapshots = {}
    for source, count in sizes.items():
        download = (
            physics.download_vector_field
            if hasattr(source, "n")
            else physics._download_scalar_field
        )
        snapshots[source] = download(source, count)
    return tuple(snapshots[source][:count] for source, count in requests)


@ti.kernel
def _publish_mesh_chunk(
    host_velocity: ti.types.ndarray(ndim=2),
    host_gradient: ti.types.ndarray(ndim=3),
    velocity: ti.template(),
    gradient: ti.template(),
    strength: ti.template(),
    rate: ti.template(),
    background: ti.template(),
    start: ti.i32,
    count: ti.i32,
    mode: ti.i32,
    rate_enabled: ti.i32,
    is_stage: ti.template(),
    has_velocity: ti.template(),
    has_gradient: ti.template(),
    include_freestream: ti.template(),
):
    """Publish one checked field chunk at its original active indices."""
    for i in range(count):
        target = start + i
        v = ti.Vector.zero(velocity.dtype, 3)
        j = ti.Matrix.zero(gradient.dtype, 3, 3)
        for a in ti.static(range(3)):
            v[a] = host_velocity[i, a]
            for b in ti.static(range(3)):
                j[a, b] = host_gradient[i, a, b]
        if ti.static(has_velocity):
            if ti.static(is_stage):
                velocity[target] += v
            else:
                if ti.static(include_freestream):
                    v += background[None]
                velocity[target] = v
        if ti.static(has_gradient):
            if ti.static(is_stage):
                gradient[target] += j
            else:
                gradient[target] = j
        if ti.static(is_stage):
            if rate_enabled == 1:
                rate[target] += stretching_rate(j, strength[target], mode)
            else:
                rate[target] = ti.Vector.zero(rate.dtype, 3)


def _publish_mesh_results(
    physics: "PhysicsBase",
    host_velocity: np.ndarray,
    host_gradient: np.ndarray,
    velocity: ti.Field,
    gradient: ti.Field,
    strength: ti.Field,
    rate: ti.Field,
    background: ti.Field,
    count: int,
    mode: int,
    rate_enabled: int,
    is_stage: bool,
    has_velocity: bool,
    has_gradient: bool,
    include_freestream: bool,
) -> None:
    """Transfer checked fields through the shared fixed-size staging buffers."""
    vector = physics._host_transfer_buffer("vector", velocity, "upload")
    matrix = physics._host_transfer_buffer("matrix", gradient, "upload")
    capacity = len(vector)
    for start in range(0, count, capacity):
        active = min(capacity, count - start)
        vector[:active] = host_velocity[start : start + active]
        matrix[:active] = host_gradient[start : start + active]
        _publish_mesh_chunk(
            vector,
            matrix,
            velocity,
            gradient,
            strength,
            rate,
            background,
            start,
            active,
            mode,
            rate_enabled,
            is_stage,
            has_velocity,
            has_gradient,
            include_freestream,
        )
        ti.sync()


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


@ti.data_oriented
class SlipSlabInduction:
    """Full-3D induction between two free-slip span planes.

    The planes are ``z_min`` and ``z_max``. For an odd reflection the
    circulation vector, an axial vector, transforms as ``(-Gx,-Gy,Gz)``.
    The images of every particle kernel use one field mesh: a broad Gaussian
    on the mesh, the kernel's own core restored within a local correction
    radius, and the image rows beyond the summed shells added in closed form.
    The omitted correction and rows are bounded rigorously before any field is
    published. Particle primary induction keeps the base backend's pair-mean
    cores; arbitrary queries use one source-only primary-plus-image field.
    ``gaussian_mesh_settings`` controls resolution and resources.

    Planar flow is one particle layer between planes one particle spacing
    apart. ``Numerics.domain_bounds`` z must equal the planes. Every viscous
    scheme and stabilization operator adds the slab images of its own field.

    Parameters
    ----------
    base : InductionMethod
        Free-space, target-capable induction with velocity gradients, e.g.
        ``DirectInduction()`` or ``FMMInduction()``.
    z_min, z_max : float
        Lower and upper slip planes in m.
    tail_tolerance : float, default=1e-4
        Bound on the omitted image field, relative to ``velocity_scale`` and
        ``gradient_scale``: what the closed-form completion of the omitted
        rows leaves, plus the core correction beyond its radius.
    max_shells : int, default=129
        Hard limit on the summed image shells; at most 1024.
    velocity_scale, gradient_scale : float, default=1.0
        Reference velocity in m/s and velocity gradient in 1/s for the bound.
    gaussian_mesh_settings : GaussianSlabSettings
        Mesh resolution and memory limits of the image field.
    """

    supported_devices = frozenset({"auto", "cpu", "vulkan", "cuda", "metal"})
    supports_gradient = supports_variable_core_radius = supports_f64 = True
    supports_target_fields = device_resident = True
    method = "slip_slab"

    def __init__(
        self,
        base: InductionMethod,
        *,
        z_min: float,
        z_max: float,
        tail_tolerance: float = 1e-4,
        max_shells: int = 129,
        velocity_scale: float = 1.0,
        gradient_scale: float = 1.0,
        gaussian_mesh_settings: GaussianSlabSettings = GaussianSlabSettings(),
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
        if not getattr(base, "supports_target_fields", False) or not getattr(
            base, "supports_gradient", False
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
        if type(gaussian_mesh_settings) is not GaussianSlabSettings:
            raise TypeError("gaussian_mesh_settings requires GaussianSlabSettings")
        self.device_resident = False  # Mesh images require host snapshots.
        self.gaussian_mesh_settings = gaussian_mesh_settings
        self._mesh_binding = None
        self._mesh_session = None
        self._mesh_cleanup_error = None
        self._mesh_execution_backend = None

    def build(self) -> "SlipSlabInduction":
        """Return an unbound copy with a freshly built base induction."""
        return type(self)(
            self.base.build(),
            z_min=self.z_min,
            z_max=self.z_max,
            tail_tolerance=self.tail_tolerance,
            max_shells=self.max_shells,
            velocity_scale=self.velocity_scale,
            gradient_scale=self.gradient_scale,
            gaussian_mesh_settings=self.gaussian_mesh_settings,
        )

    def bind(
        self, physics: object, *, kernel: RadialVortexKernel | None = None
    ) -> "SlipSlabInduction":
        """Bind the base induction and the image fields to ``physics``.

        Also records the slip planes on ``physics`` so grid diffusion adds the
        slab images.
        """
        selected_kernel = make_vortex_kernel(physics.particle_kernel) if kernel is None else kernel
        self._validate_mesh_runtime(physics, selected_kernel)
        # Dependency failure must precede base rebinding or any new
        # Taichi/mesh field allocation.
        backend = self.gaussian_mesh_settings.backend
        if backend == "cpu" or not _mesh_runtime_is_cuda():
            _check_cpu_mesh_installation()
            execution_backend = "cpu"
        else:
            from .gaussian_mesh.availability import GaussianMeshUnavailableError

            try:
                _check_cuda_mesh_installation()
            except GaussianMeshUnavailableError:
                if backend != "auto":
                    raise
                _check_cpu_mesh_installation()
                execution_backend = "cpu"
            else:
                execution_backend = "cupy_cuda"
        self.close_mesh_session()
        self._mesh_binding = None
        self.base.bind(physics, kernel=selected_kernel)
        self.physics = physics
        self.kernel = self.base.kernel
        dtype = physics.accumulator_dtype
        # Template placeholders for outputs a caller omits, not O(N) buffers.
        self._block_velocity = ti.Vector.field(3, dtype=dtype, shape=1)
        self._block_gradient = ti.Matrix.field(3, 3, dtype=dtype, shape=1)
        self._span_excess = ti.field(dtype=ti.f32, shape=())
        self._z_min_field = ti.field(dtype=dtype, shape=())
        self._z_max_field = ti.field(dtype=dtype, shape=())
        self._z_min_field[None] = self.z_min
        self._z_max_field[None] = self.z_max
        physics._slip_slab_bounds = (self.z_min, self.z_max)
        self._mesh_execution_backend = execution_backend
        self._mesh_binding = self._mesh_configuration()
        return self

    def _validate_mesh_runtime(self, physics: "PhysicsBase", kernel: "RadialVortexKernel") -> None:
        if type(self.gaussian_mesh_settings) is not GaussianSlabSettings:
            raise TypeError("gaussian_mesh_settings requires GaussianSlabSettings")
        if self.max_shells > 1024:
            raise ValueError("slab image mesh max_shells exceeds the explicit 1024-shell capacity")
        if kernel != make_vortex_kernel(physics.particle_kernel):
            raise ValueError(f"slab images require the registered {physics.particle_kernel} kernel")
        ratio = kernel_definition(physics.particle_kernel).slab_split.partner_ratio
        if self.gaussian_mesh_settings.mesh.broadening_ratio < ratio:
            raise ValueError(
                f"{physics.particle_kernel} slab images need GaussianMeshParameters "
                f"broadening_ratio >= {ratio:.6g}, the width of its Gaussian partner"
            )
        if physics.accumulator_dtype not in (ti.f32, ti.f64):
            raise ValueError("Gaussian mesh induction requires binary32 or binary64 fields")
        if self.gaussian_mesh_settings.backend == "cupy_cuda" and not _mesh_runtime_is_cuda():
            raise RuntimeError("Gaussian mesh induction requires an initialized CUDA runtime")

    def _mesh_configuration(self) -> tuple:
        return (
            id(self.physics),
            id(ti.lang.impl.get_runtime().prog),
            self.gaussian_mesh_settings,
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

    def _check_mesh_binding(self) -> None:
        if self.physics is None:
            raise RuntimeError("slab induction is unbound; bind the induction backend")
        if self._mesh_binding != self._mesh_configuration():
            raise RuntimeError(
                "slab mesh controls changed or are unbound; rebind the induction backend"
            )
        self._validate_mesh_runtime(self.physics, self.kernel)
        if self._mesh_cleanup_error is not None:
            raise RuntimeError(
                "slab image mesh cleanup remains uncertain"
            ) from self._mesh_cleanup_error

    @staticmethod
    def _mesh_count(count: int, maximum: int) -> int:
        if isinstance(count, bool) or not isinstance(count, Integral) or not 0 <= count <= maximum:
            raise ValueError("mesh evaluation count exceeds its explicit capacity")
        return int(count)

    def _mesh_field(
        self,
        value: ti.Field | None,
        count: int,
        components: tuple[int, ...],
        *,
        output: bool = False,
    ) -> None:
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
        source_position: ti.Field,
        source_strength: ti.Field,
        source_radius: ti.Field,
        targets: ti.Field,
        source_count: int,
        target_count: int,
        velocity: ti.Field | None,
        gradient: ti.Field | None,
        rate: ti.Field | None = None,
        background: ti.Field | None = None,
    ) -> tuple[int, int] | None:
        source_count = self._mesh_count(source_count, self.physics.max_n_particles)
        target_count = self._mesh_count(target_count, self.gaussian_mesh_settings.max_query_points)
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
        if not _disjoint_fields(reads, outputs) or any(
            not _disjoint_fields((value,), outputs[index + 1 :])
            for index, value in enumerate(outputs)
        ):
            raise ValueError("slab mesh source, target and output storage must not alias")
        return source_count, target_count

    def _mesh_evaluate(
        self,
        position: ti.Field,
        strength: ti.Field,
        radius: ti.Field,
        targets: ti.Field,
        source_count: int,
        target_count: int,
        *,
        source_only: bool,
    ) -> (
        tuple[np.ndarray, np.ndarray, dict[str, int]]
        | tuple[np.ndarray, np.ndarray, dict[str, str]]
        | None
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
                settings=self.gaussian_mesh_settings,
                execution_backend=self._mesh_execution_backend,
                dtype="float32" if self.physics.accumulator_dtype == ti.f32 else "float64",
                kernel=self.physics.particle_kernel,
            )
        u, j, diagnostics = self._mesh_session.evaluate(
            *_active_snapshots(
                self.physics,
                (
                    (position, source_count),
                    (strength, source_count),
                    (radius, source_count),
                    (targets, target_count),
                ),
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

    def _record_mesh_tail(self, diagnostics: Mapping[str, object]) -> None:
        # Display only; actual validation used outward interval sums and limits
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
            "tail_error_method": self.gaussian_mesh_settings.tail_error_method,
            "mesh": diagnostics,
        }

    def close_mesh_session(self) -> None:
        """Close this wrapper's private Gaussian field solver, never its base."""
        if self._mesh_cleanup_error is not None:
            raise RuntimeError(
                "Gaussian mesh cleanup remains uncertain"
            ) from self._mesh_cleanup_error
        if self._mesh_session is not None:
            try:
                self._mesh_session.close()
                if self._mesh_session.cleanup_uncertain:
                    raise RuntimeError("Gaussian mesh solver cleanup remains uncertain")
            except BaseException as error:
                self._mesh_cleanup_error = error
                raise
            self._mesh_session = None

    def validate_source_arrays(self, position: np.ndarray, strength: np.ndarray) -> None:
        """Reject physical sources outside the slab before initialization/restart."""
        validate = getattr(self.base, "validate_source_arrays", None)
        if validate is not None:
            validate(position, strength)
        z = np.asarray(position, dtype=np.float64).reshape(-1, 3)[:, 2]
        tol = 64 * np.finfo(np.float64).eps * max(1.0, abs(self.z_min), abs(self.z_max))
        if np.any(z < self.z_min - tol) or np.any(z > self.z_max + tol):
            raise ValueError("physical particle source is outside the slip slab")

    def __getattr__(self, name: str) -> Callable | None:
        return getattr(self.base, name)

    def evaluate_stage(
        self,
        *,
        position: object,
        vortex_strength: object,
        core_radius: object,
        count: int,
        velocity_out: object,
        vortex_strength_rate_out: object,
        velocity_gradient_out: object = None,
        strength_rate_enabled: bool = True,
        stage_time: float = 0.0,
    ) -> None:
        """Velocity, gradient and strength rate of the particles and all their images.

        The base backend evaluates the primary field with pair-mean cores and
        the mesh adds every slab image; nothing is published unless the image
        bounds meet the tolerance.
        """
        self._check_mesh_binding()
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
        reserve = getattr(self.base, "stage_workspace", None)
        with (
            reserve(count, reclaim=self.close_mesh_session)
            if reserve is not None
            else nullcontext()
        ):
            # Validate images before primary publication. Completed host
            # results survive cache reclamation if FMM lists must grow.
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
        _publish_mesh_results(
            self.physics,
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

    def evaluate_targets(
        self,
        *,
        target_position: object,
        source_position: object,
        source_vortex_strength: object,
        source_core_radius: object,
        target_velocity: object,
        target_velocity_gradient: object,
        target_count: int,
        source_count: int,
        include_freestream: bool,
        background_velocity: object,
    ) -> None:
        """Velocity and gradient at arbitrary targets: sources plus all their images.

        One source-only mesh field gives both, so the base backend is not called.
        """
        self._check_mesh_binding()
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
            self.physics,
            host_u,
            host_j,
            self._block_velocity if target_velocity is None else target_velocity,
            self._block_gradient if target_velocity_gradient is None else target_velocity_gradient,
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


__all__ = ["SlipSlabInduction"]
