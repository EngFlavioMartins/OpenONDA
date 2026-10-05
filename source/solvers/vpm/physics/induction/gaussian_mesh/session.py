"""Host validation and immutable source storage for Gaussian slab fields.

This is not a particle integrator or a solver dispatch. Particle induction must
still use its physical pair-mean-core primary operator; ``source_only=True``
is exclusively the coherent primary-plus-image *query* operator. One session
owns at most one finite field, even when its role or source snapshot changes.
No caller-visible field is returned until both mathematical truncation checks
and the complete finite-field evaluation succeed. Finite-grid and floating
point accuracy require independent qualification in addition to these checks.
"""

from dataclasses import dataclass, field
import math
from numbers import Integral
from threading import current_thread
from time import perf_counter

import numpy as np

from ....numerics.ieee import ieee_arithmetic, require_round_to_nearest
from ..gaussian_tail import prepare_tail_source, query_tail_bound, validate_source_values
from ..gaussian_tail._interval import add, mul, point
from .error_bounds import finite_image_correction_bound
from .parameters import GaussianMeshParameters


def _positive_int(value, name, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value <= 0
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} requires a bounded positive integer")
    return int(value)


@dataclass(frozen=True)
class GaussianSlabSettings:
    """Explicit, case-independent operator and resource controls.

    The enclosed whole-tail remainder test and mesh resolution belong in the
    numerical identity. Resource caps select bounded storage and never permit
    coarsening the field or loosening the truncation tolerance.
    """

    mesh: GaussianMeshParameters = field(default_factory=GaussianMeshParameters)
    tail_error_method: str = "gaussian_interval_remainder_v1"
    backend: str = "auto"
    max_sources: int = 1_000_000
    max_query_points: int = 1_000_000
    max_scratch_bytes: int = 2 * 1024**3
    max_correction_bytes: int = 256 * 1024**2
    max_plan_bytes: int = 128 * 1024**2
    max_total_bytes: int = 2304 * 1024**2

    def __post_init__(self):
        if type(self.mesh) is not GaussianMeshParameters:
            raise TypeError("an explicit GaussianMeshParameters value is required")
        if self.tail_error_method != "gaussian_interval_remainder_v1" or self.backend not in {
            "auto",
            "cpu",
            "cupy_cuda",
        }:
            raise ValueError("unsupported Gaussian slab backend or tail conditions")
        for name in (
            "max_sources",
            "max_query_points",
            "max_scratch_bytes",
            "max_correction_bytes",
            "max_plan_bytes",
            "max_total_bytes",
        ):
            maximum = (
                1_000_000
                if name == "max_sources"
                else 2**30
                if name == "max_query_points"
                else None
            )
            object.__setattr__(self, name, _positive_int(getattr(self, name), name, maximum))
        if self.max_plan_bytes >= self.max_scratch_bytes:
            raise ValueError("plan reserve must be smaller than smooth-field cap")
        if self.max_scratch_bytes + self.max_correction_bytes > self.max_total_bytes:
            raise ValueError("array-pool limits exceed combined cap")


def _query_snapshot(targets, maximum):
    raw = np.asarray(targets)
    if (
        raw.dtype.kind != "f"
        or raw.dtype.itemsize not in (4, 8)
        or raw.ndim != 2
        or raw.shape[1:] != (3,)
        or len(raw) > maximum
    ):
        raise ValueError("bounded binary32/binary64 target coordinates (M,3) required")
    # Immutable bytes also protect against caller mutation during a device call.
    result = np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype=raw.dtype).reshape(raw.shape)
    if not np.isfinite(result).all():
        raise ValueError("finite query coordinates required")
    return result


def _source_arrays(snapshot):
    return tuple(
        np.frombuffer(item.data_bytes, dtype=np.dtype(item.dtype)).reshape(item.shape)
        for item in snapshot.source_data.arrays
    )


class _FieldConstructionError(RuntimeError):
    """Internal cleanup result; the original numerical error stays primary."""

    def __init__(self, failure, field, cleanup_failure=None):
        super().__init__(str(failure))
        self.failure, self.field, self.cleanup_failure = failure, field, cleanup_failure


def _new_field(*args, execution_backend="cupy_cuda", portable=False, **kwargs):
    # Importing settings/session support must not import CuPy or choose a device.
    if portable:
        from .execution import PortableGaussianImageFields as GaussianImageFields

        kwargs["execution_backend"] = execution_backend
    elif execution_backend == "cpu":
        from .host_fields import GaussianHostImageFields as GaussianImageFields
    else:
        from .fields import GaussianImageFields
    # Retain a handle even when __init__ fails, so its cleanup result is
    # established rather than guessed from the exception type or message.
    field = GaussianImageFields.__new__(GaussianImageFields)
    try:
        field.__init__(*args, **kwargs)
    except BaseException as failure:
        try:
            field.close()
        except BaseException as cleanup_failure:
            raise _FieldConstructionError(failure, field, cleanup_failure) from failure
        raise _FieldConstructionError(failure, field) from failure
    return field


def _classification(snapshot, lower, upper, *, cutoff, images):
    # The classifier proof is deliberately separate from the mathematical
    # correction envelope. No unproven assumption r >= configured cutoff.
    from .correction_distance import correction_classification_bound

    return correction_classification_bound(snapshot, lower, upper, cutoff=cutoff, images=images)


class GaussianSlabFieldSession:
    """Serial, exact-content field reuse with explicit mathematical validation.

    Caller supplies complete physical source snapshots, not mutable device
    identities or epoch guesses. Source/role changes or queries outside the
    prepared logical interpolation domain close old private GPU storage BEFORE
    allocating a replacement. A failed evaluation
    revokes the field; it cannot be reused as a successful cached field.
    """

    def __init__(
        self,
        *,
        z_min,
        z_max,
        tail_tolerance,
        max_shells,
        velocity_scale=1.0,
        gradient_scale=1.0,
        dtype="float32",
        settings=None,
        execution_backend=None,
    ):
        if (
            not all(
                math.isfinite(value)
                for value in (z_min, z_max, tail_tolerance, velocity_scale, gradient_scale)
            )
            or z_max <= z_min
            or not 0 < tail_tolerance < 1
            or velocity_scale <= 0
            or gradient_scale <= 0
        ):
            raise ValueError("finite ordered slab, positive scales and tail tolerance required")
        self.max_shells = _positive_int(max_shells, "max_shells", 1024)
        if self.max_shells < 3 or dtype not in ("float32", "float64"):
            raise ValueError("at least three shells and float32/float64 outputs required")
        if settings is not None and type(settings) is not GaussianSlabSettings:
            raise TypeError("GaussianSlabSettings required")
        self.settings = GaussianSlabSettings() if settings is None else settings
        self.execution_backend = execution_backend or (
            "cupy_cuda" if self.settings.backend == "cupy_cuda" else "cpu"
        )
        if self.execution_backend not in {"cpu", "cupy_cuda"}:
            raise ValueError("finite field execution requires cpu or cupy_cuda")
        if self.settings.backend != "auto" and self.execution_backend != self.settings.backend:
            raise ValueError("explicit Gaussian execution backend differs from settings")
        self.z_min, self.z_max = float(z_min), float(z_max)
        self.tolerance = float(tail_tolerance)
        self.velocity_scale, self.gradient_scale = float(velocity_scale), float(gradient_scale)
        self.dtype = dtype
        self._snapshot = self._field = self._role = self._bounds = None
        self._failed_field = None
        self._cleanup_uncertain = False
        self.closed = False
        self._thread = current_thread()
        self._configuration = self._configuration_key()

    def _configuration_key(self):
        return (
            self.z_min,
            self.z_max,
            self.tolerance,
            self.max_shells,
            self.velocity_scale,
            self.gradient_scale,
            self.dtype,
            self.settings,
            self.execution_backend,
        )

    def _check_thread(self):
        if current_thread() is not self._thread:
            raise RuntimeError("Gaussian slab field session belongs to another thread")

    @property
    def cleanup_uncertain(self):
        """A failed drain/construction must prevent resetting the host runtime."""
        return self._cleanup_uncertain

    def _close_field(self):
        field, self._field = self._field, None
        self._role = self._bounds = None
        if field is not None:
            try:
                field.close()
            except BaseException:
                # A failed drain/release must not permit a second GPU field
                # on a later call. Retain the failed resource for diagnosis.
                self._failed_field = field
                self._cleanup_uncertain = True
                self.closed = True
                raise

    def _source(self, position, strength, radius):
        if self._snapshot is not None:
            try:
                validate_source_values(
                    self._snapshot, position, strength, radius, z_min=self.z_min, z_max=self.z_max
                )
            except ValueError:
                # A changed particle count can fail the old snapshot's shape
                # cap before value comparison. Fresh preparation below still
                # validates every new source against this session's real cap.
                pass
            else:
                return self._snapshot, True
        self._close_field()
        self._snapshot = None
        with ieee_arithmetic():
            snapshot = prepare_tail_source(
                position,
                strength,
                radius,
                z_min=self.z_min,
                z_max=self.z_max,
                max_sources=self.settings.max_sources,
            )
        self._snapshot = snapshot
        return snapshot, False

    def evaluate(self, position, strength, radius, targets, *, source_only=False):
        self._check_thread()
        if self.closed:
            raise RuntimeError("Gaussian slab field session is closed")
        # Descriptor construction outside the tail-bound scope must use the
        # same rounding rule as the enclosed world-shift arithmetic. Taichi's
        # RN+FTZ mode is supported; arbitrary directed rounding is not.
        require_round_to_nearest()
        if self._configuration_key() != self._configuration:
            self._close_field()
            raise RuntimeError("Gaussian slab controls changed; construct a new session")
        if type(source_only) is not bool:
            raise TypeError("source-only role must be explicit boolean")
        started = perf_counter()
        query = _query_snapshot(targets, self.settings.max_query_points)
        snapshot, source_hit = self._source(position, strength, radius)
        source_finished = perf_counter()
        if not snapshot.source_count or not len(query):
            self._close_field()
            return (
                np.zeros((len(query), 3), dtype=self.dtype),
                np.zeros((len(query), 3, 3), dtype=self.dtype),
                {"empty_field": True, "source_snapshot_hit": source_hit},
            )
        source = _source_arrays(snapshot)
        lower, upper = query.min(axis=0), query.max(axis=0)
        # Keep EVERY finite shell allowed by the explicit ceiling. No source,
        # descriptor, or primary query contribution is silently discarded.
        shells = self.max_shells - 1
        images = tuple(
            (k, odd)
            for k in range(-shells, shells + 1)
            for odd in (False, True)
            if source_only or (k, odd) != (0, False)
        )
        with ieee_arithmetic():
            resolved = self.settings.mesh.resolve(source[2])
            tail = query_tail_bound(snapshot, lower, upper, shells=shells)
            classification = _classification(
                snapshot, lower, upper, cutoff=resolved.correction_cutoff, images=images
            )
            omitted_radius = classification.omitted_distance_lower
            correction = finite_image_correction_bound(
                snapshot,
                lower,
                upper,
                images=images,
                tau=resolved.tau,
                omitted_distance_lower=omitted_radius,
                max_images=len(images),
            )
            velocity_error = float(
                add(point(tail.velocity_upper), point(correction.velocity_upper)).upper
            )
            gradient_error = float(
                add(point(tail.gradient_upper), point(correction.gradient_upper)).upper
            )
            # Division is not used to approve the check: multiply the threshold
            # outward DOWN so a rounded quotient cannot accept an over-budget sum.
            velocity_limit = float(mul(point(self.tolerance), point(self.velocity_scale)).lower)
            gradient_limit = float(mul(point(self.tolerance), point(self.gradient_scale)).lower)
        if velocity_error > velocity_limit or gradient_error > gradient_limit:
            self._close_field()
            raise RuntimeError(
                "Gaussian slab truncation exceeds unchanged velocity/gradient budget: "
                f"u={velocity_error}, J={gradient_error}, shells={shells}"
            )
        error_bound_finished = perf_counter()
        field_reused = False
        field_close_seconds = field_build_seconds = field_prepare_seconds = 0.0
        field_rebuild_reason = None
        try:
            if self._field is None:
                field_rebuild_reason = "absent" if source_hit else "initial_or_changed_source"
            elif self._role != source_only:
                field_rebuild_reason = "role_changed"
            elif not self._field.can_evaluate_targets(query):
                field_rebuild_reason = "outside_logical_stencil_domain"
            else:
                field_reused = True
            if not field_reused:
                phase_started = perf_counter()
                self._close_field()
                field_close_seconds = perf_counter() - phase_started
                try:
                    phase_started = perf_counter()
                    self._field = _new_field(
                        *source,
                        query,
                        zmin=self.z_min,
                        zmax=self.z_max,
                        execution_backend=self.execution_backend,
                        portable=self.settings.backend == "auto",
                        tau=resolved.tau,
                        spacing=resolved.spacing,
                        cutoff=resolved.correction_cutoff,
                        order=resolved.order,
                        dtype=self.dtype,
                        correction_dtype=self.dtype,
                        source_only_primary=source_only,
                        max_images=len(images),
                        max_query_points=self.settings.max_query_points,
                        max_scratch_bytes=self.settings.max_scratch_bytes,
                        max_correction_bytes=self.settings.max_correction_bytes,
                        max_plan_bytes=self.settings.max_plan_bytes,
                        max_total_bytes=self.settings.max_total_bytes,
                    )
                    field_build_seconds = perf_counter() - phase_started
                except _FieldConstructionError as outcome:
                    if outcome.cleanup_failure is not None:
                        self._failed_field = outcome.field
                        self._cleanup_uncertain = True
                        self.closed = True
                        outcome.failure.add_note(
                            f"Gaussian field cleanup failed: {outcome.cleanup_failure!r}"
                        )
                    # Preserve the numerical cause, not a wrapper -> primary
                    # -> wrapper exception cycle or a secondary close error.
                    raise outcome.failure from outcome.failure.__cause__
                except BaseException:
                    # Construction may itself have failed during cleanup,
                    # before returning any handle. Do not retry allocations
                    # when that cleanup result cannot be established.
                    self._cleanup_uncertain = True
                    self.closed = True
                    raise
                phase_started = perf_counter()
                self._field.prepare(images)
                field_prepare_seconds = perf_counter() - phase_started
                # The field's retained stencil domain decides reuse; these
                # initial query bounds are descriptive only.
                self._role, self._bounds = source_only, (lower.copy(), upper.copy())
            # FTZ/DAZ may change subnormal host shifts even under RN. Match
            # the actual immutable list used by correction, not a hash or an
            # algebraically equivalent expression. No query is published on
            # a mismatch, including on an otherwise valid exact-source hit.
            if self._field._prepared_world_images != classification.world_images:
                raise RuntimeError("runtime image descriptors differ from validated world images")
            velocity, gradient, finite = self._field.evaluate_prepared(query)
            # Explicit transfers; no undocumented Taichi/CuPy pointer interop.
            velocity = self._field.cp.asnumpy(velocity)
            gradient = self._field.cp.asnumpy(gradient)
            if not np.isfinite(velocity).all() or not np.isfinite(gradient).all():
                raise FloatingPointError("nonfinite complete Gaussian slab query")
        except BaseException as failure:
            try:
                self._close_field()
            except BaseException as cleanup_failure:
                # Keep the operation that failed as the caller-visible error.
                # _close_field retains the failed resource and prevents any
                # subsequent allocation/reset when its drain is uncertain.
                failure.add_note(f"Gaussian field cleanup also failed: {cleanup_failure!r}")
            raise
        return (
            velocity,
            gradient,
            {
                "source_sha256": snapshot.source_sha256,
                "source_snapshot_hit": source_hit,
                "field_reused": field_reused,
                "source_only_primary": source_only,
                "field_rebuild_reason": field_rebuild_reason,
                "query_count": len(query),
                "query_lower": lower.tolist(),
                "query_upper": upper.tolist(),
                "source_snapshot_seconds": source_finished - started,
                "tail_bound_seconds": error_bound_finished - source_finished,
                "field_close_seconds": field_close_seconds,
                "field_build_seconds": field_build_seconds,
                "field_prepare_seconds": field_prepare_seconds,
                "finite_images": len(images),
                "shell": shells,
                "velocity_tail_bound": tail.velocity_upper,
                "gradient_tail_bound": tail.gradient_upper,
                "velocity_correction_bound": correction.velocity_upper,
                "gradient_correction_bound": correction.gradient_upper,
                "omitted_distance_lower": omitted_radius,
                "finite_evaluation": finite,
                "seconds": perf_counter() - started,
                "error_bound_scope": "mathematical truncation only; not finite-grid/GPU error",
            },
        )

    def close(self):
        self._check_thread()
        if self.cleanup_uncertain:
            raise RuntimeError(
                "Gaussian slab GPU cleanup remains uncertain; do not reset its runtime"
            )
        if not self.closed:
            self._close_field()
            self._snapshot = None
            self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError("Gaussian slab field session is closed")
        return self

    def __exit__(self, exc_type, failure, traceback):
        try:
            self.close()
        except BaseException as cleanup_failure:
            if failure is None:
                raise
            failure.add_note(f"Gaussian session cleanup also failed: {cleanup_failure!r}")
