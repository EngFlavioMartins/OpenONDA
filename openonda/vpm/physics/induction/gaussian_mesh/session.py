"""Host validation and immutable source storage for Gaussian slab fields.

This is not a particle integrator or a solver dispatch. Particle induction must
still use its physical pair-mean-core primary operator; ``source_only=True``
is exclusively the coherent primary-plus-image *query* operator. Image rows
beyond the retained shells are added in closed form, so a thin slab needs as
few shells as a thick one. One session owns at most one finite field, even
when its role or source snapshot changes. No caller-visible field is returned
until both mathematical truncation checks and the complete finite-field
evaluation succeed. Finite-grid and floating
point accuracy require independent qualification in addition to these checks.
"""

from dataclasses import dataclass, field
import math
from numbers import Integral
from threading import current_thread
from time import perf_counter
from types import TracebackType

import numpy as np

from ....kernels.registry import kernel_definition
from ....numerics.ieee import ieee_arithmetic, require_round_to_nearest
from ..gaussian_tail import prepare_tail_source, query_tail_bound, validate_source_values
from ..gaussian_tail._interval import add, mul, point
from ..gaussian_tail.error_bounds import ImageTailSeparationError
from .error_bounds import finite_image_correction_bound

# A kernel with an algebraic core defect may need a wider local correction
# than the Gaussian one; it grows by this factor up to this limit.
_CUTOFF_GROWTH = 1.25
_MAX_CUTOFF_GROWTH = 8.0
from typing import TYPE_CHECKING

from .parameters import GaussianMeshParameters

if TYPE_CHECKING:
    from ..gaussian_tail.error_bounds import PreparedTailSource
    from .correction_distance import CorrectionClassificationBound
    from .execution import PortableGaussianImageFields
    from .host_fields import GaussianHostImageFields


def _positive_int(value: int, name: str, maximum: int | None = None) -> int | None:
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

    def __post_init__(self) -> None:
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


def _query_snapshot(targets: np.ndarray, maximum: int) -> np.ndarray | None:
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


def _source_arrays(snapshot: "PreparedTailSource") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.frombuffer(item.data_bytes, dtype=np.dtype(item.dtype)).reshape(item.shape)
        for item in snapshot.source_data.arrays
    )


class _FieldConstructionError(RuntimeError):
    """Internal cleanup result; the original numerical error stays primary."""

    def __init__(
        self, failure: BaseException, field: object, cleanup_failure: BaseException | None = None
    ):
        super().__init__(str(failure))
        self.failure, self.field, self.cleanup_failure = failure, field, cleanup_failure


def _new_field(
    *args, execution_backend: str = "cupy_cuda", portable: bool = False, **kwargs
) -> "GaussianHostImageFields | PortableGaussianImageFields | None":
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


def _classification(
    snapshot: "PreparedTailSource",
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    cutoff: float,
    images: tuple,
) -> "CorrectionClassificationBound":
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
        z_min: float,
        z_max: float,
        tail_tolerance: float,
        max_shells: int,
        velocity_scale: float = 1.0,
        gradient_scale: float = 1.0,
        dtype: str = "float32",
        settings: "GaussianSlabSettings | None" = None,
        execution_backend: str | None = None,
        kernel: str = "GAUSSIAN",
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
        # The mesh carries the kernel's Gaussian partner; sources are given
        # with their actual cores and evaluated with the partner widths.
        definition = kernel_definition(kernel)
        self.kernel, self._split = definition.name, definition.slab_split
        if self.settings.mesh.broadening_ratio < self._split.partner_ratio:
            raise ValueError(
                f"broadening_ratio must cover the {self.kernel} Gaussian partner width "
                f"({self._split.partner_ratio:.6g} cores)"
            )
        self._snapshot = self._field = self._role = self._bounds = None
        self._shells = self._cutoff = None
        self._failed_field = None
        self._cleanup_uncertain = False
        self.closed = False
        self._thread = current_thread()
        self._configuration = self._configuration_key()

    def _configuration_key(self) -> tuple:
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
            self.kernel,
        )

    def _check_thread(self) -> None:
        if current_thread() is not self._thread:
            raise RuntimeError("Gaussian slab field session belongs to another thread")

    @property
    def cleanup_uncertain(self) -> bool:
        """A failed drain/construction must prevent resetting the host runtime."""
        return self._cleanup_uncertain

    def _close_field(self) -> None:
        field, self._field = self._field, None
        self._role = self._bounds = None
        self._shells = self._cutoff = None
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

    def _source(
        self, position: np.ndarray, strength: np.ndarray, radius: np.ndarray
    ) -> tuple["PreparedTailSource", bool]:
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

    def evaluate(
        self,
        position: np.ndarray,
        strength: np.ndarray,
        radius: np.ndarray,
        targets: np.ndarray,
        *,
        source_only: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray, dict[str, bool]]
        | tuple[np.ndarray, np.ndarray, dict[str, str]]
        | None
    ):
        """Bounded image field at ``targets``: images only, or primary plus images.

        The shells and correction radius are chosen so that the proven bounds
        on the omitted image rows and the omitted core correction meet the
        tolerance before any field is computed. Returns velocity, velocity
        gradient and a report of the bounds and the work.
        """
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
        ratio = self._split.partner_ratio
        partner = radius if ratio == 1.0 else np.asarray(radius, dtype=np.float64) * ratio
        snapshot, source_hit = self._source(position, strength, partner)
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
        # Select complete reflected shells with the same completed-tail and
        # local correction bounds. The explicit maximum remains a hard ceiling.
        shells = min(3, self.max_shells - 1)
        if self._field is not None and self._role == source_only:
            shells = max(shells, self._shells)
        envelope = self._split.defect_envelope
        cutoff_growth = 1.0
        if self._field is not None and self._role == source_only:
            cutoff_growth = self._cutoff_growth
        while True:
            grow_cutoff = False
            images = tuple(
                (k, odd)
                for k in range(-shells, shells + 1)
                for odd in (False, True)
                if source_only or (k, odd) != (0, False)
            )
            try:
                with ieee_arithmetic():
                    # The broadening covers the actual cores, not the partners.
                    resolved = self.settings.mesh.resolve(
                        source[2] if ratio == 1.0 else source[2] / ratio
                    )
                    cutoff = resolved.correction_cutoff * cutoff_growth
                    # The completed tail stays smooth over the cardinal stencils
                    # only when it lies a few broadened cores beyond their reach.
                    separation = 3.0 * resolved.tau + resolved.order * resolved.spacing
                    tail = query_tail_bound(
                        snapshot,
                        lower,
                        upper,
                        shells=shells,
                        minimum_separation=separation,
                        kernel_defect=envelope,
                    )
                    classification = _classification(
                        snapshot, lower, upper, cutoff=cutoff, images=images
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
                        kernel_defect=envelope,
                    )
                    velocity_error = float(
                        add(point(tail.velocity_upper), point(correction.velocity_upper)).upper
                    )
                    gradient_error = float(
                        add(point(tail.gradient_upper), point(correction.gradient_upper)).upper
                    )
                    velocity_limit = float(
                        mul(point(self.tolerance), point(self.velocity_scale)).lower
                    )
                    gradient_limit = float(
                        mul(point(self.tolerance), point(self.gradient_scale)).lower
                    )
                if velocity_error <= velocity_limit and gradient_error <= gradient_limit:
                    break
                # A wider correction removes the near part of an algebraic
                # defect that more shells cannot; Gaussian runs never widen.
                correction_share = max(
                    correction.velocity_upper / velocity_limit,
                    correction.gradient_upper / gradient_limit,
                )
                tail_share = max(
                    tail.velocity_upper / velocity_limit, tail.gradient_upper / gradient_limit
                )
                grow_cutoff = (
                    envelope is not None
                    and cutoff_growth < _MAX_CUTOFF_GROWTH
                    and correction_share >= tail_share
                )
                if not grow_cutoff and shells == self.max_shells - 1:
                    self._close_field()
                    raise RuntimeError(
                        "Gaussian slab truncation exceeds unchanged velocity/gradient budget: "
                        f"u={velocity_error}, J={gradient_error}, shells={shells}, "
                        f"correction cutoff={cutoff:.6g} m"
                    )
            except ImageTailSeparationError:
                if shells == self.max_shells - 1:
                    self._close_field()
                    raise
            if grow_cutoff:
                cutoff_growth *= _CUTOFF_GROWTH
            else:
                shells = min(2 * shells + 1, self.max_shells - 1)
        error_bound_finished = perf_counter()
        field_reused = False
        field_close_seconds = field_build_seconds = field_prepare_seconds = 0.0
        field_rebuild_reason = None
        try:
            if self._field is None:
                field_rebuild_reason = "absent" if source_hit else "initial_or_changed_source"
            elif self._role != source_only:
                field_rebuild_reason = "role_changed"
            elif self._shells != shells:
                field_rebuild_reason = "image_shells_changed"
            elif self._cutoff != cutoff:
                field_rebuild_reason = "correction_cutoff_changed"
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
                        cutoff=cutoff,
                        order=resolved.order,
                        dtype=self.dtype,
                        correction_dtype=self.dtype,
                        source_only_primary=source_only,
                        complete_image_tail=True,
                        max_images=len(images),
                        max_query_points=self.settings.max_query_points,
                        max_scratch_bytes=self.settings.max_scratch_bytes,
                        max_correction_bytes=self.settings.max_correction_bytes,
                        max_plan_bytes=self.settings.max_plan_bytes,
                        max_total_bytes=self.settings.max_total_bytes,
                        kernel=self.kernel,
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
                self._shells, self._cutoff = shells, cutoff
                self._cutoff_growth = cutoff_growth
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
                "correction_cutoff": cutoff,
                "kernel": self.kernel,
                "finite_evaluation": finite,
                "seconds": perf_counter() - started,
                "error_bound_scope": "mathematical truncation only; not finite-grid/GPU error",
            },
        )

    def close(self) -> None:
        """Close the cached field and source snapshot; idempotent."""
        self._check_thread()
        if self.cleanup_uncertain:
            raise RuntimeError(
                "Gaussian slab GPU cleanup remains uncertain; do not reset its runtime"
            )
        if not self.closed:
            self._close_field()
            self._snapshot = None
            self.closed = True

    def __enter__(self) -> "GaussianSlabFieldSession":
        if self.closed:
            raise RuntimeError("Gaussian slab field session is closed")
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        failure: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup_failure:
            if failure is None:
                raise
            failure.add_note(f"Gaussian session cleanup also failed: {cleanup_failure!r}")
