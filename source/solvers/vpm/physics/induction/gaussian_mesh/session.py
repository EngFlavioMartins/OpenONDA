"""Host admission and exact-source ownership for optional Gaussian slab fields.

This is not a particle integrator or a solver dispatch. Particle induction must
still use its physical pair-mean-core primary operator; ``source_only=True``
is exclusively the coherent primary-plus-image *query* operator. One session
owns at most one GPU field owner, even when its role or source snapshot changes.
No caller-visible field is returned until both mathematical truncation gates
and the complete finite-field evaluation succeed. Finite-grid and floating
point accuracy require independent qualification in addition to these gates.
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
from .policy import GaussianMeshParameters


def _positive_int(value, name, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} requires a bounded positive integer")
    return int(value)


@dataclass(frozen=True)
class GaussianSlabPolicy:
    """Explicit, case-independent operator and resource controls.

    The contract name must be fingerprinted by any adopting solver. This
    changes the legacy empirical image-block test to an enclosed whole-tail
    remainder test, not to a looser tolerance. Mesh parameters also belong in
    the numerical identity. Resource caps never authorize coarsening.
    """

    mesh: GaussianMeshParameters = field(default_factory=GaussianMeshParameters)
    tail_contract: str = "gaussian_interval_remainder_v1"
    backend: str = "auto"
    max_sources: int = 1_000_000
    max_query_points: int = 1_000_000
    max_scratch_bytes: int = 2*1024**3
    max_correction_bytes: int = 256*1024**2
    max_plan_bytes: int = 128*1024**2
    max_total_bytes: int = 2304*1024**2

    def __post_init__(self):
        if type(self.mesh) is not GaussianMeshParameters:
            raise TypeError("an explicit GaussianMeshParameters value is required")
        if (self.tail_contract != "gaussian_interval_remainder_v1"
                or self.backend not in {"auto", "cpu", "cupy_cuda"}):
            raise ValueError("unsupported Gaussian slab backend or tail contract")
        for name in ("max_sources", "max_query_points", "max_scratch_bytes",
                     "max_correction_bytes", "max_plan_bytes", "max_total_bytes"):
            maximum = 1_000_000 if name == "max_sources" else 2**30 if name == "max_query_points" else None
            object.__setattr__(self, name, _positive_int(getattr(self, name), name, maximum))
        if self.max_plan_bytes >= self.max_scratch_bytes:
            raise ValueError("plan reserve must be smaller than smooth-field cap")
        if self.max_scratch_bytes+self.max_correction_bytes > self.max_total_bytes:
            raise ValueError("array-pool limits exceed combined cap")


def _query_snapshot(targets, maximum):
    raw = np.asarray(targets)
    if (raw.dtype.kind != "f" or raw.dtype.itemsize not in (4, 8)
            or raw.ndim != 2 or raw.shape[1:] != (3,) or len(raw) > maximum):
        raise ValueError("bounded binary32/binary64 target coordinates (M,3) required")
    # Immutable bytes also protect against caller mutation during a device call.
    result = np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype=raw.dtype).reshape(raw.shape)
    if not np.isfinite(result).all():
        raise ValueError("finite query coordinates required")
    return result


def _source_arrays(snapshot):
    return tuple(np.frombuffer(item.payload, dtype=np.dtype(item.dtype)).reshape(item.shape)
                 for item in snapshot.source_identity.arrays)


class _FieldConstructionError(RuntimeError):
    """Internal ownership outcome; the original numerical error stays primary."""

    def __init__(self, failure, owner, cleanup_failure=None):
        super().__init__(str(failure))
        self.failure, self.owner, self.cleanup_failure = failure, owner, cleanup_failure


def _new_field_owner(*args, execution_backend="cupy_cuda", portable=False, **kwargs):
    # Importing policy/session support must not import CuPy or choose a device.
    if portable:
        from .execution import PortableGaussianImageFields as GaussianImageFields
        kwargs["execution_backend"] = execution_backend
    elif execution_backend == "cpu":
        from .host_fields import GaussianHostImageFields as GaussianImageFields
    else:
        from .fields import GaussianImageFields
    # Retain a handle even when __init__ fails, so its ownership outcome is
    # established rather than guessed from the exception type or message.
    owner = GaussianImageFields.__new__(GaussianImageFields)
    try:
        owner.__init__(*args, **kwargs)
    except BaseException as failure:
        try:
            owner.close()
        except BaseException as cleanup_failure:
            raise _FieldConstructionError(failure, owner, cleanup_failure) from failure
        raise _FieldConstructionError(failure, owner) from failure
    return owner


def _classification(snapshot, lower, upper, *, cutoff, images):
    # The classifier proof is deliberately separate from the mathematical
    # correction envelope. No unproven assumption r >= configured cutoff.
    from .correction_admission import correction_classification_bound
    return correction_classification_bound(snapshot, lower, upper, cutoff=cutoff, images=images)


class GaussianSlabFieldSession:
    """Serial, exact-content field reuse with explicit mathematical admission.

    Caller supplies complete physical source snapshots, not mutable device
    identities or epoch guesses. Source/role changes or queries outside the
    prepared logical interpolation domain close old private GPU storage BEFORE
    allocating a replacement. A failed evaluation
    revokes the owner; it cannot be reused as a successful cached field.
    """

    def __init__(self, *, z_min, z_max, tail_tolerance, max_shells,
                 velocity_scale=1., gradient_scale=1., dtype="float32", policy=None,
                 execution_backend=None):
        if (not all(math.isfinite(value) for value in
                    (z_min, z_max, tail_tolerance, velocity_scale, gradient_scale))
                or z_max <= z_min or not 0 < tail_tolerance < 1
                or velocity_scale <= 0 or gradient_scale <= 0):
            raise ValueError("finite ordered slab, positive scales and tail tolerance required")
        self.max_shells = _positive_int(max_shells, "max_shells", 1024)
        if self.max_shells < 3 or dtype not in ("float32", "float64"):
            raise ValueError("at least three shells and float32/float64 outputs required")
        if policy is not None and type(policy) is not GaussianSlabPolicy:
            raise TypeError("GaussianSlabPolicy required")
        self.policy = GaussianSlabPolicy() if policy is None else policy
        self.execution_backend = execution_backend or (
            "cupy_cuda" if self.policy.backend == "cupy_cuda" else "cpu")
        if self.execution_backend not in {"cpu", "cupy_cuda"}:
            raise ValueError("finite field execution requires cpu or cupy_cuda")
        if self.policy.backend != "auto" and self.execution_backend != self.policy.backend:
            raise ValueError("explicit Gaussian execution backend differs from policy")
        self.z_min, self.z_max = float(z_min), float(z_max)
        self.tolerance = float(tail_tolerance)
        self.velocity_scale, self.gradient_scale = float(velocity_scale), float(gradient_scale)
        self.dtype = dtype
        self._snapshot = self._owner = self._role = self._bounds = None
        self._failed_owner = None
        self._cleanup_uncertain = False
        self.closed = False
        self._thread = current_thread()
        self._configuration = self._configuration_key()

    def _configuration_key(self):
        return (self.z_min, self.z_max, self.tolerance, self.max_shells,
                self.velocity_scale, self.gradient_scale, self.dtype, self.policy,
                self.execution_backend)

    def _admit_thread(self):
        if current_thread() is not self._thread:
            raise RuntimeError("Gaussian slab field session belongs to another thread")

    @property
    def cleanup_uncertain(self):
        """A failed drain/construction must prevent resetting the host runtime."""
        return self._cleanup_uncertain

    def _release_owner(self):
        owner, self._owner = self._owner, None
        self._role = self._bounds = None
        if owner is not None:
            try:
                owner.close()
            except BaseException:
                # A failed drain/release must not permit a second GPU owner
                # on a later call. Retain the failed resource for diagnosis.
                self._failed_owner = owner
                self._cleanup_uncertain = True
                self.closed = True
                raise

    def _source(self, position, strength, radius):
        if self._snapshot is not None:
            try:
                validate_source_values(self._snapshot, position, strength, radius,
                                       z_min=self.z_min, z_max=self.z_max)
            except ValueError:
                # A changed particle count can fail the old snapshot's shape
                # cap before value comparison. Fresh preparation below still
                # validates every new source against this session's real cap.
                pass
            else:
                return self._snapshot, True
        self._release_owner()
        self._snapshot = None
        with ieee_arithmetic():
            snapshot = prepare_tail_source(position, strength, radius,
                z_min=self.z_min, z_max=self.z_max, max_sources=self.policy.max_sources)
        self._snapshot = snapshot
        return snapshot, False

    def evaluate(self, position, strength, radius, targets, *, source_only=False):
        self._admit_thread()
        if self.closed:
            raise RuntimeError("Gaussian slab field session is closed")
        # Descriptor construction outside the certificate scope must use the
        # same rounding rule as the enclosed world-shift arithmetic. Taichi's
        # RN+FTZ mode is supported; arbitrary directed rounding is not.
        require_round_to_nearest()
        if self._configuration_key() != self._configuration:
            self._release_owner()
            raise RuntimeError("Gaussian slab controls changed; construct a new session")
        if type(source_only) is not bool:
            raise TypeError("source-only role must be explicit boolean")
        started = perf_counter()
        query = _query_snapshot(targets, self.policy.max_query_points)
        snapshot, source_hit = self._source(position, strength, radius)
        source_finished = perf_counter()
        if not snapshot.source_count or not len(query):
            self._release_owner()
            return (np.zeros((len(query), 3), dtype=self.dtype),
                    np.zeros((len(query), 3, 3), dtype=self.dtype),
                    {"empty_field": True, "source_snapshot_hit": source_hit})
        source = _source_arrays(snapshot)
        lower, upper = query.min(axis=0), query.max(axis=0)
        # Keep EVERY finite shell allowed by the explicit ceiling. No source,
        # descriptor, or primary query contribution is silently discarded.
        shells = self.max_shells-1
        images = tuple((k, odd) for k in range(-shells, shells+1)
                       for odd in (False, True) if source_only or (k, odd) != (0, False))
        with ieee_arithmetic():
            resolved = self.policy.mesh.resolve(source[2])
            tail = query_tail_bound(snapshot, lower, upper, shells=shells)
            classification = _classification(snapshot, lower, upper,
                                               cutoff=resolved.correction_cutoff, images=images)
            omitted_radius = classification.omitted_distance_lower
            correction = finite_image_correction_bound(snapshot, lower, upper, images=images,
                tau=resolved.tau, omitted_distance_lower=omitted_radius, max_images=len(images))
            velocity_error = float(add(point(tail.velocity_upper), point(correction.velocity_upper)).upper)
            gradient_error = float(add(point(tail.gradient_upper), point(correction.gradient_upper)).upper)
            # Division is not used to approve the gate: multiply the threshold
            # outward DOWN so a rounded quotient cannot admit an over-budget sum.
            velocity_limit = float(mul(point(self.tolerance), point(self.velocity_scale)).lower)
            gradient_limit = float(mul(point(self.tolerance), point(self.gradient_scale)).lower)
        if velocity_error > velocity_limit or gradient_error > gradient_limit:
            self._release_owner()
            raise RuntimeError("Gaussian slab truncation exceeds unchanged velocity/gradient budget: "
                               f"u={velocity_error}, J={gradient_error}, shells={shells}")
        certificate_finished = perf_counter()
        owner_hit = False
        owner_close_seconds = owner_build_seconds = owner_prepare_seconds = 0.0
        owner_miss_reason = None
        try:
            if self._owner is None:
                owner_miss_reason = "absent" if source_hit else "initial_or_changed_source"
            elif self._role != source_only:
                owner_miss_reason = "role_changed"
            elif not self._owner.can_evaluate_targets(query):
                owner_miss_reason = "outside_logical_stencil_domain"
            else:
                owner_hit = True
            if not owner_hit:
                phase_started = perf_counter()
                self._release_owner()
                owner_close_seconds = perf_counter()-phase_started
                try:
                    phase_started = perf_counter()
                    self._owner = _new_field_owner(*source, query, zmin=self.z_min, zmax=self.z_max,
                        execution_backend=self.execution_backend, portable=self.policy.backend == "auto",
                        tau=resolved.tau, spacing=resolved.spacing, cutoff=resolved.correction_cutoff,
                        order=resolved.order, dtype=self.dtype, correction_dtype=self.dtype,
                        source_only_primary=source_only, max_images=len(images),
                        max_query_points=self.policy.max_query_points,
                        max_scratch_bytes=self.policy.max_scratch_bytes,
                        max_correction_bytes=self.policy.max_correction_bytes,
                        max_plan_bytes=self.policy.max_plan_bytes, max_total_bytes=self.policy.max_total_bytes)
                    owner_build_seconds = perf_counter()-phase_started
                except _FieldConstructionError as outcome:
                    if outcome.cleanup_failure is not None:
                        self._failed_owner = outcome.owner
                        self._cleanup_uncertain = True
                        self.closed = True
                        outcome.failure.add_note(
                            f"Gaussian owner cleanup failed: {outcome.cleanup_failure!r}")
                    # Preserve the numerical cause, not a wrapper -> primary
                    # -> wrapper exception cycle or a secondary close error.
                    raise outcome.failure from outcome.failure.__cause__
                except BaseException:
                    # Construction may itself have failed during cleanup,
                    # before returning any handle. Do not retry allocations
                    # when that ownership outcome cannot be established.
                    self._cleanup_uncertain = True
                    self.closed = True
                    raise
                phase_started = perf_counter()
                self._owner.prepare(images)
                owner_prepare_seconds = perf_counter()-phase_started
                # Descriptive initial query bounds only, NOT cache admission.
                # The owner's complete cardinal-stencil domain is wider when
                # physical/reflected source support already fills the grid.
                self._role, self._bounds = source_only, (lower.copy(), upper.copy())
            # FTZ/DAZ may change subnormal host shifts even under RN. Match
            # the actual immutable list used by correction, not a hash or an
            # algebraically equivalent expression. No query is published on
            # a mismatch, including on an otherwise valid exact-source hit.
            if self._owner._prepared_world_images != classification.world_images:
                raise RuntimeError("runtime image descriptors differ from certified world images")
            velocity, gradient, finite = self._owner.evaluate_prepared(query)
            # Explicit transfers; no undocumented Taichi/CuPy pointer interop.
            velocity = self._owner.cp.asnumpy(velocity)
            gradient = self._owner.cp.asnumpy(gradient)
            if not np.isfinite(velocity).all() or not np.isfinite(gradient).all():
                raise FloatingPointError("nonfinite complete Gaussian slab query")
        except BaseException as failure:
            try:
                self._release_owner()
            except BaseException as cleanup_failure:
                # Keep the operation that failed as the caller-visible error.
                # _release_owner retains the failed resource and prevents any
                # subsequent allocation/reset when its drain is uncertain.
                failure.add_note(f"Gaussian owner cleanup also failed: {cleanup_failure!r}")
            raise
        return velocity, gradient, {
            "source_sha256": snapshot.source_sha256, "source_snapshot_hit": source_hit,
            "field_owner_hit": owner_hit, "source_only_primary": source_only,
            "field_owner_miss_reason": owner_miss_reason,
            "query_count": len(query), "query_lower": lower.tolist(), "query_upper": upper.tolist(),
            "source_snapshot_seconds": source_finished-started,
            "query_certificate_seconds": certificate_finished-source_finished,
            "owner_close_seconds": owner_close_seconds, "owner_build_seconds": owner_build_seconds,
            "owner_prepare_seconds": owner_prepare_seconds,
            "finite_images": len(images), "shell": shells,
            "velocity_tail_bound": tail.velocity_upper, "gradient_tail_bound": tail.gradient_upper,
            "velocity_correction_bound": correction.velocity_upper,
            "gradient_correction_bound": correction.gradient_upper,
            "omitted_distance_lower": omitted_radius, "finite_evaluation": finite,
            "seconds": perf_counter()-started,
            "certificate_scope": "mathematical truncation only; not finite-grid/GPU error",
        }

    def close(self):
        self._admit_thread()
        if self.cleanup_uncertain:
            raise RuntimeError("Gaussian slab GPU cleanup remains uncertain; do not reset its runtime")
        if not self.closed:
            self._release_owner()
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
