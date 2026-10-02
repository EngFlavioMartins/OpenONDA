"""UNWIRED same-source, finite-image compact-field reuse qualification.

The existing fused operator is unchanged. During preparation this subclass
copies the logical (unpadded) part of each inverse transform before its usual
gather. A later query ONLY constructs stencils, gathers those twelve fields,
and evaluates the same immutable source-only Gaussian core correction. It
does not construct new images, extrapolate, infer source identity, or approve
an image tail. Every new target stencil must fit the original logical grid.

One owner snapshots its input sources; it accepts no replacement source fields.
The public source/query arrays of the inherited qualification object are not
an application mutation API. Returned fields are separate allocations, never
views of the retained compact grids. Array pools and FFT plans remain bounded
by the existing isolated qualification ownership protocol, not a proposed
production/global FFT-cache policy.
"""

from itertools import islice
import math
import time

import numpy as np

from tests.vpm._cupy_cardinal_stencil import cardinal_stencil_gpu
from tests.vpm._cupy_slab_field_mesh_fused import SlabFieldMeshFusedGPU
from tests.vpm._finite_image_mesh_reference import _positive_cap
from tests.vpm._finite_slab_field_mesh_reference import slab_coordinates, slab_world_images
from tests.vpm._gaussian_core_correction_gpu import GaussianCoreCorrectionGPU


class CompactSlabFieldMeshGPU(SlabFieldMeshFusedGPU):
    """Private immutable-source owner; finite image fields, not a tail policy."""

    def __init__(self, source_x, source_gamma, source_sigma, targets, *, cutoff=.6,
                 correction_dtype="float32", max_correction_bytes=256*1024**2,
                 max_total_bytes=2304*1024**2, max_query_points=1_000_000, **kwargs):
        self._compact_fields = None
        self._capture_columns = None
        self._prepared_images = None
        self._prepared_world_images = None
        self._correction = None
        self._prepare_diagnostics = None
        if any(np.iscomplexobj(array) for array in (source_x, source_gamma, source_sigma, targets)):
            raise ValueError("real source, core and target arrays required")
        self.max_query_points = _positive_cap(max_query_points, "max_query_points")
        self.max_total_bytes = _positive_cap(max_total_bytes, "max_total_bytes")
        correction_cap = _positive_cap(max_correction_bytes, "max_correction_bytes")
        mesh_cap = _positive_cap(kwargs.get("max_scratch_bytes", 2*1024**3), "max_scratch_bytes")
        if mesh_cap+correction_cap > self.max_total_bytes:
            raise MemoryError("combined smooth/correction pool limits exceed total cap")
        self._input_sigma = np.array(source_sigma, dtype=np.float64, copy=True, order="C")
        super().__init__(source_x, source_gamma, targets, **kwargs)
        try:
            if len(self.host_targets) > self.max_query_points:
                raise ValueError("initial targets exceed query cap")
            if np.any(self.host_x[:, 2] < self.zmin) or np.any(self.host_x[:, 2] > self.zmax):
                raise ValueError("physical sources must lie inside the slip slab")
            self.compact_bytes = 12*math.prod(self.shape)*self.dtype.itemsize
            self.estimated_payload_bytes += self.compact_bytes
            if self.estimated_payload_bytes > self.max_scratch_bytes-self.max_plan_bytes:
                raise MemoryError("compact grids exceed smooth pool payload admission")
            self._input_sigma.setflags(write=False)
            self._correction = GaussianCoreCorrectionGPU(
                self.host_x, self.host_gamma, self._input_sigma,
                tau=self.tau, cutoff=cutoff, accumulation_dtype=correction_dtype,
                max_scratch_bytes=correction_cap, max_images=self.max_images)
        except BaseException:
            self.close()
            raise

    def _launch(self, name, count, arguments):
        if name == "gather" and self._capture_columns is not None:
            column = int(arguments[7])
            if column in self._capture_columns or not 0 <= column < 12:
                raise RuntimeError("duplicate or invalid compact capture column")
            field = arguments[6]
            crop = tuple(slice(0, size) for size in self.shape)
            self.cp.copyto(self._compact_fields[column], field[crop])
            self._capture_columns.add(column)
        super()._launch(name, count, arguments)

    def prepare(self, images):
        """Replace the finite-block cache; any failure revokes its eligibility."""
        self._admit()
        self._prepared_images = self._prepared_world_images = None
        self._prepare_diagnostics = None
        # Snapshot descriptors before any device work; lists supplied by the
        # caller cannot change the operator after it has been prepared.
        descriptors = tuple(islice(iter(images), self.max_images+1))
        if len(descriptors) > self.max_images:
            raise ValueError("finite image count exceeds explicit qualification cap")
        world, _ = slab_world_images(descriptors, self.zmin, self.zmax,
                                     self.cells, self.max_images)
        descriptors = tuple((int(k), bool(odd)) for k, odd in descriptors)
        started = time.perf_counter()
        with self.cp.cuda.using_allocator(self.pool.malloc):
            if self._compact_fields is None:
                self._compact_fields = self.cp.empty((12, *self.shape), dtype=self.dtype)
            self._capture_columns = set()
            try:
                velocity, gradient, diagnostics = super().evaluate(descriptors)
                del velocity, gradient
                self.stream.synchronize()
                if self._capture_columns != set(range(12)):
                    raise RuntimeError("incomplete compact-grid capture")
                if not bool(self.cp.isfinite(self._compact_fields).all()):
                    raise FloatingPointError("nonfinite compact field; cache not published")
                self._prepared_images = descriptors
                self._prepared_world_images = tuple(world)
                self._prepare_diagnostics = {
                    "runtime_admissible": False, "tail_certified": False,
                    "finite_image_count": len(descriptors), "compact_bytes": self.compact_bytes,
                    "logical_shape": self.shape, "fft_shape": self.fft_shape,
                    "preparation_seconds": time.perf_counter()-started,
                    "smooth": diagnostics,
                }
                return dict(self._prepare_diagnostics)
            finally:
                self._capture_columns = None

    def evaluate(self, images):
        """Return complete finite image fields, INCLUDING local core correction.

        Unlike the inherited smooth-only prototype, callers must NOT add a
        second Gaussian core correction to these outputs.
        """
        self.prepare(images)
        return self.evaluate_prepared(self.host_targets)

    def evaluate_prepared(self, targets):
        """Fresh target stencils + compact gather + local core correction only."""
        self._admit()
        if self._prepared_images is None:
            raise RuntimeError("no successfully prepared finite image field")
        if isinstance(targets, self.cp.ndarray):
            raise TypeError("explicit host target snapshot required for this qualification")
        if np.iscomplexobj(targets):
            raise ValueError("real target coordinates required")
        q = np.array(targets, dtype=np.float64, copy=True, order="C")
        if q.ndim != 2 or q.shape[1:] != (3,) or len(q) > self.max_query_points:
            raise ValueError("bounded targets (N,3) required")
        lattice, _, _ = slab_coordinates(q, self.zmin, self.zmax, float(self.steps[0]))
        started = time.perf_counter()
        with self.cp.cuda.using_allocator(self.pool.malloc):
            first, weight, stencil = cardinal_stencil_gpu(
                lattice, self.origin, self.order, self.shape, dtype=self.dtype,
                pool=self.pool, max_points=self.max_query_points,
                max_new_bytes=self.max_scratch_bytes)
            output = self.cp.empty((len(q), 12), dtype=self.dtype)
            gather_started = time.perf_counter()
            try:
                if len(q):
                    for column in range(12):
                        self._launch("gather", len(q), (np.int32(len(q)), np.int32(self.order),
                            first, weight, np.int32(self.shape[1]), np.int32(self.shape[2]),
                            self._compact_fields[column], np.int32(column), output))
                self.stream.synchronize()
                gather_seconds = time.perf_counter()-gather_started
                correction_u, correction_j, correction = self._correction.evaluate(
                    q, self._prepared_world_images)
                output[:, :3] += correction_u
                output[:, 3:] += correction_j.reshape(-1, 9)
                del correction_u, correction_j
                self.stream.synchronize()
                if not bool(self.cp.isfinite(output).all()):
                    raise FloatingPointError("nonfinite compact query; nothing published")
            except BaseException:
                # CUDA/evaluation failures cannot leave a seemingly certified
                # reusable result. Bad stencil admission above performs no
                # cache writes and may be retried with a valid query.
                self._prepared_images = self._prepared_world_images = None
                raise
            return output[:, :3], output[:, 3:].reshape(-1, 3, 3), {
                "runtime_admissible": False, "tail_certified": False,
                "source_replaced": False, "target_count": len(q),
                "finite_image_count": len(self._prepared_images),
                "compact_bytes": self.compact_bytes, "stencil": stencil,
                "gather_seconds": gather_seconds, "correction": correction,
                "query_seconds": time.perf_counter()-started,
                "inverse_transforms": 0, "source_scatters": 0,
                "smooth_pool_reserved_bytes": self.pool.total_bytes(),
                "correction_pool_reserved_bytes": self._correction.pool.total_bytes(),
                "combined_pool_cap": self.max_total_bytes,
            }

    def close(self):
        if getattr(self, "closed", True):
            return
        self._admit()
        self.stream.synchronize()
        self._prepared_images = self._prepared_world_images = None
        self._capture_columns = None
        self._compact_fields = None
        error = None
        try:
            if self._correction is not None:
                self._correction.close()
        except BaseException as failure:
            error = failure
        self._correction = None
        try:
            super().close()
        finally:
            if error is not None:
                raise error
