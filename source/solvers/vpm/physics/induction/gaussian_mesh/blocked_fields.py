"""Bounded CUDA blocks of the unchanged finite Gaussian image operator."""

import math
import threading
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import psutil

from .coordinates import finite_images, slab_coordinates
from .runtime import positive_integer


def _snapshot(values, name, *, vector=True):
    if hasattr(values, "__cuda_array_interface__") or np.iscomplexobj(values):
        raise ValueError(f"{name} requires real host values")
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if (array.ndim != (2 if vector else 1)
            or (vector and array.shape[1:] != (3,)) or not np.isfinite(array).all()):
        raise ValueError(f"{name} requires finite matching host values")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class GaussianBlockedCUDAFields:
    """Stream disjoint source/query products through one CUDA leaf at a time.

    Each leaf includes its own source-core correction. Source splits add
    fields; target splits write disjoint query rows. Cardinal weights retain
    the common origin of the entire request, including reflected sources.
    """

    execution_backend = "cupy_cuda"

    def __init__(self, source_x, source_gamma, source_sigma, initial_targets, *,
                 zmin, zmax, tau, spacing, cutoff, order=10, dtype="float32",
                 correction_dtype="float32", max_scratch_bytes=2 * 1024**3,
                 max_correction_bytes=256 * 1024**2, max_plan_bytes=128 * 1024**2,
                 max_total_bytes=2304 * 1024**2, max_images=513,
                 max_query_points=1_000_000, source_only_primary=False, profile=False):
        self.closed = False
        self._thread = threading.get_ident()
        self._leaf = self._failed_owner = None
        self._prepared_images = self._prepared_world_images = self._integer_images = None
        self._cached_query = self._cached_values = self._cached_report = None
        self.cp = SimpleNamespace(asnumpy=np.asarray)
        self.dtype = np.dtype(dtype)
        self.correction_dtype = np.dtype(correction_dtype)
        if (self.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or self.correction_dtype not in (np.dtype("float32"), np.dtype("float64"))):
            raise ValueError("float32/float64 field and correction precision required")
        if isinstance(order, bool) or order not in (4, 6, 8, 10):
            raise ValueError("even cardinal order4..10 required")
        if not all(math.isfinite(v) and v > 0 for v in (tau, spacing, cutoff)):
            raise ValueError("positive finite tau, spacing and cutoff required")
        if type(source_only_primary) is not bool:
            raise ValueError("source-only primary permission must be explicit boolean")
        self.order, self.zmin, self.zmax = int(order), float(zmin), float(zmax)
        self.tau, self.spacing, self.cutoff = float(tau), float(spacing), float(cutoff)
        self.source_only_primary, self.profile = source_only_primary, bool(profile)
        for name in ("max_scratch_bytes", "max_correction_bytes", "max_plan_bytes",
                     "max_total_bytes", "max_images", "max_query_points"):
            setattr(self, name, positive_integer(locals()[name], name))
        if self.max_scratch_bytes + self.max_correction_bytes > self.max_total_bytes:
            raise MemoryError("combined array-pool limits exceed total cap")
        if self.max_plan_bytes >= self.max_scratch_bytes:
            raise ValueError("plan reserve must be smaller than smooth pool")
        if self.max_query_points > 2**30:
            raise ValueError("query count exceeds index range")
        self.host_x = _snapshot(source_x, "source")
        self.host_gamma = _snapshot(source_gamma, "strength")
        self._host_sigma = _snapshot(source_sigma, "core", vector=False)
        self.host_targets = self._query(initial_targets)
        if (not 1 <= len(self.host_x) <= np.iinfo(np.int32).max
                or self.host_gamma.shape != self.host_x.shape
                or self._host_sigma.shape != (len(self.host_x),)
                or np.any(self._host_sigma <= 0) or np.any(self._host_sigma > self.tau)):
            raise ValueError("matching nonempty sources and 0<core<=tau required")
        if not len(self.host_targets):
            raise ValueError("nonempty initial target envelope required")
        if np.any(self.host_x[:, 2] < zmin) or np.any(self.host_x[:, 2] > zmax):
            raise ValueError("physical sources must lie within the slip slab")
        self._lattice_x, self.steps, self.cells = slab_coordinates(
            self.host_x, self.zmin, self.zmax, self.spacing
        )
        self._options = {"zmin": self.zmin, "zmax": self.zmax, "tau": self.tau,
                             "spacing": self.spacing, "cutoff": self.cutoff, "order": self.order,
                             "dtype": self.dtype.name, "correction_dtype": self.correction_dtype.name,
                             "source_only_primary": self.source_only_primary, "profile": self.profile}
        for name in ("max_scratch_bytes", "max_correction_bytes", "max_plan_bytes",
                     "max_total_bytes", "max_images", "max_query_points"):
            self._options[name] = getattr(self, name)
        self.snapshot_bytes = sum(a.nbytes for a in (
            self.host_x, self.host_gamma, self._host_sigma, self.host_targets, self._lattice_x
        ))
        self.initial_diagnostics = {"backend": "cupy_cuda", "snapshot_bytes": self.snapshot_bytes,
                                    "runtime_admissible": False, "tail_certified": False}

    def _admit(self):
        if self._failed_owner is not None:
            raise RuntimeError("Blocked Gaussian CUDA cleanup remains uncertain")
        if self.closed:
            raise RuntimeError("Blocked Gaussian CUDA owner is closed")
        if threading.get_ident() != self._thread:
            raise RuntimeError("Blocked Gaussian CUDA owner belongs to another thread")

    def _query(self, values):
        query = _snapshot(values, "target")
        if len(query) > self.max_query_points:
            raise ValueError("target count exceeds explicit capacity")
        return query

    def can_evaluate_targets(self, targets):
        self._admit()
        if self._prepared_images is None:
            raise RuntimeError("no successfully prepared finite image field")
        query = self._query(targets)
        slab_coordinates(query, self.zmin, self.zmax, self.spacing)
        return True

    def prepare(self, images):
        self._admit()
        self._prepared_images = self._prepared_world_images = self._integer_images = None
        self._cached_query = self._cached_values = self._cached_report = None
        saved, world, integer = finite_images(
            images, self.zmin, self.zmax, self.cells, self.max_images,
            include_primary=self.source_only_primary,
        )
        self._prepared_images, self._prepared_world_images, self._integer_images = saved, world, integer
        return {"backend": "cupy_cuda", "execution_plan": "blocked_linear_convolution",
                "finite_image_count": len(saved), "tail_certified": False}

    def _retire_leaf(self):
        leaf, self._leaf = self._leaf, None
        if leaf is None:
            return
        try:
            leaf.close()
        except BaseException as error:
            self._failed_owner = leaf
            raise RuntimeError("Blocked Gaussian CUDA leaf cleanup remains uncertain") from error

    @staticmethod
    def _split(source, target, source_points, target_points):
        sx, tq = source_points[source], target_points[target]
        se = np.ptp(sx, axis=0) if len(source) > 1 else np.full(3, -1.0)
        te = np.ptp(tq, axis=0) if len(target) > 1 else np.full(3, -1.0)
        split_source = len(source) > 1 and (
            len(target) == 1 or se.max() > te.max()
            or (se.max() == te.max() and len(source) >= len(target))
        )
        ids, points = (source, sx) if split_source else (target, tq)
        if len(ids) < 2:
            raise MemoryError("CUDA cannot hold one unchanged cardinal source/query block")
        ordered = ids[np.argsort(points[:, int(np.argmax(np.ptp(points, axis=0)))], kind="stable")]
        left, right = ordered[:len(ids) // 2], ordered[len(ids) // 2:]
        return ((left, target), (right, target)) if split_source else ((source, left), (source, right))

    def _evaluate_leaf(self, source, target, query, origin):
        from .fields import GaussianImageFields

        leaf = GaussianImageFields.__new__(GaussianImageFields)
        self._leaf = leaf
        try:
            leaf.__init__(self.host_x[source], self.host_gamma[source], self._host_sigma[source],
                          query[target], **self._options, _lattice_origin=origin)
            prepared = leaf.prepare(self._prepared_images)
            if leaf._prepared_world_images != self._prepared_world_images:
                raise RuntimeError("CUDA block image descriptors differ from the complete request")
            u, j, queried = leaf.evaluate_prepared(query[target])
            report = {**prepared, **queried}
            for key in ("inverse_transforms", "kernel_forward_transforms", "radial_kernel_launches"):
                report[key] = int(prepared.get(key, 0)) + int(queried.get(key, 0))
            correction = queried.get("correction", {})
            report["correction_pairs"] = int(correction.get("accepted_pairs", queried.get("correction_pairs", 0)))
            report["correction_candidates"] = int(correction.get("candidate_pairs", queried.get("correction_candidates", 0)))
            report["pool_reserved_bytes"] = max(int(prepared.get("pool_reserved_bytes", 0)),
                                                int(queried.get("smooth_pool_reserved_bytes", 0)))
            report["combined_pool_reserved_bytes"] = (report["pool_reserved_bytes"]
                                                      + int(queried.get("correction_pool_reserved_bytes", 0)))
            try:
                values = np.empty((len(target), 12), dtype=np.float64)
                values[:, :3] = leaf.cp.asnumpy(u)
                values[:, 3:] = leaf.cp.asnumpy(j).reshape(-1, 9)
            finally:
                del u, j
            if not np.isfinite(values).all():
                raise FloatingPointError("nonfinite complete Gaussian CUDA block")
            return values, report
        finally:
            self._retire_leaf()

    def evaluate_prepared(self, targets):
        self._admit()
        if self._prepared_images is None:
            raise RuntimeError("no successfully prepared finite image field")
        query = self._query(targets)
        started = perf_counter()
        if self._cached_query is not None and np.array_equal(
            query.view(np.uint64), self._cached_query.view(np.uint64)
        ):
            values = self._cached_values.copy()
            report = {**self._cached_report, "query_seconds": perf_counter() - started,
                      "exact_query_output_hit": True, "fft_blocks": 0}
            for key in ("block_splits", "memory_retries", "inverse_transforms", "kernel_forward_transforms",
                        "radial_kernel_launches", "correction_pairs", "correction_candidates",
                        "peak_pool_reserved_bytes", "peak_estimated_payload_bytes", "peak_plan_work_bytes",
                        "peak_combined_pool_reserved_bytes", "gather_seconds"):
                report[key] = 0
            report["passes"] = {}
            return values[:, :3], values[:, 3:].reshape(-1, 3, 3), report
        self._cached_query = self._cached_values = self._cached_report = None
        lattice_q, _, _ = slab_coordinates(query, self.zmin, self.zmax, self.spacing)
        reflected_min = self._lattice_x.min(axis=0).copy()
        reflected_min[2] = min(reflected_min[2], -self._lattice_x[:, 2].max())
        minimum = np.minimum(reflected_min, lattice_q.min(axis=0)) if len(query) else reflected_min
        origin = np.floor(minimum) - self.order
        output = np.zeros((len(query), 12), dtype=np.float64)
        report = {"backend": "cupy_cuda", "execution_plan": "blocked_linear_convolution",
                  "fft_blocks": 0, "block_splits": 0, "memory_retries": 0,
                  "peak_pool_reserved_bytes": 0, "peak_estimated_payload_bytes": 0,
                  "peak_plan_work_bytes": 0, "peak_combined_pool_reserved_bytes": 0,
                  "passes": {}, "gather_seconds": 0., "target_count": len(query),
                  "finite_image_count": len(self._prepared_images), "core_correction_included": True,
                  "tail_certified": False, "runtime_admissible": False,
                  "exact_query_output_hit": False, "snapshot_bytes": self.snapshot_bytes,
                  "query_output_bytes": len(query) * 12 * self.dtype.itemsize}
        stack = [(np.arange(len(self.host_x)), np.arange(len(query)))] if len(query) else []
        try:
            while stack:
                source, target = stack.pop()
                try:
                    values, leaf_report = self._evaluate_leaf(source, target, query, origin)
                except MemoryError:
                    children = self._split(source, target, self._lattice_x, lattice_q)
                    stack.extend(reversed(children))
                    report["block_splits"] += 1
                    report["memory_retries"] += 1
                    continue
                output[target] += values
                del values
                report["fft_blocks"] += 1
                for label, key in (("peak_pool_reserved_bytes", "pool_reserved_bytes"),
                                   ("peak_estimated_payload_bytes", "estimated_payload_bytes"),
                                   ("peak_plan_work_bytes", "plan_peak_work_bytes"),
                                   ("peak_combined_pool_reserved_bytes", "combined_pool_reserved_bytes")):
                    report[label] = max(report[label], int(leaf_report.get(key, 0)))
                for key in ("inverse_transforms", "kernel_forward_transforms", "radial_kernel_launches",
                            "correction_pairs", "correction_candidates"):
                    report[key] = report.get(key, 0) + int(leaf_report.get(key, 0))
                for name, seconds in leaf_report.get("passes", {}).items():
                    report["passes"][name] = report["passes"].get(name, 0.) + float(seconds)
                report["gather_seconds"] += float(leaf_report.get("gather_seconds", 0.))
            values = output.astype(self.dtype, copy=False)
            if not np.isfinite(values).all():
                raise FloatingPointError("nonfinite complete blocked Gaussian CUDA field")
        except BaseException:
            self._prepared_images = self._prepared_world_images = self._integer_images = None
            raise
        report["query_seconds"] = perf_counter() - started
        cache_bytes = query.nbytes + values.nbytes
        if cache_bytes <= self.max_total_bytes and psutil.virtual_memory().available >= 2 * cache_bytes:
            try:
                cached = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
            except MemoryError:
                pass
            else:
                self._cached_query, self._cached_values, self._cached_report = query, cached, dict(report)
        return values[:, :3], values[:, 3:].reshape(-1, 3, 3), report

    def evaluate(self, images):
        self.prepare(images)
        return self.evaluate_prepared(self.host_targets)

    def close(self):
        if self._failed_owner is not None:
            raise RuntimeError("Blocked Gaussian CUDA cleanup remains uncertain")
        if self.closed:
            return
        self._admit()
        self._retire_leaf()
        self._prepared_images = self._prepared_world_images = self._integer_images = None
        self._cached_query = self._cached_values = self._cached_report = None
        self.closed = True

    def __enter__(self):
        self._admit()
        return self

    def __exit__(self, *_):
        self.close()
