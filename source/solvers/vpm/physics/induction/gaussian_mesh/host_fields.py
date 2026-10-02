"""Portable, bounded FFT evaluation of the finite Gaussian image operator.

Each source/target block has its own integer lattice origin. The convolution
kernel carries their origin difference, so a remote query never allocates the
empty space between source and query. Splitting blocks changes accumulation
order only: cardinal weights, spacing, image descriptors and radial kernels
are unchanged. This owner does not admit either the infinite image tail or
the omitted local correction; the enclosing session performs those gates.
"""

import math
import threading
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import psutil
from scipy.fft import irfftn, next_fast_len, rfftn
from scipy.spatial import cKDTree
from scipy.special import erf

from .coordinates import finite_images, slab_coordinates
from .correction import _possibly_near, correction_factors_host
from .runtime import positive_integer

_COMPONENTS = ((0, -1), (1, -1), (2, -1), (0, 0), (0, 1),
               (0, 2), (1, 1), (1, 2), (2, 2))


def _routes(column):
    row = column if column < 3 else (column-3)//3
    derivative = -1 if column < 3 else (column-3)%3
    a, b = ((1, 2), (2, 0), (0, 1))[row]
    return ((a, derivative, b, 1), (b, derivative, a, -1))


def _snapshot(value, name, *, vector=True):
    if hasattr(value, "__cuda_array_interface__") or np.iscomplexobj(value):
        raise ValueError(f"{name} requires real host values")
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if (array.ndim != (2 if vector else 1)
            or (vector and array.shape[1:] != (3,)) or not np.isfinite(array).all()):
        raise ValueError(f"{name} must contain finite matching host values")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _stencil(points, order, dtype):
    """Global integer first nodes and the same ordered cardinal products."""
    if not np.isfinite(points).all() or np.any(np.abs(points) > 2**52-order):
        raise ValueError("normalized coordinates exceed exact lattice range")
    first = np.floor(points).astype(np.int64)-order//2+1
    argument = points-first
    weights = np.empty((len(points), 3, order), dtype=dtype)
    for i in range(order):
        value = np.ones((len(points), 3), dtype=np.float64)
        for j in range(order):
            if i != j:
                value *= (argument-j)/(i-j)
        weights[:, :, i] = value
    if not np.isfinite(weights).all():
        raise FloatingPointError("nonfinite cardinal weights")
    return first, weights


def _geometry(points, order):
    if not np.isfinite(points).all() or np.any(np.abs(points) > 2**52-order):
        raise ValueError("normalized coordinates exceed exact lattice range")
    first = np.floor(points).astype(np.int64)-order//2+1
    origin = first.min(axis=0)
    shape = tuple((first.max(axis=0)-origin+order).tolist())
    return origin, shape


def _payload(shape, source_count, target_count, order, itemsize):
    fft_shape = tuple(next_fast_len(2*n-1) for n in shape)
    volume = math.prod(fft_shape)
    spectrum = math.prod((*fft_shape[:2], fft_shape[2]//2+1))
    # Three density and nine kernel spectra, one result and one product.
    # Include twelve real grids for fused radial channels and FFT workspace.
    # Radial arithmetic is streamed by x planes rather than allocating Vx3
    # coordinates or Vx3x3 derivatives. PocketFFT has no retained plan pool.
    grid_bytes = itemsize*(28*spectrum+12*volume)
    plane_bytes = 8*32*fft_shape[1]*fft_shape[2]
    # First nodes, coordinates, geometry-generator copies, ordered cardinal
    # products and scatter/gather index/value temporaries are all included.
    stencil_bytes = (source_count+target_count)*(24*order+192)
    query_bytes = target_count*12*8
    return fft_shape, grid_bytes+plane_bytes+stencil_bytes+query_bytes


def _scatter(first, weights, strength, shape, fft_shape, dtype, component):
    grid = np.zeros(fft_shape, dtype=dtype)
    order = weights.shape[-1]
    values = strength[:, component].astype(dtype)
    for a in range(order):
        for b in range(order):
            for c in range(order):
                factor = weights[:, 0, a]*weights[:, 1, b]*weights[:, 2, c]
                np.add.at(grid, (first[:, 0]+a, first[:, 1]+b, first[:, 2]+c), factor*values)
    return grid


def _gather(field, first, weights):
    values = np.zeros(len(first), dtype=np.float64)
    order = weights.shape[-1]
    for a in range(order):
        for b in range(order):
            for c in range(order):
                factor = (weights[:, 0, a].astype(np.float64)
                          * weights[:, 1, b].astype(np.float64)
                          * weights[:, 2, c].astype(np.float64))
                values += factor*field[first[:, 0]+a, first[:, 1]+b, first[:, 2]+c]
    return values


def _kernels(shape, fft_shape, offset, steps, shifts, tau, dtype):
    """Nine fused radial channels, with plane temporaries and exact origins."""
    if (any(abs(int(delta))+n-1 > 2**53 for delta, n in zip(offset[:2], shape[:2], strict=True))
            or any(abs(int(offset[2])-int(shift))+shape[2]-1 > 2**53 for shift in shifts)):
        raise ValueError("image lags exceed exact kernel-coordinate range")
    axes, valid = [], []
    for size, padded in zip(shape, fft_shape, strict=True):
        index = np.arange(padded)
        axes.append(np.where(index < size, index, index-padded))
        valid.append((index < size) | (index >= padded-size+1))
    result = np.zeros((9, *fft_shape), dtype=dtype)
    pi15 = math.pi**-1.5
    yz_mask = valid[1][:, None] & valid[2][None, :]
    for ix, lag_x in enumerate(axes[0]):
        if not valid[0][ix]:
            continue
        rx = float(lag_x+int(offset[0]))*steps[0]
        ry = (axes[1]+int(offset[1]))[:, None]*steps[1]
        plane = np.zeros((9, *fft_shape[1:]), dtype=np.float64)
        for shift in shifts:
            rz = (axes[2]+int(offset[2])-int(shift))[None, :]*steps[2]
            r2 = rx*rx+ry*ry+rz*rz
            rho2 = r2/(tau*tau)
            near = rho2 < 1.
            a, b = np.zeros_like(r2), np.zeros_like(r2)
            term = np.ones(np.count_nonzero(near))
            sa, sb = np.zeros_like(term), np.zeros_like(term)
            for n in range(24):
                sa += term/(2*n+3)
                sb += 2*term/(2*n+5)
                term *= -rho2[near]/(n+1)
            a[near], b[near] = pi15*sa/tau**3, pi15*sb/tau**5
            far = ~near
            radius = np.sqrt(r2[far])
            rho = radius/tau
            e = np.exp(-rho*rho)
            q = (erf(rho)-2*rho*e/math.sqrt(math.pi))/(4*math.pi)
            a[far] = q/(r2[far]*radius)
            b[far] = 3*q/(r2[far]*r2[far]*radius)-pi15*e/(tau**3*r2[far])
            coordinates = (rx, ry, rz)
            for channel, (axis, derivative) in enumerate(_COMPONENTS):
                value = (-a*coordinates[axis] if derivative < 0 else
                         b*coordinates[axis]*coordinates[derivative]-(a if axis == derivative else 0.))
                plane[channel] += np.where(yz_mask, value, 0.)
        result[:, ix] = plane
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite Gaussian radial grid")
    return result


class GaussianImageFieldsHost:
    """Immutable finite image owner using source/target blocked convolution.

    The scratch cap bounds one FFT job including conservative FFT workspace
    and plane arithmetic. Returned query storage and immutable source identity
    are reported separately, just as host snapshots are outside CUDA pools.
    No allocation proportional to a source/query separation is performed.
    """

    def __init__(self, source_x, source_gamma, source_sigma, initial_targets, *,
                 zmin, zmax, tau, spacing, cutoff, order=10, dtype="float32",
                 correction_dtype="float32", max_scratch_bytes=2*1024**3,
                 max_correction_bytes=256*1024**2, max_plan_bytes=128*1024**2,
                 max_total_bytes=2304*1024**2, max_images=513,
                 max_query_points=1_000_000, source_only_primary=False, profile=False):
        self.closed = False
        self._thread = threading.get_ident()
        self._prepared_images = self._prepared_world_images = self._integer_images = None
        self.cp = SimpleNamespace(asnumpy=np.asarray)
        self.dtype = np.dtype(dtype)
        self.correction_dtype = np.dtype(correction_dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")) or self.correction_dtype not in (
                np.dtype("float32"), np.dtype("float64")):
            raise ValueError("float32/float64 field and correction precision required")
        if isinstance(order, bool) or order not in (4, 6, 8, 10):
            raise ValueError("even cardinal order4..10 required")
        self.order = int(order)
        if not all(math.isfinite(v) and v > 0 for v in (tau, spacing, cutoff)):
            raise ValueError("positive finite tau, spacing and cutoff required")
        self.zmin, self.zmax, self.tau = float(zmin), float(zmax), float(tau)
        self.spacing, self.cutoff = float(spacing), float(cutoff)
        for name in ("max_scratch_bytes", "max_correction_bytes", "max_plan_bytes",
                     "max_total_bytes", "max_images", "max_query_points"):
            setattr(self, name, positive_integer(locals()[name], name))
        if self.max_scratch_bytes+self.max_correction_bytes > self.max_total_bytes:
            raise MemoryError("combined scratch limits exceed total cap")
        if self.max_plan_bytes >= self.max_scratch_bytes:
            raise ValueError("plan reserve must be smaller than smooth-field cap")
        if type(source_only_primary) is not bool:
            raise ValueError("source-only primary permission must be explicit boolean")
        self.source_only_primary, self.profile = source_only_primary, bool(profile)
        self.host_x, self.host_gamma = _snapshot(source_x, "source"), _snapshot(source_gamma, "strength")
        self._host_sigma = _snapshot(source_sigma, "core", vector=False)
        self.host_targets = _snapshot(initial_targets, "target")
        if (not len(self.host_x) or self.host_gamma.shape != self.host_x.shape
                or self._host_sigma.shape != (len(self.host_x),)
                or np.any(self._host_sigma <= 0) or np.any(self._host_sigma > self.tau)):
            raise ValueError("matching nonempty sources and 0<core<=tau required")
        if not 1 <= len(self.host_targets) <= self.max_query_points:
            raise ValueError("nonempty bounded initial targets required")
        if np.any(self.host_x[:, 2] < zmin) or np.any(self.host_x[:, 2] > zmax):
            raise ValueError("physical sources must lie within the slip slab")
        self._lattice_x, self.steps, self.cells = slab_coordinates(self.host_x, zmin, zmax, spacing)
        _, minimum = _payload((order,)*3, 1, 1, order, self.dtype.itemsize)
        self._minimum_payload = minimum
        self._effective_scratch_bytes = self.max_scratch_bytes-self.max_plan_bytes
        self._effective_correction_bytes = self.max_correction_bytes
        if minimum > self.max_scratch_bytes-self.max_plan_bytes:
            raise MemoryError("scratch cap cannot hold one unchanged cardinal FFT block")
        if self.max_correction_bytes < 1024:
            raise MemoryError("correction cap cannot hold one source neighborhood")
        self.snapshot_bytes = sum(a.nbytes for a in (
            self.host_x, self.host_gamma, self._host_sigma, self.host_targets, self._lattice_x))
        self.initial_diagnostics = {"backend": "scipy_cpu", "snapshot_bytes": self.snapshot_bytes,
                                    "runtime_admissible": False, "tail_certified": False}

    def _admit(self):
        if self.closed:
            raise RuntimeError("finite Gaussian host owner is closed")
        if threading.get_ident() != self._thread:
            raise RuntimeError("Gaussian host owner belongs to another thread")

    def _query(self, targets):
        query = _snapshot(targets, "target")
        if len(query) > self.max_query_points:
            raise ValueError("target count exceeds explicit capacity")
        return query

    def can_evaluate_targets(self, targets):
        self._admit()
        if self._prepared_images is None:
            raise RuntimeError("no successfully prepared finite image field")
        query = self._query(targets)
        lattice, _, _ = slab_coordinates(query, self.zmin, self.zmax, self.spacing)
        return bool(not len(lattice) or np.all(np.abs(lattice) <= 2**52-self.order))

    def prepare(self, images):
        self._admit()
        self._prepared_images, self._prepared_world_images, self._integer_images = finite_images(
            images, self.zmin, self.zmax, self.cells, self.max_images,
            include_primary=self.source_only_primary)
        return {"backend": "scipy_cpu", "execution_plan": "blocked_linear_convolution",
                "finite_image_count": len(self._prepared_images), "tail_certified": False}

    def _blocks(self, source_points, query_points, source_ids, target_ids):
        """Recursively partition point sets; every point pair remains present."""
        stack = [(source_ids, target_ids)]
        cap = self._effective_scratch_bytes
        while stack:
            source, target = stack.pop()
            sx, tq = source_points[source], query_points[target]
            so, ss = _geometry(sx, self.order)
            qo, qs = _geometry(tq, self.order)
            shape = tuple(max(a, b) for a, b in zip(ss, qs, strict=True))
            fft_shape, payload = _payload(shape, len(source), len(target), self.order, self.dtype.itemsize)
            if payload <= cap:
                yield source, target, so, qo, shape, fft_shape, payload
                continue
            source_spread = np.ptp(sx, axis=0) if len(source) > 1 else np.full(3, -1.)
            target_spread = np.ptp(tq, axis=0) if len(target) > 1 else np.full(3, -1.)
            # A count-only split also bounds coincident-cloud stencil memory.
            split_source = (len(source) > 1 and (len(target) == 1
                            or source_spread.max() > target_spread.max()
                            or (source_spread.max() == target_spread.max() and len(source) >= len(target))))
            ids, points = (source, sx) if split_source else (target, tq)
            if len(ids) < 2:
                raise MemoryError("scratch cap cannot hold one unchanged cardinal FFT block")
            axis = int(np.argmax(np.ptp(points, axis=0)))
            sorted_ids = ids[np.argsort(points[:, axis], kind="stable")]
            middle = len(ids)//2
            if split_source:
                stack.extend(((sorted_ids[middle:], target), (sorted_ids[:middle], target)))
            else:
                stack.extend(((source, sorted_ids[middle:]), (source, sorted_ids[:middle])))

    def _smooth(self, query, report):
        lattice_q, _, _ = slab_coordinates(query, self.zmin, self.zmax, self.spacing)
        output = np.zeros((len(query), 12), dtype=self.dtype)
        source_ids, target_ids = np.arange(len(self.host_x)), np.arange(len(query))
        for odd in (False, True):
            shifts = [shift for shift, parity in self._integer_images if parity == odd]
            if not shifts or not len(query):
                continue
            points = self._lattice_x.copy() if odd else self._lattice_x
            if odd:
                points[:, 2] *= -1
            for ids, targets, so, qo, shape, fft_shape, payload in self._blocks(
                    points, lattice_q, source_ids, target_ids):
                sf, sw = _stencil(points[ids], self.order, self.dtype)
                tf, tw = _stencil(lattice_q[targets], self.order, self.dtype)
                sf -= so
                tf -= qo
                strengths = self.host_gamma[ids].copy()
                if odd:
                    strengths[:, :2] *= -1
                density_hat = []
                for component in range(3):
                    density = _scatter(sf, sw, strengths, shape, fft_shape, self.dtype, component)
                    density_hat.append(rfftn(density, workers=1, overwrite_x=True))
                    del density
                spectrum_shape = density_hat[0].shape
                kernels = _kernels(shape, fft_shape, qo-so, self.steps, shifts, self.tau, self.dtype)
                kernel_hats = [rfftn(kernels[channel], workers=1) for channel in range(9)]
                del kernels
                for column in range(12):
                    result_hat = np.zeros(spectrum_shape, dtype=density_hat[0].dtype)
                    for axis, derivative, component, sign in _routes(column):
                        pair = (axis, derivative) if derivative < 0 else tuple(sorted((axis, derivative)))
                        channel = _COMPONENTS.index(pair)
                        kernel_hat = np.multiply(kernel_hats[channel], density_hat[component])
                        if sign < 0:
                            np.negative(kernel_hat, out=kernel_hat)
                        np.add(result_hat, kernel_hat, out=result_hat)
                        del kernel_hat
                    field = irfftn(result_hat, s=fft_shape, workers=1, overwrite_x=True)
                    values = _gather(field, tf, tw)
                    output[targets, column] += values.astype(self.dtype)
                    del field, result_hat, values
                report["fft_blocks"] += 1
                report["peak_estimated_scratch_bytes"] = max(report["peak_estimated_scratch_bytes"], payload)
                report["inverse_transforms"] += 12
                report["kernel_forward_transforms"] += 9
        return output

    def _correction(self, query, report):
        output = np.zeros((len(query), 12), dtype=self.correction_dtype)
        # One tree and one candidate list are live. The conservative per-source
        # reserve includes tree nodes, copied coordinates and candidate indices.
        if not len(query):
            return output
        block_size = max(1, min(len(self.host_x), self._effective_correction_bytes//512))
        query_block_size = max(1, min(1024, self._effective_correction_bytes//512))
        target_min, target_max = query.min(axis=0), query.max(axis=0)
        for start in range(0, len(self.host_x), block_size):
            end = min(start+block_size, len(self.host_x))
            source = self.host_x[start:end]
            tree = cKDTree(source, copy_data=False)
            source_min, source_max = source.min(axis=0), source.max(axis=0)
            for shift, odd in self._prepared_world_images:
                if not _possibly_near(source_min, source_max, target_min, target_max,
                                      shift, odd, self.cutoff):
                    continue
                scale = max(1., abs(shift), float(np.max(np.abs(source))),
                            float(np.max(np.abs(query))) if len(query) else 1.)
                margin = 64*np.finfo(float).eps*scale
                if not math.isfinite(margin) or margin > self.cutoff/8:
                    raise ValueError("image correction coordinate resolution is insufficient")
                search_radius = np.nextafter(self.cutoff+margin, np.inf)
                for qstart in range(0, len(query), query_block_size):
                    qend = min(qstart+query_block_size, len(query))
                    transformed = query[qstart:qend].copy()
                    transformed[:, 2] = shift-transformed[:, 2] if odd else transformed[:, 2]-shift
                    counts = tree.query_ball_point(transformed, search_radius, return_length=True)
                    # Bound the aggregate Python neighbor-list storage before
                    # materializing it, even for a dense coincident cloud.
                    first = 0
                    while first < len(transformed):
                        last, total = first, 0
                        while last < len(transformed) and total+counts[last] <= block_size:
                            total += int(counts[last])
                            last += 1
                        if last == first:
                            last += 1
                        lists = tree.query_ball_point(transformed[first:last], search_radius)
                        for lane, candidates in enumerate(lists, start=first):
                            target = qstart+lane
                            report["correction_candidates"] += len(candidates)
                            for local in candidates:
                                index = start+local
                                image = self.host_x[index].copy()
                                image[2] = shift-image[2] if odd else image[2]+shift
                                displacement = query[target]-image
                                radius = float(np.linalg.norm(displacement))
                                if radius >= self.cutoff:
                                    continue
                                vector = self.host_gamma[index].copy()
                                if odd:
                                    vector[:2] *= -1
                                a, b = correction_factors_host(radius, self._host_sigma[index], self.tau)
                                cross = np.array([[0., -vector[2], vector[1]], [vector[2], 0., -vector[0]],
                                                  [-vector[1], vector[0], 0.]])
                                output[target, :3] += (a*(cross@displacement)).astype(self.correction_dtype)
                                output[target, 3:] += (cross@(a*np.eye(3)-b*np.outer(displacement, displacement))).reshape(9).astype(self.correction_dtype)
                                report["correction_pairs"] += 1
                        first = last
            del tree
        return output

    def evaluate_prepared(self, targets):
        self._admit()
        if self._prepared_images is None:
            raise RuntimeError("no successfully prepared finite image field")
        query = self._query(targets)
        started = perf_counter()
        available_bytes = int(psutil.virtual_memory().available)
        # Outside the per-job FFT cap: full reflected source coordinates,
        # global IDs/query lattice, final field and correction field outputs.
        resident_bytes = (len(self.host_x)*32+len(query)*(
            32+12*self.dtype.itemsize+12*self.correction_dtype.itemsize))
        live_budget = max(0, available_bytes//2-resident_bytes)
        self._effective_scratch_bytes = min(self.max_scratch_bytes-self.max_plan_bytes, live_budget)
        self._effective_correction_bytes = min(self.max_correction_bytes, live_budget//2)
        if self._effective_scratch_bytes < self._minimum_payload or self._effective_correction_bytes < 1024:
            raise MemoryError("available host memory cannot hold one unchanged cardinal FFT block and outputs")
        report = {"backend": "scipy_cpu", "execution_plan": "blocked_linear_convolution",
                  "fft_blocks": 0, "peak_estimated_scratch_bytes": 0, "inverse_transforms": 0,
                  "kernel_forward_transforms": 0, "correction_candidates": 0, "correction_pairs": 0,
                  "core_correction_included": True, "tail_certified": False,
                  "runtime_admissible": False, "snapshot_bytes": self.snapshot_bytes,
                  "query_output_bytes": len(query)*12*self.dtype.itemsize,
                  "host_available_bytes": available_bytes, "host_resident_transient_bytes": resident_bytes,
                  "effective_scratch_bytes": self._effective_scratch_bytes,
                  "effective_correction_bytes": self._effective_correction_bytes, "host_memory_retries": 0,
                  "combined_pool_cap": self.max_total_bytes, "target_count": len(query),
                  "finite_image_count": len(self._prepared_images)}
        try:
            while True:
                try:
                    output = self._smooth(query, report)
                    break
                except MemoryError:
                    if self._effective_scratch_bytes <= self._minimum_payload:
                        raise
                    next_cap = max(self._minimum_payload, self._effective_scratch_bytes//2)
                # Leave the exception scope before retrying so its traceback
                # releases every FFT buffer and the discarded partial output.
                # A fresh attempt accumulates each source contribution once.
                self._effective_scratch_bytes = next_cap
                report["effective_scratch_bytes"] = next_cap
                report["host_memory_retries"] += 1
            output += self._correction(query, report).astype(self.dtype)
            if not np.isfinite(output).all():
                raise FloatingPointError("nonfinite complete Gaussian host field")
        except BaseException:
            self._prepared_images = self._prepared_world_images = self._integer_images = None
            raise
        report["query_seconds"] = perf_counter()-started
        return output[:, :3], output[:, 3:].reshape(-1, 3, 3), report

    def evaluate(self, images):
        self.prepare(images)
        return self.evaluate_prepared(self.host_targets)

    def close(self):
        if self.closed:
            return
        self._admit()
        self._prepared_images = self._prepared_world_images = self._integer_images = None
        self.closed = True

    def __enter__(self):
        self._admit()
        return self

    def __exit__(self, *_):
        self.close()


# Public owner spelling shared with the backend execution factory.
GaussianHostImageFields = GaussianImageFieldsHost
