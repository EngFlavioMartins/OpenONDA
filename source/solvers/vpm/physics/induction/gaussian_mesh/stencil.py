"""Bounded cardinal stencils for already-normalized auxiliary coordinates.

Same ordered f64 product and final dtype cast as the qualified reference.
No snapping, wrapping or partition-of-unity repair. The caller supplies the
private capped pool; allocator selection is scoped and always restored.
"""

import math
from numbers import Integral
import time

import numpy as np


def _parameters(origin, order, shape, dtype, max_points):
    if isinstance(order, bool) or not isinstance(order, Integral) or order not in (4, 6, 8, 10):
        raise ValueError("even cardinal order4..10 required")
    if len(shape) != 3 or any(
        isinstance(v, bool) or not isinstance(v, Integral) or not order <= v <= 2**30 for v in shape
    ):
        raise ValueError("three integer assignment dimensions in [order,2**30] required")
    if (
        isinstance(max_points, bool)
        or not isinstance(max_points, Integral)
        or not 0 <= max_points <= 2**30
    ):
        raise ValueError("bounded integer max_points required")
    if np.iscomplexobj(origin):
        raise ValueError("real auxiliary origin required")
    origin = np.asarray(origin, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("finite three-vector auxiliary origin required")
    if dtype not in ("float32", "float64", np.dtype("float32"), np.dtype("float64")):
        raise ValueError("float32 or float64 weight dtype required")
    return origin.copy(), int(order), tuple(map(int, shape)), np.dtype(dtype), int(max_points)


def _source(dtype, order):
    return r"""
    typedef WEIGHT Weight;
    extern "C" __global__ void cardinal_stencil(
        const double *point,const double ox,const double oy,const double oz,
        const int count,const int sx,const int sy,const int sz,
        int *first,Weight *weights,int *error) {
      long long axis=(long long)blockDim.x*blockIdx.x+threadIdx.x;
      if(axis>=3LL*count)return;
      int component=axis%3;
      double origin=component==0?ox:(component==1?oy:oz);
      int shape=component==0?sx:(component==1?sy:sz);
      double coordinate=point[axis]-origin;
      if(!isfinite(coordinate)||fabs(coordinate)>1073741824.0) {
        atomicOr(error,1); return;
      }
      long long start=(long long)floor(coordinate)-ORDER/2+1;
      if(start<0||start+ORDER>shape) {atomicOr(error,2);return;}
      double argument=coordinate-(double)start;
      first[axis]=(int)start;
      #pragma unroll
      for(int i=0;i<ORDER;i++) {
        double value=1.;
        #pragma unroll
        for(int j=0;j<ORDER;j++) if(i!=j)value*=(argument-(double)j)/(double)(i-j);
        if(!isfinite(value))atomicOr(error,4);
        weights[axis*ORDER+i]=(Weight)value;
      }
    }
    """.replace("WEIGHT", "float" if np.dtype(dtype) == np.dtype("float32") else "double").replace(
        "ORDER", str(order)
    )


def cardinal_stencil_gpu(
    points,
    origin,
    order,
    shape,
    *,
    dtype="float32",
    pool,
    max_points=1_000_000,
    max_new_bytes=256 * 1024**2,
):
    """Return fresh `(first_i32, weights, diagnostics)` using caller-owned pool.

    Accepts host arrays or a same-device CuPy array. A contiguous f64 device
    array is read directly; otherwise a private f64 coordinate copy is made.
    This does not expose or borrow Taichi internals. All calls are synchronous
    on their entry stream, making copy/kernel wall timings explicit.
    """
    import cupy as cp

    origin, order, shape, dtype, max_points = _parameters(origin, order, shape, dtype, max_points)
    if not isinstance(pool, cp.cuda.MemoryPool) or not 0 < pool.get_limit() < 2**63:
        raise ValueError("caller must supply a capped CuPy MemoryPool")
    if (
        isinstance(max_new_bytes, bool)
        or not isinstance(max_new_bytes, Integral)
        or max_new_bytes <= 0
    ):
        raise ValueError("positive integer max_new_bytes required")
    stream = cp.cuda.get_current_stream()
    if isinstance(points, cp.ndarray) and points.device.id != cp.cuda.runtime.getDevice():
        raise ValueError("device points must belong to the current CUDA device")
    if not hasattr(points, "shape"):
        points = np.asarray(points)
    if len(points.shape) != 2 or points.shape[1:] != (3,) or len(points) > max_points:
        raise ValueError("bounded normalized point array (N,3) required")
    if np.dtype(points.dtype).kind not in "biuf":
        raise ValueError("real numeric normalized coordinates required")
    count = len(points)
    borrow = (
        isinstance(points, cp.ndarray) and points.dtype == cp.float64 and points.flags.c_contiguous
    )
    declared_bytes = count * (12 + 3 * order * dtype.itemsize + (0 if borrow else 24)) + 4
    if declared_bytes > max_new_bytes:
        raise MemoryError("cardinal coordinate/output validation exceeds max_new_bytes")
    # No operator action occurs before validation. Input arrays are never used
    # as scratch or passed to a write-enabled kernel argument.
    stream.synchronize()
    start = time.perf_counter()
    before_used, before_reserved = pool.used_bytes(), pool.total_bytes()
    with cp.cuda.using_allocator(pool.malloc):
        copy_started = time.perf_counter()
        coordinate = points if borrow else cp.asarray(points, dtype=cp.float64, order="C")
        stream.synchronize()
        copy_seconds = time.perf_counter() - copy_started
        first = cp.empty((count, 3), dtype=cp.int32)
        weights = cp.empty((count, 3, order), dtype=dtype)
        error = cp.zeros(1, dtype=cp.int32)
        kernel_started = time.perf_counter()
        if count:
            kernel = cp.RawKernel(
                _source(dtype, order), "cardinal_stencil", options=("--std=c++11",)
            )
            kernel(
                ((count * 3 + 255) // 256,),
                (256,),
                (
                    coordinate,
                    *map(np.float64, origin),
                    np.int32(count),
                    *map(np.int32, shape),
                    first,
                    weights,
                    error,
                ),
            )
        stream.synchronize()
        kernel_seconds = time.perf_counter() - kernel_started
        status = int(error.item())
        if status:
            raise ValueError(
                f"device cardinal stencil validation failed (flags={status}); no stencil published"
            )
        diagnostics = {
            "production_admissible": False,
            "coordinate_normalization_changed": False,
            "source_points_mutated": False,
            "point_count": count,
            "order": order,
            "dtype": dtype.name,
            "declared_new_bytes": declared_bytes,
            "borrowed_f64_device_coordinates": borrow,
            "copy_seconds": copy_seconds,
            "kernel_and_check_seconds": kernel_seconds,
            "wall_seconds": time.perf_counter() - start,
            "pool_used_before": before_used,
            "pool_reserved_before": before_reserved,
            "pool_used_after": pool.used_bytes(),
            "pool_reserved_after": pool.total_bytes(),
            "pool_limit": pool.get_limit(),
        }
    if not all(
        math.isfinite(diagnostics[name])
        for name in ("copy_seconds", "kernel_and_check_seconds", "wall_seconds")
    ):
        raise RuntimeError("invalid qualification timer")
    return first, weights, diagnostics
