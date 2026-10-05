"""Finite Gaussian fields with private GPU allocations.

Used by Gaussian SlipSlab sessions. Finite mesh/core accuracy and
infinite-tail validation remain separate caller responsibilities; this storage
solver alone grants neither. No CuPy import at module load.
"""

from contextlib import contextmanager
import math
import time

import numpy as np
from scipy.fft import next_fast_len

from .coordinates import finite_images, slab_coordinates
from .correction import GaussianCoreCorrectionGPU
from .planning import (
    correction_query_reserve,
    device_field_execution_plan,
    field_execution_plan,
    query_batch_size,
    required_channels,
)
from .runtime import CUDAMemoryPool, FFTPlanPair, positive_integer
from .stencil import cardinal_stencil_gpu


def _logical_stencils_fit(lattice, origin, order, shape, fft_shape):
    """Conservative validation to the *logical* linear-convolution output.

    Source assignment occupies only logical indices [0,n-1]. If every query
    stencil also lies there, all source/query lags lie in [-(n-1),n-1], which
    is exactly the signed kernel support represented by padding >=2*n-1.
    FFT padding itself is NEVER a query domain. A one-ULP enclosure of the
    host subtraction covers the device's rounded f64 subtraction before
    floor; a point exactly at a limiting stencil boundary may conservatively
    miss. The actual GPU stencil builder checks every evaluated query.
    """
    points, offset = np.asarray(lattice, dtype=np.float64), np.asarray(origin, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or offset.shape != (3,)
        or not np.isfinite(points).all()
        or not np.isfinite(offset).all()
    ):
        raise ValueError("finite normalized query points and logical origin required")
    if (
        type(order) is not int
        or order not in (4, 6, 8, 10)
        or len(shape) != 3
        or any(type(n) is not int or not order <= n <= 2**30 for n in shape)
        or (
            fft_shape is not None
            and (
                len(fft_shape) != 3
                or any(
                    type(f) is not int or f < 2 * n - 1
                    for n, f in zip(shape, fft_shape, strict=True)
                )
            )
        )
    ):
        raise RuntimeError("invalid logical/linear-convolution domain metadata")
    with np.errstate(over="ignore", invalid="ignore"):
        coordinate = points - offset
    if not np.isfinite(coordinate).all() or np.any(np.abs(coordinate) > 2**30):
        return False
    first_lower = np.floor(np.nextafter(coordinate, -np.inf)) - order // 2 + 1
    first_upper = np.floor(np.nextafter(coordinate, np.inf)) - order // 2 + 1
    return bool(np.all(first_lower >= 0) and np.all(first_upper + order <= np.asarray(shape)))


def _target_stencil_window(lattice, origin, order, shape):
    """Integer output crop enclosing every checked initial query stencil.

    Keep the original logical origin for coordinate subtraction and cardinal
    weights. The crop changes only which inverse-FFT cells are retained.
    One-ULP bounds cover the same device subtraction as logical validation.
    """
    if len(lattice) == 0 or not _logical_stencils_fit(lattice, origin, order, shape, None):
        raise ValueError("initial query stencils must fit the logical field")
    coordinate = np.asarray(lattice, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    lower = np.floor(np.nextafter(coordinate, -np.inf)).min(axis=0) - order // 2 + 1
    upper = np.floor(np.nextafter(coordinate, np.inf)).max(axis=0) - order // 2 + 1 + order
    start = tuple(lower.astype(np.int64).tolist())
    retained = tuple((upper - lower).astype(np.int64).tolist())
    return start, retained


def _retained_stencils_fit(
    lattice, origin, order, shape, fft_shape, start, retained_shape, *, source_shape=None
):
    """Validate only complete original stencils inside the retained output crop."""
    if (
        len(start) != 3
        or len(retained_shape) != 3
        or any(
            type(s) is not int or type(n) is not int or s < 0 or n < order or s + n > full
            for s, n, full in zip(start, retained_shape, shape, strict=True)
        )
    ):
        raise RuntimeError("invalid retained query-window metadata")
    if source_shape is not None:
        if (
            len(source_shape) != 3
            or len(fft_shape) != 3
            or any(
                type(s) is not int or not order <= s <= full or type(f) is not int or f < s + q - 1
                for s, q, full, f in zip(
                    source_shape, retained_shape, shape, fft_shape, strict=True
                )
            )
        ):
            raise RuntimeError("invalid source/query linear-convolution metadata")
        fft_shape = None
    if not _logical_stencils_fit(lattice, origin, order, shape, fft_shape):
        return False
    coordinate = np.asarray(lattice, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    lower = np.floor(np.nextafter(coordinate, -np.inf)) - order // 2 + 1
    upper = np.floor(np.nextafter(coordinate, np.inf)) - order // 2 + 1 + order
    return bool(
        np.all(lower >= np.asarray(start))
        and np.all(upper <= np.asarray(start) + np.asarray(retained_shape))
    )


_GAUSSIAN_RADIAL_CUDA = r"""    if(rho2<(real)1) {
      real term=1,sa=0,sb=0;
      for(int n=0;n<24;++n) {
        sa+=term/(real)(2*n+3); sb+=2*term/(real)(2*n+5);
        term*=(-rho2)/(real)(n+1);
      }
      a=pi15*sa/(tau*tau*tau); b=pi15*sb/(tau*tau*tau*tau*tau);
    } else {
      real radius=SQRT(r2),rho=radius/tau,e=EXP(-rho2);
      real q=(ERF(rho)-2*rho*e/SQRT(pi))/(4*pi);
      a=q/(r2*radius); b=3*q/(r2*r2*radius)-pi15*e/(tau*tau*tau*r2);
    }
"""


_CUDA = (
    r"""
typedef REAL real;
extern "C" __global__ void scatter(
    const long long total, const int order, const int *first, const real *weights,
    const real *gamma,const int ox,const int oy,const int oz,
    const int fy, const int fz, const long long volume, real *grid) {
  long long lane=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  const int stencil=order*order*order;
  if(lane>=total) return;
  const int particle=lane/stencil;
  int offset=lane%stencil;
  const int c=offset%order; offset/=order;
  const int b=offset%order, a=offset/order;
  const int *base=first+3*particle;
  const real *w=weights+3*order*particle;
  real factor=w[a]*w[order+b]*w[2*order+c];
  long long index=((long long)(base[0]+a-ox)*fy+base[1]+b-oy)*fz+base[2]+c-oz;
  for(int component=0;component<3;++component)
    atomicAdd(grid+component*volume+index, factor*gamma[3*particle+component]);
}
extern "C" __global__ void gaussian_kernel(
    const long long volume, const int sx,const int sy,const int sz,
    const int tx,const int ty,const int tz,const int fx,const int fy,const int fz,
    const long long ox,const long long oy,const long long oz,
    const real hx,const real hy,const real hz,
    const long long *shifts,const int images,const real tau,
    const int axis,const int derivative,real *kernel) {
  long long lane=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(lane>=volume) return;
  int iz=lane%fz, iy=(lane/fz)%fy, ix=lane/((long long)fy*fz);
  if((ix>=tx && ix<fx-sx+1)||(iy>=ty && iy<fy-sy+1)||(iz>=tz && iz<fz-sz+1)) {
    kernel[lane]=0; return;
  }
  int lx=ix<tx?ix:ix-fx, ly=iy<ty?iy:iy-fy, lz=iz<tz?iz:iz-fz;
  real value=0;
  const real pi=(real)3.141592653589793238462643383279502884;
  const real pi15=(real)0.17958712212516656168908198362769276;
  for(int image=0;image<images;++image) {
    real r[3]={(real)((long long)lx+ox)*hx,(real)((long long)ly+oy)*hy,
               (real)((long long)lz+oz-shifts[image])*hz};
    real r2=r[0]*r[0]+r[1]*r[1]+r[2]*r[2], rho2=r2/(tau*tau), a,b;
"""
    + _GAUSSIAN_RADIAL_CUDA
    + r"""    value+=derivative<0 ? -a*r[axis] : b*r[axis]*r[derivative]-(axis==derivative?a:0);
  }
  kernel[lane]=value;
}
extern "C" __global__ void product_add(
    const long long count, const real *kernel, const real *density,
    const real sign, real *output) {
  long long lane=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(lane>=count) return;
  real kr=kernel[2*lane],ki=kernel[2*lane+1],dr=density[2*lane],di=density[2*lane+1];
  output[2*lane]+=sign*(kr*dr-ki*di);
  output[2*lane+1]+=sign*(kr*di+ki*dr);
}
extern "C" __global__ void gather(
    const int count,const int order,const int *first,const real *weights,
    const int ox,const int oy,const int oz,const int fy,const int fz,
    const real *field, const int component,const int components,real *output) {
  int target=blockIdx.x*blockDim.x+threadIdx.x;
  if(target>=count) return;
  const int *base=first+3*target;
  const real *w=weights+3*order*target;
  double value=0;
  for(int a=0;a<order;++a) for(int b=0;b<order;++b) for(int c=0;c<order;++c) {
    long long index=((long long)(base[0]+a-ox)*fy+base[1]+b-oy)*fz+base[2]+c-oz;
    value+=(double)w[a]*(double)w[order+b]*(double)w[2*order+c]*(double)field[index];
  }
  output[components*(long long)target+component]=(real)value;
}
"""
)


_GAUSSIAN_CHANNEL_BODY = (
    r"""  long long lane=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(lane>=volume) return;
  int iz=lane%fz, iy=(lane/fz)%fy, ix=lane/((long long)fy*fz);
  real values[9]={0,0,0,0,0,0,0,0,0};
  if(!((ix>=tx && ix<fx-sx+1)||(iy>=ty && iy<fy-sy+1)||(iz>=tz && iz<fz-sz+1))) {
    int lx=ix<tx?ix:ix-fx, ly=iy<ty?iy:iy-fy, lz=iz<tz?iz:iz-fz;
    const real pi=(real)3.141592653589793238462643383279502884;
    const real pi15=(real)0.17958712212516656168908198362769276;
    for(int image=0;image<images;++image) {
      real r[3]={(real)((long long)lx+ox)*hx,(real)((long long)ly+oy)*hy,
                 (real)((long long)lz+oz-shifts[image])*hz};
      real r2=r[0]*r[0]+r[1]*r[1]+r[2]*r[2], rho2=r2/(tau*tau), a,b;
"""
    + _GAUSSIAN_RADIAL_CUDA
    + r"""      values[0]+=-a*r[0]; values[1]+=-a*r[1]; values[2]+=-a*r[2];
      values[3]+=b*r[0]*r[0]-a;
      values[4]+=b*r[0]*r[1]; values[5]+=b*r[0]*r[2];
      values[6]+=b*r[1]*r[1]-a;
      values[7]+=b*r[1]*r[2]; values[8]+=b*r[2]*r[2]-a;
    }
  }
"""
)


def _gaussian_channel_kernel(name, *, batch):
    arguments = "const int *channels,const int channel_count," if batch else ""
    publication = (
        "  for(int slot=0;slot<channel_count;++slot) kernel[slot*volume+lane]=values[channels[slot]];\n"
        if batch
        else "  for(int channel=0;channel<9;++channel) kernel[channel*volume+lane]=values[channel];\n"
    )
    return (
        f'extern "C" __global__ void {name}(\n'
        "    const long long volume,const int sx,const int sy,const int sz,\n"
        "    const int tx,const int ty,const int tz,const int fx,const int fy,const int fz,\n"
        "    const long long ox,const long long oy,const long long oz,\n"
        "    const real hx,const real hy,const real hz,\n"
        f"    const long long *shifts,const int images,const real tau,{arguments}real *kernel) {{\n"
        + _GAUSSIAN_CHANNEL_BODY
        + publication
        + "}\n"
    )


_FUSED_CUDA = _gaussian_channel_kernel("gaussian_kernel_fused", batch=False)


_STREAM_CUDA = r"""
extern "C" __global__ void scatter_component(
    const long long total, const int order, const int *first, const real *weights,
    const real *gamma,const int ox,const int oy,const int oz,
    const int fy, const int fz, const int component, real *grid) {
  long long lane=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  const int stencil=order*order*order;
  if(lane>=total) return;
  const int particle=lane/stencil;
  int offset=lane%stencil;
  const int c=offset%order; offset/=order;
  const int b=offset%order, a=offset/order;
  const int *base=first+3*particle;
  const real *w=weights+3*order*particle;
  real factor=w[a]*w[order+b]*w[2*order+c];
  long long index=((long long)(base[0]+a-ox)*fy+base[1]+b-oy)*fz+base[2]+c-oz;
  atomicAdd(grid+index, factor*gamma[3*particle+component]);
}
"""


_BATCH_CUDA = _gaussian_channel_kernel("gaussian_kernel_batch", batch=True)


def _cuda_source(dtype):
    scalar = "float" if dtype == "float32" else "double"
    suffix = "f" if dtype == "float32" else ""
    definitions = (
        f"#define REAL {scalar}\n"
        f"#define SQRT sqrt{suffix}\n"
        f"#define EXP exp{suffix}\n"
        f"#define ERF erf{suffix}\n"
    )
    return definitions + _CUDA + _FUSED_CUDA + _STREAM_CUDA + _BATCH_CUDA


# Each pair is (gradient axis, differentiation axis); -1 denotes velocity.
_CHANNEL_COMPONENTS = (
    ((0, -1),),
    ((1, -1),),
    ((2, -1),),
    ((0, 0),),
    ((0, 1), (1, 0)),
    ((0, 2), (2, 0)),
    ((1, 1),),
    ((1, 2), (2, 1)),
    ((2, 2),),
)


def channel_routes(channel):
    """Map a symmetric potential derivative to curl output products."""
    routes = []
    for axis, derivative in _CHANNEL_COMPONENTS[channel]:
        for component, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
            if axis in (a, b):
                source = b if axis == a else a
                sign = 1 if axis == a else -1
                column = component if derivative < 0 else 3 + 3 * component + derivative
                routes.append((column, source, sign))
    return tuple(routes)


class GaussianImageFields:
    """Owned finite source-only field; NOT an image-tail or solver validation.

    Default image-only mode excludes the primary descriptor (0, False).
    Explicit source_only_primary=True permits a coherent source-only target
    field; it NEVER implements particle pair-averaged-core self induction.
    All evaluate outputs INCLUDE local narrow-minus-broad core correction.
    Sources are immutable private host snapshots with no replacement API.
    Queries must fit the retained initial-query stencil window inside the
    original logical grid; there is no extrapolation.
    Dimensional tau, spacing and cutoff are deliberately explicit here.
    """

    def __init__(
        self,
        source_x,
        source_gamma,
        source_sigma,
        initial_targets,
        *,
        zmin,
        zmax,
        tau,
        spacing,
        cutoff,
        order=10,
        dtype="float32",
        correction_dtype="float32",
        max_scratch_bytes=2 * 1024**3,
        max_correction_bytes=256 * 1024**2,
        max_plan_bytes=128 * 1024**2,
        max_total_bytes=2304 * 1024**2,
        max_images=513,
        max_query_points=1_000_000,
        source_only_primary=False,
        profile=False,
        _lattice_origin=None,
    ):
        self._memory_pool = self._plans = self._correction = None
        self._compact_fields = self._kernel_scratch = self._inverse = None
        self._result_spectra = ()
        self.spectra, self._source_stencils = {}, {}
        self._prepared_images = self._prepared_world_images = None
        self.closed = False
        self._cleanup_failure = None
        self.profile = bool(profile)
        if type(source_only_primary) is not bool:
            raise ValueError("source_only_primary must be an explicit boolean")
        self.source_only_primary = source_only_primary
        self.max_images = positive_integer(max_images, "max_images")
        self.max_query_points = positive_integer(max_query_points, "max_query_points")
        if self.max_query_points > 2**30:
            raise ValueError("query count exceeds index range")
        self.max_scratch_bytes = positive_integer(max_scratch_bytes, "max_scratch_bytes")
        self.max_correction_bytes = positive_integer(max_correction_bytes, "max_correction_bytes")
        self.max_plan_bytes = positive_integer(max_plan_bytes, "max_plan_bytes")
        self.max_total_bytes = positive_integer(max_total_bytes, "max_total_bytes")
        if self.max_scratch_bytes + self.max_correction_bytes > self.max_total_bytes:
            raise MemoryError("combined array-pool limits exceed total cap")
        if self.max_plan_bytes >= self.max_scratch_bytes:
            raise ValueError("plan reserve must be smaller than smooth pool")
        if dtype not in ("float32", "float64") or correction_dtype not in ("float32", "float64"):
            raise ValueError("float32/float64 field and correction precision required")
        if order not in (4, 6, 8, 10) or isinstance(order, bool):
            raise ValueError("even cardinal order4..10 required")
        self.dtype = np.dtype(dtype)
        self.real_type = np.float32 if dtype == "float32" else np.float64
        self.complex_type = np.complex64 if dtype == "float32" else np.complex128
        if not all(math.isfinite(v) and v > 0 for v in (tau, spacing, cutoff)):
            raise ValueError("finite positive tau/spacing/cutoff required")
        self.zmin, self.zmax, self.tau = float(zmin), float(zmax), float(tau)
        self.spacing, self.cutoff, self.order = float(spacing), float(cutoff), int(order)
        self.host_x = self._host_array(source_x, "source")
        self.host_gamma = self._host_array(source_gamma, "strength")
        self.host_targets = self._host_array(initial_targets, "target")
        if hasattr(source_sigma, "__cuda_array_interface__") or np.iscomplexobj(source_sigma):
            raise ValueError("real host source cores required")
        self._host_sigma = self._immutable_snapshot(
            np.array(source_sigma, dtype=np.float64, copy=True, order="C")
        )
        count = len(self.host_x)
        if (
            not 1 <= count <= np.iinfo(np.int32).max
            or self.host_gamma.shape != self.host_x.shape
            or self._host_sigma.shape != (count,)
            or not np.isfinite(self._host_sigma).all()
            or np.any(self._host_sigma <= 0)
            or np.any(self._host_sigma > self.tau)
        ):
            raise ValueError("matching finite sources and 0<source core<=tau required")
        if not 1 <= len(self.host_targets) <= self.max_query_points:
            raise ValueError("nonempty bounded initial target envelope required")
        x, self.steps, self.cells = slab_coordinates(self.host_x, zmin, zmax, spacing)
        q, _, _ = slab_coordinates(self.host_targets, zmin, zmax, spacing)
        if np.any(self.host_x[:, 2] < zmin) or np.any(self.host_x[:, 2] > zmax):
            raise ValueError("physical sources must lie within the slip slab")
        reflected = x.copy()
        reflected[:, 2] *= -1
        points = np.concatenate((x, reflected, q))
        self.origin = np.floor(points.min(axis=0)) - order
        if _lattice_origin is not None:
            supplied_origin = np.array(_lattice_origin, dtype=np.float64, copy=True)
            if (
                supplied_origin.shape != (3,)
                or not np.isfinite(supplied_origin).all()
                or np.any(supplied_origin != np.floor(supplied_origin))
            ):
                raise ValueError("shared lattice origin must contain three finite integers")
            self.origin = supplied_origin
        dimensions = np.ceil(points.max(axis=0) - self.origin) + order + 1
        if (
            not np.isfinite(dimensions).all()
            or np.any(dimensions < 1)
            or np.any(dimensions > 2**30)
        ):
            raise ValueError("auxiliary logical shape exceeds bounds")
        self.shape = tuple(dimensions.astype(np.int64).tolist())
        self.source_start, self.source_shape = _target_stencil_window(
            np.concatenate((x, reflected)), self.origin, order, self.shape
        )
        self.compact_start, self.compact_shape = _target_stencil_window(
            q, self.origin, order, self.shape
        )
        self.lag_offset = tuple(
            target - source
            for target, source in zip(self.compact_start, self.source_start, strict=True)
        )
        # Only target stencil cells are observed. Every queried source lag
        # lies in [-(source_size-1), target_size-1], so this shorter padding
        # represents the same linear convolution without filling the empty
        # distance between source and query windows.
        self.fft_shape = tuple(
            next_fast_len(source + target - 1)
            for source, target in zip(self.source_shape, self.compact_shape, strict=True)
        )
        self.volume = math.prod(self.fft_shape)
        self.spectrum_shape = (*self.fft_shape[:2], self.fft_shape[2] // 2 + 1)
        self.spectrum_count = math.prod(self.spectrum_shape)
        self._compact_slices = tuple(slice(0, n) for n in self.compact_shape)
        self.compact_bytes = 12 * math.prod(self.compact_shape) * self.dtype.itemsize
        self.execution_plan = field_execution_plan(
            self.shape,
            self.fft_shape,
            count,
            len(q),
            order,
            self.dtype.itemsize,
            self.max_scratch_bytes,
            0,
            retained_shape=self.compact_shape,
        )
        self.estimated_field_bytes = self.execution_plan.field_bytes
        started = time.perf_counter()
        try:
            self._memory_pool = CUDAMemoryPool(self.max_scratch_bytes)
            self.cp, self.pool = self._memory_pool.cp, self._memory_pool.pool
            self.stream = self._memory_pool.stream
            # Build the exact source index before measuring live free memory.
            # Its configured cap is an upper bound, not an allocation required
            # by every source/query block. Keep the handle through failures.
            self._correction = GaussianCoreCorrectionGPU.__new__(GaussianCoreCorrectionGPU)
            self._correction.__init__(
                self.host_x,
                self.host_gamma,
                self._host_sigma,
                tau=tau,
                cutoff=cutoff,
                accumulation_dtype=correction_dtype,
                max_scratch_bytes=self.max_correction_bytes,
                max_images=self.max_images,
            )
            self._correction.release_build_scratch()
            free, total = self.cp.cuda.runtime.memGetInfo()
            query_reserve = correction_query_reserve(1, self._correction.dtype.itemsize)
            measured_work = None
            try:
                self.execution_plan, self.effective_smooth_pool_cap = device_field_execution_plan(
                    self.shape,
                    self.fft_shape,
                    count,
                    len(q),
                    order,
                    self.dtype.itemsize,
                    self.max_scratch_bytes,
                    self.max_plan_bytes,
                    query_reserve,
                    int(free),
                    retained_shape=self.compact_shape,
                )
            except MemoryError:
                # Reject impossible array arrays before constructing plans.
                # Driver free was captured before probing, so measured plan
                # storage is counted once against this complete live budget.
                _, self.effective_smooth_pool_cap = device_field_execution_plan(
                    self.shape,
                    self.fft_shape,
                    count,
                    len(q),
                    order,
                    self.dtype.itemsize,
                    self.max_scratch_bytes,
                    0,
                    query_reserve,
                    int(free),
                    retained_shape=self.compact_shape,
                )
                self.pool.set_limit(size=self.effective_smooth_pool_cap)
                self._plans = FFTPlanPair.__new__(FFTPlanPair)
                self._plans.__init__(
                    self._memory_pool,
                    self.fft_shape,
                    self.dtype,
                    self.max_plan_bytes,
                    single_workspace=True,
                )
                measured_work = self._plans.measure_directional_work()
                combined_work = sum(measured_work)
                if combined_work <= self.max_plan_bytes:
                    try:
                        candidate = field_execution_plan(
                            self.shape,
                            self.fft_shape,
                            count,
                            len(q),
                            order,
                            self.dtype.itemsize,
                            self.effective_smooth_pool_cap,
                            combined_work,
                            retained_shape=self.compact_shape,
                        )
                    except MemoryError:
                        candidate = None
                else:
                    candidate = None
                if candidate is not None and candidate.mode == "all_channels":
                    self.execution_plan = candidate
                    self._plans.retain_both_directions()
                else:
                    self.execution_plan = field_execution_plan(
                        self.shape,
                        self.fft_shape,
                        count,
                        len(q),
                        order,
                        self.dtype.itemsize,
                        self.effective_smooth_pool_cap,
                        max(measured_work),
                        retained_shape=self.compact_shape,
                        allow_all_channels=False,
                    )
            self.estimated_field_bytes = self.execution_plan.field_bytes
            self.pool.set_limit(size=self.effective_smooth_pool_cap)
            self.device_checks = {
                "free_device_bytes": int(free),
                "total_device_bytes": int(total),
                "configured_smooth_pool_cap": self.max_scratch_bytes,
                "effective_smooth_pool_cap": self.effective_smooth_pool_cap,
                "configured_correction_pool_cap": self.max_correction_bytes,
                "correction_owned_bytes": self._correction.pool.total_bytes(),
                "correction_reserve_bytes": query_reserve,
            }
            if measured_work is not None:
                self.device_checks["measured_forward_workspace_bytes"] = measured_work[0]
                self.device_checks["measured_inverse_workspace_bytes"] = measured_work[1]
                self.device_checks["workspace_reserve_bytes"] = (
                    sum(measured_work)
                    if self.execution_plan.mode == "all_channels"
                    else max(measured_work)
                )
            code = _cuda_source(dtype)
            self._program = self.cp.RawModule(code=code, options=("--std=c++11",), backend="nvrtc")
            for odd, points in ((False, x), (True, reflected)):
                first, weights, _ = cardinal_stencil_gpu(
                    points,
                    self.origin,
                    order,
                    self.shape,
                    dtype=self.dtype,
                    pool=self.pool,
                    max_points=count,
                    max_new_bytes=self.max_scratch_bytes,
                )
                self._source_stencils[odd] = first, weights
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(
                    f"Gaussian field construction cleanup also failed: {cleanup_error!r}"
                )
                raise error from cleanup_error
            raise
        self.initial_diagnostics = {
            "setup_seconds": time.perf_counter() - started,
            "runtime_admissible": False,
            "tail_bound_checked": False,
        }

    @staticmethod
    def _immutable_snapshot(array):
        # A write-disabled owning ndarray is not immutable: setflags(write=True)
        # can enable mutation and desynchronise cached smooth families from the
        # core-correction source copy. A bytes solver cannot be made writable.
        return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)

    @staticmethod
    def _host_array(value, name):
        if hasattr(value, "__cuda_array_interface__") or np.iscomplexobj(value):
            raise ValueError(f"{name} requires real host coordinates/values")
        array = np.array(value, dtype=np.float64, copy=True, order="C")
        if array.ndim != 2 or array.shape[1:] != (3,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite (N,3)")
        return GaussianImageFields._immutable_snapshot(array)

    def _validate_image_indices(self, integer):
        # The CUDA kernel casts the DIFFERENCE lz-shift, not shift alone.
        # Shifted lags span [offset-(source_size-1), offset+target_size-1].
        # Check every endpoint before the cast; FFT indices alone are smaller
        # than physical lags for remote queries and cannot check_context precision.
        exact_limit = 2 ** (24 if self.dtype == np.dtype("float32") else 53)
        bounds = [
            (
                self.lag_offset[a] - self.source_shape[a] + 1,
                self.lag_offset[a] + self.compact_shape[a] - 1,
            )
            for a in range(3)
        ]
        if any(max(abs(low), abs(high)) > exact_limit for low, high in bounds[:2]) or any(
            max(abs(bounds[2][0] - shift), abs(bounds[2][1] - shift)) > exact_limit
            for shift, _ in integer
        ):
            raise ValueError("image lags exceed exact kernel-coordinate qualification")

    def _check_context(self):
        if self.closed or self._memory_pool is None:
            raise RuntimeError("finite Gaussian image solver is closed")
        self._memory_pool.check_context()

    def can_evaluate_targets(self, targets):
        """Whether complete target stencils fit this retained output window.

        This read-only geometry check allocates no GPU storage, performs no
        extrapolation and validates no tail/correction or finite-mesh error.
        Session callers must still validate exact sources/role and the new
        query's mathematical bounds. Queries can differ from the initial
        points if their complete stencils fit the retained integer window.
        True is geometry validation only, not a promise of query scratch
        capacity; evaluation retains its capped allocation and device guards.
        """
        self._check_context()
        if self._prepared_images is None:
            raise RuntimeError("no successfully prepared finite image field")
        _, world, _ = finite_images(
            self._prepared_images,
            self.zmin,
            self.zmax,
            self.cells,
            self.max_images,
            include_primary=self.source_only_primary,
        )
        if world != self._prepared_world_images:
            raise RuntimeError("prepared Gaussian field image descriptors changed")
        if hasattr(targets, "__cuda_array_interface__"):
            raise TypeError("explicit host target snapshot required")
        if np.iscomplexobj(targets):
            raise ValueError("real target coordinates required")
        query = np.asarray(targets, dtype=np.float64)
        if (
            query.ndim != 2
            or query.shape[1:] != (3,)
            or len(query) > self.max_query_points
            or not np.isfinite(query).all()
        ):
            raise ValueError("bounded finite targets (N,3) required")
        lattice, _, _ = slab_coordinates(query, self.zmin, self.zmax, float(self.steps[0]))
        return _retained_stencils_fit(
            lattice,
            self.origin,
            self.order,
            self.shape,
            self.fft_shape,
            self.compact_start,
            self.compact_shape,
            source_shape=self.source_shape,
        )

    @contextmanager
    def _phase(self, diagnostics, name):
        if not self.profile:
            yield
            return
        self.stream.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stream.synchronize()
            diagnostics.setdefault("passes", {}).setdefault(name, 0.0)
            diagnostics["passes"][name] += time.perf_counter() - started

    def _launch(self, name, count, arguments):
        self._program.get_function(name)(((count + 255) // 256,), (256,), arguments)

    def _kernel_arguments(self, shifts):
        """Shared signed-lag geometry for scalar, fused and batched channels."""
        return (
            np.int64(self.volume),
            *(np.int32(n) for n in self.source_shape),
            *(np.int32(n) for n in self.compact_shape),
            *(np.int32(n) for n in self.fft_shape),
            *(np.int64(n) for n in self.lag_offset),
            *(self.real_type(h) for h in self.steps),
            shifts,
            np.int32(len(shifts)),
            self.real_type(self.tau),
        )

    def _prepare_family(self, odd, diagnostics):
        if odd in self.spectra:
            return
        if self.execution_plan.source_families == 1:
            self.spectra.clear()
        with self._phase(diagnostics, "source_scatter_and_fft"):
            if odd not in self._source_stencils:
                points, _, _ = slab_coordinates(self.host_x, self.zmin, self.zmax, self.spacing)
                if odd:
                    points[:, 2] *= -1
                first, weights, _ = cardinal_stencil_gpu(
                    points,
                    self.origin,
                    self.order,
                    self.shape,
                    dtype=self.dtype,
                    pool=self.pool,
                    max_points=len(points),
                    max_new_bytes=self.max_scratch_bytes,
                )
                self._source_stencils[odd] = first, weights
            first, weights = self._source_stencils[odd]
            strengths = self.host_gamma.copy()
            if odd:
                strengths[:, :2] *= -1
            gamma = self.cp.asarray(strengths, dtype=self.dtype)
            if self.execution_plan.mode == "all_channels":
                if self._kernel_scratch is None:
                    grid = self.cp.zeros((3, *self.fft_shape), dtype=self.dtype)
                else:
                    # A late, previously absent family may be prepared after
                    # all persistent buffers exist. Reuse their first three
                    # radial grids instead of adding an unbudgeted 3V peak.
                    grid = self._kernel_scratch[:3]
                    grid.fill(0)
                self._launch(
                    "scatter",
                    len(first) * self.order**3,
                    (
                        np.int64(len(first) * self.order**3),
                        np.int32(self.order),
                        first,
                        weights,
                        gamma,
                        *(np.int32(n) for n in self.source_start),
                        np.int32(self.fft_shape[1]),
                        np.int32(self.fft_shape[2]),
                        np.int64(self.volume),
                        grid,
                    ),
                )
                self.spectra[odd] = tuple(
                    self._plans.rfft(grid[c], aligned_scratch=self._inverse) for c in range(3)
                )
            else:
                if self._kernel_scratch is None:
                    self._kernel_scratch = self.cp.empty(
                        (self.execution_plan.kernel_channels, *self.fft_shape), dtype=self.dtype
                    )
                grid = self._kernel_scratch[0]
                spectra = []
                for component in range(3):
                    grid.fill(0)
                    self._launch(
                        "scatter_component",
                        len(first) * self.order**3,
                        (
                            np.int64(len(first) * self.order**3),
                            np.int32(self.order),
                            first,
                            weights,
                            gamma,
                            *(np.int32(n) for n in self.source_start),
                            np.int32(self.fft_shape[1]),
                            np.int32(self.fft_shape[2]),
                            np.int32(component),
                            grid,
                        ),
                    )
                    spectra.append(self._plans.rfft(grid, aligned_scratch=self._inverse))
                self.spectra[odd] = tuple(spectra)
            del first, weights, gamma, grid
            self._source_stencils.pop(odd)

    def _stream_fields(self, families, diagnostics):
        """Exact same channel products with memory-bounded recomputation.

        Retaining two source families keeps the original twelve inverses.
        The lower-memory one-family mode completes each family separately
        and adds its compact real fields (twenty-four inverses). The latter
        changes rounding order, never the finite operator or interpolation.
        """
        cp, batch = self.cp, self.execution_plan.output_batch
        one_family = self.execution_plan.source_families == 1
        if one_family:
            self._compact_fields.fill(0)
        outer_families = (
            [((odd, shifts),) for odd, shifts in families.items()]
            if one_family
            else [tuple(families.items())]
        )
        for active in outer_families:
            for first in range(0, 12, batch):
                columns = tuple(range(first, min(first + batch, 12)))
                for spectrum in self._result_spectra:
                    spectrum.fill(0)
                for odd, shifts in active:
                    self._prepare_family(odd, diagnostics)
                    needed = required_channels(columns)
                    capacity = self.execution_plan.kernel_channels
                    for start in range(0, len(needed), capacity):
                        selected = needed[start : start + capacity]
                        device_channels = cp.asarray(selected, dtype=cp.int32)
                        with self._phase(diagnostics, "streamed_kernel_build"):
                            self._launch(
                                "gaussian_kernel_batch",
                                self.volume,
                                (
                                    *self._kernel_arguments(shifts),
                                    device_channels,
                                    np.int32(len(selected)),
                                    self._kernel_scratch,
                                ),
                            )
                            diagnostics["radial_kernel_launches"] += 1
                        for slot, channel in enumerate(selected):
                            with self._phase(diagnostics, "kernel_fft"):
                                kernel_hat = self._plans.rfft(
                                    self._kernel_scratch[slot], aligned_scratch=self._inverse
                                )
                                diagnostics["kernel_forward_transforms"] += 1
                            routes = [
                                route for route in channel_routes(channel) if route[0] in columns
                            ]
                            for column, source, sign in routes:
                                self._launch(
                                    "product_add",
                                    self.spectrum_count,
                                    (
                                        np.int64(self.spectrum_count),
                                        kernel_hat,
                                        self.spectra[odd][source],
                                        self.real_type(sign),
                                        self._result_spectra[column - first],
                                    ),
                                )
                            del kernel_hat
                        del device_channels
                for index, column in enumerate(columns):
                    with self._phase(diagnostics, "inverse_fft_and_gather"):
                        self._plans.irfft(self._result_spectra[index], self._inverse)
                        crop = self._inverse[self._compact_slices]
                        if one_family:
                            self._compact_fields[column] += crop
                        else:
                            cp.copyto(self._compact_fields[column], crop)
                        diagnostics["inverse_transforms"] += 1

    def _all_channel_fields(self, families, diagnostics):
        """Original fused layout, unchanged when its bounded field_bytes fits."""
        result_hat = self._result_spectra
        for spectrum in result_hat:
            spectrum.fill(0)
        for odd, shifts in families.items():
            with self._phase(diagnostics, "fused_kernel_build"):
                kernels = self._kernel_scratch
                self._launch(
                    "gaussian_kernel_fused",
                    self.volume,
                    (
                        *self._kernel_arguments(shifts),
                        kernels,
                    ),
                )
                diagnostics["radial_kernel_launches"] += 1
            for channel in range(9):
                with self._phase(diagnostics, "kernel_fft"):
                    kernel_hat = self._plans.rfft(kernels[channel], aligned_scratch=self._inverse)
                    diagnostics["kernel_forward_transforms"] += 1
                with self._phase(diagnostics, "spectral_product"):
                    for column, source, sign in channel_routes(channel):
                        self._launch(
                            "product_add",
                            self.spectrum_count,
                            (
                                np.int64(self.spectrum_count),
                                kernel_hat,
                                self.spectra[odd][source],
                                self.real_type(sign),
                                result_hat[column],
                            ),
                        )
                    del kernel_hat
        for column in range(12):
            with self._phase(diagnostics, "inverse_fft_and_gather"):
                self._plans.irfft(result_hat[column], self._inverse)
                self.cp.copyto(self._compact_fields[column], self._inverse[self._compact_slices])
                diagnostics["inverse_transforms"] += 1

    def prepare(self, images):
        """Prepare compact finite smooth fields; no tail or runtime validation."""
        self._check_context()
        cp = self.cp
        self._prepared_images = self._prepared_world_images = None
        descriptors, world, integer = finite_images(
            images,
            self.zmin,
            self.zmax,
            self.cells,
            self.max_images,
            include_primary=self.source_only_primary,
        )
        self._validate_image_indices(integer)
        diagnostics = {
            "runtime_admissible": False,
            "dtype": self.dtype.name,
            "variant": "fused-nine-radial-channels",
            "finite_image_count": len(integer),
            "world_images": world,
            "integer_images": integer,
            "shape": self.shape,
            "fft_shape": self.fft_shape,
            "compact_start": self.compact_start,
            "compact_shape": self.compact_shape,
            "source_start": self.source_start,
            "source_shape": self.source_shape,
            "lag_offset": self.lag_offset,
            "spacing": self.steps.tolist(),
            "estimated_field_bytes": self.estimated_field_bytes,
            "passes": {},
            "inverse_transforms": 0,
            "kernel_forward_transforms": 0,
            "radial_kernel_launches": 0,
            "core_correction_included": False,
        }
        diagnostics["execution_plan"] = vars(self.execution_plan).copy()
        diagnostics["device_checks"] = self.device_checks.copy()
        started = time.perf_counter()
        with self._memory_pool.allocation_scope():
            if self._plans is None:
                self._plans = FFTPlanPair(
                    self._memory_pool,
                    self.fft_shape,
                    self.dtype,
                    self.max_plan_bytes,
                    single_workspace=self.execution_plan.mode == "streamed",
                )
            alignment_before = (self._plans.alignment_copies, self._plans.alignment_copy_bytes)
            # This real grid was already included in every execution plan.
            # Allocate it before family FFTs as well: both three-component
            # source grids and radial grids can have unaligned odd-volume
            # channel views. No second FFT staging allocation is permitted.
            if self._inverse is None:
                self._inverse = cp.empty(self.fft_shape, dtype=self.dtype)
            families = {}
            for odd in (False, True):
                shifts = [shift for shift, reflection in integer if reflection == odd]
                if shifts:
                    if self.execution_plan.source_families == 2:
                        self._prepare_family(odd, diagnostics)
                    families[odd] = cp.asarray(shifts, dtype=cp.int64)
            # Keep the large allocations intact across calls. Releasing this
            # nine-grid block allowed subsequent smaller FFT allocations to
            # split it, eventually exceeding the unchanged private pool cap.
            # These buffers are never returned to callers; output owns its
            # allocation and remains valid after later evaluations.
            if self._kernel_scratch is None:
                self._kernel_scratch = cp.empty(
                    (self.execution_plan.kernel_channels, *self.fft_shape), dtype=self.dtype
                )
            if not self._result_spectra:
                self._result_spectra = tuple(
                    cp.empty(self.spectrum_shape, dtype=self.complex_type)
                    for _ in range(self.execution_plan.output_batch)
                )
            if self._compact_fields is None:
                self._compact_fields = cp.empty((12, *self.compact_shape), dtype=self.dtype)
            if self.execution_plan.mode == "all_channels":
                self._all_channel_fields(families, diagnostics)
            else:
                self._stream_fields(families, diagnostics)
            self.stream.synchronize()
            if not bool(cp.isfinite(self._compact_fields).all()):
                raise FloatingPointError("nonfinite fused smooth GPU output; nothing published")
            free, total = cp.cuda.runtime.memGetInfo()
            diagnostics.update(
                seconds=time.perf_counter() - started,
                pool_used_bytes=self.pool.used_bytes(),
                pool_reserved_bytes=self.pool.total_bytes(),
                plan_bytes=self._plans.work_bytes,
                plan_peak_work_bytes=self._plans.peak_work_bytes,
                plan_builds=self._plans.plan_builds,
                plan_build_seconds=self._plans.plan_build_seconds,
                fft_alignment_copies=self._plans.alignment_copies - alignment_before[0],
                fft_alignment_copy_bytes=self._plans.alignment_copy_bytes - alignment_before[1],
                device_free_bytes=free,
                device_total_bytes=total,
            )
            self._prepared_images, self._prepared_world_images = descriptors, world
            diagnostics.update(tail_bound_checked=False, compact_bytes=self.compact_bytes)
            return diagnostics

    def evaluate_prepared(self, targets):
        """Fresh bounded query batches, compact gather and exact correction.

        The complete public output is fresh device storage. Query coordinates,
        stencils and correction results live only for the active batch, so a
        later, larger query does not require a larger persistent field solver.
        """
        self._check_context()
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
        if not _retained_stencils_fit(
            lattice,
            self.origin,
            self.order,
            self.shape,
            self.fft_shape,
            self.compact_start,
            self.compact_shape,
            source_shape=self.source_shape,
        ):
            raise ValueError("query stencil lies outside the retained Gaussian field window")
        started = time.perf_counter()
        free, _ = self.cp.cuda.runtime.memGetInfo()
        batch = query_batch_size(
            len(q),
            self.order,
            self.dtype.itemsize,
            self._correction.dtype.itemsize,
            smooth_cap=self.pool.get_limit(),
            smooth_used=self.pool.used_bytes(),
            correction_cap=self._correction.pool.get_limit(),
            correction_used=self._correction.pool.used_bytes(),
            device_available=int(free)
            + self.pool.free_bytes()
            + self._correction.pool.free_bytes(),
        )
        with self._memory_pool.allocation_scope():
            # One allocation preserves the checked 12*N-element footprint.
            # Both public views are contiguous, so host copies need no
            # contiguous-device temporary in CuPy's unrelated default pool.
            output = self.cp.empty(12 * len(q), dtype=self.dtype)
            velocity = output[: 3 * len(q)].reshape(-1, 3)
            gradient = output[3 * len(q) :].reshape(-1, 3, 3)
            gather_seconds, batches, retries, first_target = 0.0, 0, 0, 0
            stencil, correction = None, None
            evaluated_images = set()
            try:
                while first_target < len(q):
                    count = min(batch, len(q) - first_target)
                    last_target = first_target + count
                    chunk_velocity = velocity[first_target:last_target]
                    chunk_gradient = gradient[first_target:last_target]
                    first = weight = correction_u = correction_j = None
                    retry = False
                    try:
                        first, weight, current_stencil = cardinal_stencil_gpu(
                            lattice[first_target:last_target],
                            self.origin,
                            self.order,
                            self.shape,
                            dtype=self.dtype,
                            pool=self.pool,
                            max_points=count,
                            max_new_bytes=self.max_scratch_bytes,
                        )
                        gather_started = time.perf_counter()
                        for column in range(12):
                            self._launch(
                                "gather",
                                count,
                                (
                                    np.int32(count),
                                    np.int32(self.order),
                                    first,
                                    weight,
                                    *(np.int32(n) for n in self.compact_start),
                                    np.int32(self.compact_shape[1]),
                                    np.int32(self.compact_shape[2]),
                                    self._compact_fields[column],
                                    np.int32(column if column < 3 else column - 3),
                                    np.int32(3 if column < 3 else 9),
                                    chunk_velocity if column < 3 else chunk_gradient,
                                ),
                            )
                        self.stream.synchronize()
                        current_gather_seconds = time.perf_counter() - gather_started
                        # Stencils and corrections do not need to coexist.
                        first = weight = None
                        correction_u, correction_j, current_correction = self._correction.evaluate(
                            q[first_target:last_target], self._prepared_world_images
                        )
                        chunk_velocity += correction_u
                        chunk_gradient += correction_j
                        correction_u = correction_j = None
                        self.stream.synchronize()
                        if not (
                            bool(self.cp.isfinite(chunk_velocity).all())
                            and bool(self.cp.isfinite(chunk_gradient).all())
                        ):
                            raise FloatingPointError("nonfinite compact query; nothing published")
                    except MemoryError:
                        if count == 1:
                            raise
                        # A driver allocation or a split cached block can fail
                        # despite scalar validation. Retry this unpublished
                        # chunk after reducing only its execution batch. Every
                        # gather overwrites all columns before correction.
                        batch, retry = max(1, count // 2), True
                        retries += 1
                    finally:
                        first = weight = correction_u = correction_j = None
                    if retry:
                        # Leave the exception scope first: its traceback can
                        # otherwise retain arrays from the failed allocation.
                        self.stream.synchronize()
                        self.pool.free_all_blocks()
                        self._correction.pool.free_all_blocks()
                        continue
                    gather_seconds += current_gather_seconds
                    if stencil is None:
                        stencil = current_stencil.copy()
                        stencil.update(point_count=len(q), max_batch_points=count)
                    else:
                        stencil["max_batch_points"] = max(stencil["max_batch_points"], count)
                        for name in (
                            "copy_seconds",
                            "kernel_and_check_seconds",
                            "wall_seconds",
                        ):
                            stencil[name] += current_stencil[name]
                        stencil["declared_new_bytes"] = max(
                            stencil["declared_new_bytes"], current_stencil["declared_new_bytes"]
                        )
                    if correction is None:
                        correction = current_correction.copy()
                        correction.update(
                            target_count=len(q),
                            candidate_pairs=0,
                            accepted_pairs=0,
                            query_seconds=0.0,
                        )
                    for name in ("candidate_pairs", "accepted_pairs", "query_seconds"):
                        correction[name] += current_correction[name]
                    for name in ("pool_high_water_bytes", "pool_reserved_bytes", "pool_used_bytes"):
                        correction[name] = max(correction[name], current_correction[name])
                    evaluated_images.update(current_correction["evaluated_images"])
                    first_target, batches = last_target, batches + 1
            except ValueError:
                # Rejected target stencils never mutate the prepared fields;
                # any partial query output is private and is not published.
                raise
            except BaseException:
                # CUDA/evaluation failures cannot leave a seemingly validated
                # reusable result.
                self._prepared_images = self._prepared_world_images = None
                raise
            if correction is None:
                # Empty queries keep the usual diagnostic shape without
                # allocating coordinate/stencil or correction workspaces.
                correction = {
                    **self._correction.diagnostics,
                    "target_count": 0,
                    "images": self._prepared_world_images,
                    "candidate_pairs": 0,
                    "accepted_pairs": 0,
                    "query_seconds": 0.0,
                }
                stencil = {"point_count": 0, "max_batch_points": 0, "declared_new_bytes": 0}
            correction["evaluated_images"] = [
                image for image in correction["images"] if image in evaluated_images
            ]
            correction["aabb_skipped_images"] = [
                image for image in correction["images"] if image not in evaluated_images
            ]
            correction["query_batches"] = batches
            return (
                velocity,
                gradient,
                {
                    "runtime_admissible": False,
                    "tail_bound_checked": False,
                    "core_correction_included": True,
                    "source_replaced": False,
                    "target_count": len(q),
                    "finite_image_count": len(self._prepared_images),
                    "compact_bytes": self.compact_bytes,
                    "compact_start": self.compact_start,
                    "compact_shape": self.compact_shape,
                    "stencil": stencil,
                    "gather_seconds": gather_seconds,
                    "correction": correction,
                    "query_seconds": time.perf_counter() - started,
                    "query_batches": batches,
                    "query_batch_size": batch,
                    "query_allocation_retries": retries,
                    "inverse_transforms": 0,
                    "source_scatters": 0,
                    "smooth_pool_reserved_bytes": self.pool.total_bytes(),
                    "correction_pool_reserved_bytes": self._correction.pool.total_bytes(),
                    "combined_pool_cap": self.max_total_bytes,
                },
            )

    def evaluate(self, images):
        """Initial queries, INCLUDING source-only local core correction."""
        self.prepare(images)
        return self.evaluate_prepared(self.host_targets)

    def close(self):
        if self._cleanup_failure is not None:
            raise RuntimeError(
                "Gaussian field GPU cleanup remains uncertain"
            ) from self._cleanup_failure
        if self.closed:
            return
        if self._memory_pool is None:
            self.closed = True
            return
        try:
            self._memory_pool.check_context()
            self.stream.synchronize()
        except BaseException as error:
            self._cleanup_failure = error
            self.closed = True
            raise
        self._prepared_images = self._prepared_world_images = None
        self.closed = True
        failure = None
        for name in ("_plans", "_correction"):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    failure = failure or error
                else:
                    setattr(self, name, None)
        self._compact_fields = self._kernel_scratch = self._inverse = None
        self._result_spectra = ()
        self.spectra.clear()
        self._source_stencils.clear()
        self._program = None
        try:
            self._memory_pool.close()
        except BaseException as error:
            failure = failure or error
        if failure is not None:
            self._cleanup_failure = failure
            raise failure

    def __enter__(self):
        self._check_context()
        return self

    def __exit__(self, *_):
        self.close()
