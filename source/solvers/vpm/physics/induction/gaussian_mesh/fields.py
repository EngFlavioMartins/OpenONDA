"""Finite Gaussian fields with private GPU ownership.

Used by Gaussian SlipSlab sessions. Finite mesh/core accuracy and
infinite-tail admission remain separate caller responsibilities; this storage
owner alone grants neither. No CuPy import at module load.
"""

from contextlib import contextmanager
import math
import time

import numpy as np
from scipy.fft import next_fast_len

from .coordinates import finite_images, slab_coordinates
from .correction import GaussianCoreCorrectionGPU
from .planning import (
    device_field_execution_plan,
    field_execution_plan,
    query_batch_size,
    required_channels,
)
from .runtime import DeviceOwner, FFTPlanPair, positive_integer
from .stencil import cardinal_stencil_gpu


def _logical_stencils_fit(lattice, origin, order, shape, fft_shape):
    """Conservative admission to the *logical* linear-convolution output.

    Source assignment occupies only logical indices [0,n-1]. If every query
    stencil also lies there, all source/query lags lie in [-(n-1),n-1], which
    is exactly the signed kernel support represented by padding >=2*n-1.
    FFT padding itself is NEVER a query domain. A one-ULP enclosure of the
    host subtraction covers the device's rounded f64 subtraction before
    floor; a point exactly at a limiting stencil boundary may conservatively
    miss. The actual GPU stencil builder remains authoritative on evaluation.
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
        or len(fft_shape) != 3
        or any(type(n) is not int or not order <= n <= 2**30 for n in shape)
        or any(type(f) is not int or f < 2 * n - 1 for n, f in zip(shape, fft_shape, strict=True))
    ):
        raise RuntimeError("invalid logical/linear-convolution domain metadata")
    with np.errstate(over="ignore", invalid="ignore"):
        coordinate = points - offset
    if not np.isfinite(coordinate).all() or np.any(np.abs(coordinate) > 2**30):
        return False
    first_lower = np.floor(np.nextafter(coordinate, -np.inf)) - order // 2 + 1
    first_upper = np.floor(np.nextafter(coordinate, np.inf)) - order // 2 + 1
    return bool(np.all(first_lower >= 0) and np.all(first_upper + order <= np.asarray(shape)))


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
    const real *gamma, const int fy, const int fz, const long long volume, real *grid) {
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
  long long index=((long long)(base[0]+a)*fy+base[1]+b)*fz+base[2]+c;
  for(int component=0;component<3;++component)
    atomicAdd(grid+component*volume+index, factor*gamma[3*particle+component]);
}
extern "C" __global__ void gaussian_kernel(
    const long long volume, const int nx,const int ny,const int nz,
    const int fx,const int fy,const int fz, const real hx,const real hy,const real hz,
    const long long *shifts,const int images,const real tau,
    const int axis,const int derivative,real *kernel) {
  long long lane=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(lane>=volume) return;
  int iz=lane%fz, iy=(lane/fz)%fy, ix=lane/((long long)fy*fz);
  if((ix>=nx && ix<fx-nx+1)||(iy>=ny && iy<fy-ny+1)||(iz>=nz && iz<fz-nz+1)) {
    kernel[lane]=0; return;
  }
  int lx=ix<nx?ix:ix-fx, ly=iy<ny?iy:iy-fy, lz=iz<nz?iz:iz-fz;
  real value=0;
  const real pi=(real)3.141592653589793238462643383279502884;
  const real pi15=(real)0.17958712212516656168908198362769276;
  for(int image=0;image<images;++image) {
    real r[3]={(real)lx*hx,(real)ly*hy,(real)((long long)lz-shifts[image])*hz};
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
    const int fy,const int fz,const real *field, const int component,real *output) {
  int target=blockIdx.x*blockDim.x+threadIdx.x;
  if(target>=count) return;
  const int *base=first+3*target;
  const real *w=weights+3*order*target;
  double value=0;
  for(int a=0;a<order;++a) for(int b=0;b<order;++b) for(int c=0;c<order;++c) {
    long long index=((long long)(base[0]+a)*fy+base[1]+b)*fz+base[2]+c;
    value+=(double)w[a]*(double)w[order+b]*(double)w[2*order+c]*(double)field[index];
  }
  output[12*(long long)target+component]=(real)value;
}
"""
)


_GAUSSIAN_CHANNEL_BODY = (
    r"""  long long lane=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(lane>=volume) return;
  int iz=lane%fz, iy=(lane/fz)%fy, ix=lane/((long long)fy*fz);
  real values[9]={0,0,0,0,0,0,0,0,0};
  if(!((ix>=nx && ix<fx-nx+1)||(iy>=ny && iy<fy-ny+1)||(iz>=nz && iz<fz-nz+1))) {
    int lx=ix<nx?ix:ix-fx, ly=iy<ny?iy:iy-fy, lz=iz<nz?iz:iz-fz;
    const real pi=(real)3.141592653589793238462643383279502884;
    const real pi15=(real)0.17958712212516656168908198362769276;
    for(int image=0;image<images;++image) {
      real r[3]={(real)lx*hx,(real)ly*hy,(real)((long long)lz-shifts[image])*hz};
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
        "    const long long volume, const int nx,const int ny,const int nz,\n"
        "    const int fx,const int fy,const int fz, const real hx,const real hy,const real hz,\n"
        f"    const long long *shifts,const int images,const real tau,{arguments}real *kernel) {{\n"
        + _GAUSSIAN_CHANNEL_BODY
        + publication
        + "}\n"
    )


_FUSED_CUDA = _gaussian_channel_kernel("gaussian_kernel_fused", batch=False)


_STREAM_CUDA = r"""
extern "C" __global__ void scatter_component(
    const long long total, const int order, const int *first, const real *weights,
    const real *gamma, const int fy, const int fz, const int component, real *grid) {
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
  long long index=((long long)(base[0]+a)*fy+base[1]+b)*fz+base[2]+c;
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
    """Owned finite source-only field; NOT an image-tail or solver admission.

    Default image-only mode excludes the primary descriptor (0, False).
    Explicit source_only_primary=True permits a coherent source-only target
    field; it NEVER implements particle pair-averaged-core self induction.
    All evaluate outputs INCLUDE local narrow-minus-broad core correction.
    Sources are immutable private host snapshots with no replacement API.
    Queries must fit the original logical grid; there is no extrapolation.
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
    ):
        self._owner = self._plans = self._correction = None
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
        dimensions = np.ceil(points.max(axis=0) - self.origin) + order + 1
        if (
            not np.isfinite(dimensions).all()
            or np.any(dimensions < 1)
            or np.any(dimensions > 2**20)
        ):
            raise ValueError("auxiliary logical shape exceeds bounds")
        self.shape = tuple(dimensions.astype(np.int64).tolist())
        self.fft_shape = tuple(next_fast_len(2 * n - 1) for n in self.shape)
        self.volume = math.prod(self.fft_shape)
        self.spectrum_shape = (*self.fft_shape[:2], self.fft_shape[2] // 2 + 1)
        self.spectrum_count = math.prod(self.spectrum_shape)
        self.compact_bytes = 12 * math.prod(self.shape) * self.dtype.itemsize
        self.execution_plan = field_execution_plan(
            self.shape,
            self.fft_shape,
            count,
            len(q),
            order,
            self.dtype.itemsize,
            self.max_scratch_bytes,
            self.max_plan_bytes,
        )
        self.estimated_payload_bytes = self.execution_plan.payload_bytes
        started = time.perf_counter()
        try:
            self._owner = DeviceOwner(self.max_scratch_bytes)
            self.cp, self.pool = self._owner.cp, self._owner.pool
            self.stream = self._owner.stream
            free, total = self.cp.cuda.runtime.memGetInfo()
            self.execution_plan, self.effective_smooth_pool_cap = device_field_execution_plan(
                self.shape,
                self.fft_shape,
                count,
                len(q),
                order,
                self.dtype.itemsize,
                self.max_scratch_bytes,
                self.max_plan_bytes,
                self.max_correction_bytes,
                int(free),
            )
            self.estimated_payload_bytes = self.execution_plan.payload_bytes
            self.pool.set_limit(size=self.effective_smooth_pool_cap)
            self.device_admission = {
                "free_device_bytes": int(free),
                "total_device_bytes": int(total),
                "configured_smooth_pool_cap": self.max_scratch_bytes,
                "effective_smooth_pool_cap": self.effective_smooth_pool_cap,
                "correction_reserve_bytes": self.max_correction_bytes,
            }
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
            self._correction = GaussianCoreCorrectionGPU(
                self.host_x,
                self.host_gamma,
                self._host_sigma,
                tau=tau,
                cutoff=cutoff,
                accumulation_dtype=correction_dtype,
                max_scratch_bytes=self.max_correction_bytes,
                max_images=self.max_images,
            )
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
            "tail_certified": False,
        }

    @staticmethod
    def _immutable_snapshot(array):
        # A write-disabled owning ndarray is not immutable: setflags(write=True)
        # can enable mutation and desynchronise cached smooth families from the
        # core-correction source copy. A bytes owner cannot be made writable.
        return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)

    @staticmethod
    def _host_array(value, name):
        if hasattr(value, "__cuda_array_interface__") or np.iscomplexobj(value):
            raise ValueError(f"{name} requires real host coordinates/values")
        array = np.array(value, dtype=np.float64, copy=True, order="C")
        if array.ndim != 2 or array.shape[1:] != (3,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite (N,3)")
        return GaussianImageFields._immutable_snapshot(array)

    def _admit_integer_images(self, integer):
        # The CUDA kernel casts the DIFFERENCE lz-shift, not shift alone.
        # Signed logical convolution lags span [-(nz-1), nz-1]. Check its
        # entire range before the cast so every integer remains representable.
        exact_limit = 2 ** (24 if self.dtype == np.dtype("float32") else 53)
        if any(abs(shift) + self.shape[2] - 1 > exact_limit for shift, _ in integer):
            raise ValueError("image lags exceed exact kernel-coordinate qualification")

    def _admit(self):
        if self.closed or self._owner is None:
            raise RuntimeError("finite Gaussian image owner is closed")
        self._owner.admit()

    def can_evaluate_targets(self, targets):
        """Whether complete target stencils fit this prepared logical field.

        This read-only geometry check allocates no GPU storage, performs no
        extrapolation and certifies no tail/correction or finite-mesh error.
        Session callers must still validate exact sources/role and the new
        query's mathematical bounds. A different initial-query AABB is not
        itself a reason to discard an otherwise valid source-wide field.
        True is geometry admission only, not a promise of query scratch
        capacity; evaluation retains its capped allocation and device guards.
        """
        self._admit()
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
        return _logical_stencils_fit(lattice, self.origin, self.order, self.shape, self.fft_shape)

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
                                    np.int64(self.volume),
                                    *(np.int32(n) for n in self.shape),
                                    *(np.int32(n) for n in self.fft_shape),
                                    *(self.real_type(h) for h in self.steps),
                                    shifts,
                                    np.int32(len(shifts)),
                                    self.real_type(self.tau),
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
                        crop = self._inverse[tuple(slice(0, n) for n in self.shape)]
                        if one_family:
                            self._compact_fields[column] += crop
                        else:
                            cp.copyto(self._compact_fields[column], crop)
                        diagnostics["inverse_transforms"] += 1

    def _all_channel_fields(self, families, diagnostics):
        """Original fused layout, unchanged when its bounded payload fits."""
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
                        np.int64(self.volume),
                        *(np.int32(n) for n in self.shape),
                        *(np.int32(n) for n in self.fft_shape),
                        *(self.real_type(h) for h in self.steps),
                        shifts,
                        np.int32(len(shifts)),
                        self.real_type(self.tau),
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
                crop = tuple(slice(0, n) for n in self.shape)
                self.cp.copyto(self._compact_fields[column], self._inverse[crop])
                diagnostics["inverse_transforms"] += 1

    def prepare(self, images):
        """Prepare compact finite smooth fields; no tail or runtime admission."""
        self._admit()
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
        self._admit_integer_images(integer)
        diagnostics = {
            "runtime_admissible": False,
            "dtype": self.dtype.name,
            "variant": "fused-nine-radial-channels",
            "finite_image_count": len(integer),
            "world_images": world,
            "integer_images": integer,
            "shape": self.shape,
            "fft_shape": self.fft_shape,
            "spacing": self.steps.tolist(),
            "estimated_payload_bytes": self.estimated_payload_bytes,
            "passes": {},
            "inverse_transforms": 0,
            "kernel_forward_transforms": 0,
            "radial_kernel_launches": 0,
            "core_correction_included": False,
        }
        diagnostics["execution_plan"] = vars(self.execution_plan).copy()
        diagnostics["device_admission"] = self.device_admission.copy()
        started = time.perf_counter()
        with self._owner.allocation_scope():
            if self._plans is None:
                self._plans = FFTPlanPair(
                    self._owner,
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
                self._compact_fields = cp.empty((12, *self.shape), dtype=self.dtype)
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
            diagnostics.update(tail_certified=False, compact_bytes=self.compact_bytes)
            return diagnostics

    def evaluate_prepared(self, targets):
        """Fresh bounded query batches, compact gather and exact correction.

        The complete public output is fresh device storage. Query coordinates,
        stencils and correction results live only for the active batch, so a
        later, larger query does not require a larger persistent field owner.
        """
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
        with self._owner.allocation_scope():
            output = self.cp.empty((len(q), 12), dtype=self.dtype)
            gather_seconds, batches, retries, first_target = 0.0, 0, 0, 0
            stencil, correction = None, None
            evaluated_images = set()
            try:
                while first_target < len(q):
                    count = min(batch, len(q) - first_target)
                    last_target = first_target + count
                    chunk = output[first_target:last_target]
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
                                    np.int32(self.shape[1]),
                                    np.int32(self.shape[2]),
                                    self._compact_fields[column],
                                    np.int32(column),
                                    chunk,
                                ),
                            )
                        self.stream.synchronize()
                        current_gather_seconds = time.perf_counter() - gather_started
                        # Stencils and corrections do not need to coexist.
                        first = weight = None
                        correction_u, correction_j, current_correction = self._correction.evaluate(
                            q[first_target:last_target], self._prepared_world_images
                        )
                        chunk[:, :3] += correction_u
                        chunk[:, 3:] += correction_j.reshape(-1, 9)
                        correction_u = correction_j = None
                        self.stream.synchronize()
                        if not bool(self.cp.isfinite(chunk).all()):
                            raise FloatingPointError("nonfinite compact query; nothing published")
                    except MemoryError:
                        if count == 1:
                            raise
                        # A driver allocation or a split cached block can fail
                        # despite scalar admission. Retry this unpublished
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
                            "kernel_and_admission_seconds",
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
                # CUDA/evaluation failures cannot leave a seemingly certified
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
                output[:, :3],
                output[:, 3:].reshape(-1, 3, 3),
                {
                    "runtime_admissible": False,
                    "tail_certified": False,
                    "core_correction_included": True,
                    "source_replaced": False,
                    "target_count": len(q),
                    "finite_image_count": len(self._prepared_images),
                    "compact_bytes": self.compact_bytes,
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
        if self._owner is None:
            self.closed = True
            return
        try:
            self._owner.admit()
            self.stream.synchronize()
        except BaseException as error:
            self._cleanup_failure = error
            self.closed = True
            raise
        self._prepared_images = self._prepared_world_images = None
        self.closed = True
        failure = None
        for resource in (self._plans, self._correction):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    failure = failure or error
        self._plans = self._correction = None
        self._compact_fields = self._kernel_scratch = self._inverse = None
        self._result_spectra = ()
        self.spectra.clear()
        self._source_stencils.clear()
        self._program = None
        try:
            self._owner.close()
        except BaseException as error:
            failure = failure or error
        if failure is not None:
            self._cleanup_failure = failure
            raise failure

    def __enter__(self):
        self._admit()
        return self

    def __exit__(self, *_):
        self.close()
