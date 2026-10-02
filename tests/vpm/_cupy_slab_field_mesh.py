"""Optional, UNWIRED CuPy smooth finite-image field-mesh experiment.

No production imports or device-pointer interop. Host transfers, scatter,
kernel generation, FFTs and gather are timed. Integer slab images preserve
the supplied finite sum; physical primary (0,False) is rejected. The local
source-core correction is owned by a separate experimental operator.

Only three output spectra live at once. For each derivative column, even
and odd family contributions are added BEFORE the inverse transforms:
IFFT(sum_f F(K_f)*F(Gamma_f)) = sum_f K_f*Gamma_f by linearity. Floating
summation changes are qualified against the separate float64 reference.
"""

from contextlib import contextmanager
import math
import threading
import time

import numpy as np
from scipy.fft import next_fast_len

from tests.vpm._finite_image_mesh_reference import _positive_cap, _validate_stencil
from tests.vpm._finite_slab_field_mesh_reference import slab_coordinates, slab_world_images

_CUDA = r'''
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
    if(rho2<(real)1) {
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
    value+=derivative<0 ? -a*r[axis] : b*r[axis]*r[derivative]-(axis==derivative?a:0);
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
'''


def _weights(points, origin, order, shape):
    """Stable product-form cardinal weights; no high-degree power cancellation."""
    coordinate = points-origin
    if not np.isfinite(coordinate).all() or np.any(np.abs(coordinate) > 2**30):
        raise ValueError("unresolved or excessive auxiliary-grid coordinates")
    first = np.floor(coordinate).astype(np.int64)-order//2+1
    _validate_stencil(first, order, shape)
    argument = coordinate-first
    weights = np.ones((len(points), 3, order), dtype=np.float64)
    for i in range(order):
        for j in range(order):
            if i != j:
                weights[:, :, i] *= (argument-j)/(i-j)
    if not np.isfinite(weights).all():
        raise FloatingPointError("nonfinite assignment weights")
    return np.ascontiguousarray(first, dtype=np.int32), weights


class SlabFieldMeshGPU:
    """Private bounded GPU ownership for one immutable source/target scope.

    Requires an initially empty current-thread CuPy FFT cache; this isolated
    qualification process must not share ownership with another FFT client.
    Returned device arrays are owned by the caller until explicitly released.
    """

    def __init__(self, source_x, source_gamma, targets, *, zmin, zmax, tau=.12,
                 spacing=.035, order=10, dtype="float32", max_scratch_bytes=2*1024**3,
                 max_plan_bytes=128*1024**2, max_images=513, stencil_backend="cpu"):
        started = time.perf_counter()
        import cupy as cp

        self.cp, self.closed = cp, False
        self.device_id = cp.cuda.runtime.getDevice()
        self.stream = cp.cuda.get_current_stream()
        self.owner_thread = threading.get_ident()
        self.max_scratch_bytes = _positive_cap(max_scratch_bytes, "max_scratch_bytes")
        self.max_plan_bytes = _positive_cap(max_plan_bytes, "max_plan_bytes")
        self.max_images = _positive_cap(max_images, "max_images")
        if dtype not in ("float32", "float64") or order not in (4, 6, 8, 10):
            raise ValueError("qualified experiment supports float32/float64 and even order4..10")
        if stencil_backend not in ("cpu", "gpu"):
            raise ValueError("stencil_backend must be cpu or gpu")
        self.stencil_backend = stencil_backend
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("positive finite broadening required")
        if any(isinstance(value, cp.ndarray) for value in (source_x, source_gamma, targets)):
            raise TypeError("explicit host inputs required so transfers remain visible in qualification")
        self.host_x = np.array(source_x, dtype=np.float64, copy=True, order="C")
        self.host_gamma = np.array(source_gamma, dtype=np.float64, copy=True, order="C")
        self.host_targets = np.array(targets, dtype=np.float64, copy=True, order="C")
        if self.host_gamma.shape != self.host_x.shape or not np.isfinite(self.host_gamma).all():
            raise ValueError("finite source strength vectors required")
        if not len(self.host_x) or not len(self.host_targets):
            raise ValueError("nonempty scope required for this bounded GPU experiment")
        self.zmin, self.zmax, self.tau, self.order = zmin, zmax, tau, order
        self.dtype = np.dtype(dtype)
        self.real_type = np.float32 if dtype == "float32" else np.float64
        self.complex_type = cp.complex64 if dtype == "float32" else cp.complex128
        x, self.steps, self.cells = slab_coordinates(self.host_x, zmin, zmax, spacing)
        q, _, _ = slab_coordinates(self.host_targets, zmin, zmax, spacing)
        reflected = x.copy()
        reflected[:, 2] *= -1
        points = np.concatenate((x, reflected, q))
        self.origin = np.floor(points.min(axis=0))-order
        dimensions = np.ceil(points.max(axis=0)-self.origin)+order+1
        if not np.isfinite(dimensions).all() or np.any(dimensions < 1) or np.any(dimensions > 2**20):
            raise ValueError("invalid bounded auxiliary shape")
        self.shape = tuple(dimensions.astype(np.int64).tolist())
        self.fft_shape = tuple(next_fast_len(2*n-1) for n in self.shape)
        self.volume = math.prod(self.fft_shape)
        self.spectrum_shape = (*self.fft_shape[:2], self.fft_shape[2]//2+1)
        self.spectrum_count = math.prod(self.spectrum_shape)
        # Includes conservative FFT temporaries, three result spectra and six
        # cached source spectra. The independent pool limit remains enforced.
        self.estimated_payload_bytes = (12*self.spectrum_count*2+4*self.volume)*self.dtype.itemsize
        self.estimated_payload_bytes += (2*len(x)+len(q))*3*order*self.dtype.itemsize
        self.estimated_payload_bytes += (len(x)+len(q))*3*8+len(q)*12*self.dtype.itemsize
        if self.stencil_backend == "gpu":
            # Three retained index sets plus the largest transient normalized
            # f64 coordinate upload. The pool independently enforces its cap.
            self.estimated_payload_bytes += (2*len(x)+len(q))*12+max(len(x), len(q))*24
        if self.estimated_payload_bytes > self.max_scratch_bytes-self.max_plan_bytes:
            raise MemoryError("estimated bounded smooth-operator payload exceeds private pool admission")
        self._host_stencils, self._device_stencils = {}, {}
        self._target_stencil = None
        if self.stencil_backend == "cpu":
            self._host_stencils = {False: _weights(x, self.origin, order, self.shape),
                                   True: _weights(reflected, self.origin, order, self.shape)}
            self._target_stencil = _weights(q, self.origin, order, self.shape)
        for array in (self.host_x, self.host_gamma, self.host_targets):
            array.setflags(write=False)
        self.pool = cp.cuda.MemoryPool()
        self.pool.set_limit(size=self.max_scratch_bytes)
        self.cache = cp.fft.config.get_plan_cache()
        if self.cache.get_curr_size():
            raise RuntimeError("isolated prototype requires initially empty per-thread FFT cache")
        self.old_cache_limits = self.cache.get_size(), self.cache.get_memsize()
        self.spectra = {}
        self._first = self._weight = self.source_x = self.source_gamma = self.target_x = None
        self.initial_diagnostics = {"host_geometry_seconds": time.perf_counter()-started,
                                    "stencil_backend": self.stencil_backend, "passes": {}}
        self.program = None
        try:
            self.cache.set_size(2)
            self.cache.set_memsize(self.max_plan_bytes)
            self.program = self._compile_program()
            with self._phase(self.initial_diagnostics, "host_to_device"):
                self.source_x = cp.asarray(self.host_x)
                self.source_gamma = cp.asarray(self.host_gamma)
                self.target_x = cp.asarray(self.host_targets)
            with self._phase(self.initial_diagnostics, "stencil_prepare"):
                self._prepare_stencils(x, reflected, q)
            self.initial_diagnostics["construction_seconds"] = time.perf_counter()-started
        except BaseException:
            self.close()
            raise

    def _prepare_stencils(self, source, reflected, target):
        """Build private assignment geometry without changing coordinate maps."""
        if self.stencil_backend == "cpu":
            first, weights = self._target_stencil
            self._first = self.cp.asarray(first)
            self._weight = self.cp.asarray(weights, dtype=self.dtype)
            return
        from tests.vpm._cupy_cardinal_stencil import cardinal_stencil_gpu

        records = {}
        self.initial_diagnostics["gpu_stencil_builds"] = records
        for odd, label, points in ((False, "source_even", source), (True, "source_odd", reflected)):
            first, weights, records[label] = cardinal_stencil_gpu(
                points, self.origin, self.order, self.shape, dtype=self.dtype, pool=self.pool,
                max_points=max(len(source), len(target)), max_new_bytes=self.max_scratch_bytes)
            self._device_stencils[odd] = first, weights
        self._first, self._weight, records["target"] = cardinal_stencil_gpu(
            target, self.origin, self.order, self.shape, dtype=self.dtype, pool=self.pool,
            max_points=max(len(source), len(target)), max_new_bytes=self.max_scratch_bytes)

    def _family_stencil(self, odd):
        if self.stencil_backend == "gpu":
            return self._device_stencils[odd]
        first, weights = self._host_stencils[odd]
        return self.cp.asarray(first), self.cp.asarray(weights, dtype=self.dtype)

    def _compile_program(self):
        float32 = self.dtype == np.dtype("float32")
        code = _CUDA.replace("REAL", "float" if float32 else "double")
        for name in ("SQRT", "EXP", "ERF"):
            code = code.replace(name, name.lower()+("f" if float32 else ""))
        return self.cp.RawModule(code=code, options=("--std=c++11",), backend="nvrtc")

    def _admit(self):
        if self.closed:
            raise RuntimeError("closed smooth operator")
        if (threading.get_ident() != self.owner_thread or self.cp.cuda.runtime.getDevice() != self.device_id
                or self.cp.cuda.get_current_stream().ptr != self.stream.ptr):
            raise RuntimeError("smooth scope requires its original device, thread and stream")

    @contextmanager
    def _phase(self, diagnostics, name):
        self._admit()
        with self.cp.cuda.using_allocator(self.pool.malloc):
            self.stream.synchronize()
            started = time.perf_counter()
            try:
                yield
            finally:
                self.stream.synchronize()
                diagnostics.setdefault("passes", {}).setdefault(name, 0.)
                diagnostics["passes"][name] += time.perf_counter()-started

    def _launch(self, name, count, arguments):
        self.program.get_function(name)(((count+255)//256,), (256,), arguments)

    def _prepare_family(self, odd, diagnostics):
        if odd in self.spectra:
            return
        cp = self.cp
        with self._phase(diagnostics, "source_scatter_and_fft"):
            first_d, weights_d = self._family_stencil(odd)
            strengths = self.host_gamma.copy()
            if odd:
                strengths[:, :2] *= -1
            gamma_d = cp.asarray(strengths, dtype=self.dtype)
            grid = cp.zeros((3, *self.fft_shape), dtype=self.dtype)
            total = len(first_d)*self.order**3
            self._launch("scatter", total, (np.int64(total), np.int32(self.order), first_d, weights_d,
                         gamma_d, np.int32(self.fft_shape[1]), np.int32(self.fft_shape[2]),
                         np.int64(self.volume), grid))
            self.spectra[odd] = tuple(cp.fft.rfftn(grid[c]) for c in range(3))
            del first_d, weights_d, gamma_d, grid
            # The immutable density spectra now own all source-family work;
            # retaining assignment geometry after this point serves no use.
            self._device_stencils.pop(odd, None)

    def evaluate(self, images):
        """Return private smooth u/J arrays and complete block execution evidence."""
        self._admit()
        cp = self.cp
        world, integer = slab_world_images(images, self.zmin, self.zmax, self.cells, self.max_images)
        if not integer or any(shift == 0 and not odd for shift, odd in integer):
            raise ValueError("nonempty image-only block required; physical primary excluded")
        if self.dtype == np.dtype("float32") and any(abs(shift) > 2**24 for shift, _ in integer):
            raise ValueError("image indices exceed exact float32 kernel-coordinate qualification")
        diagnostics = {"runtime_admissible": False, "dtype": self.dtype.name,
                       "stencil_backend": self.stencil_backend,
                       "finite_image_count": len(integer), "world_images": world, "integer_images": integer,
                       "shape": self.shape, "fft_shape": self.fft_shape, "spacing": self.steps.tolist(),
                       "estimated_payload_bytes": self.estimated_payload_bytes, "passes": {},
                       "inverse_transforms": 12, "kernel_forward_transforms": 0,
                       "core_correction_included": False}
        started = time.perf_counter()
        with cp.cuda.using_allocator(self.pool.malloc):
            families = {}
            for odd in (False, True):
                shifts = [shift for shift, reflection in integer if reflection == odd]
                if shifts:
                    self._prepare_family(odd, diagnostics)
                    families[odd] = cp.asarray(shifts, dtype=cp.int64)
            output = cp.empty((len(self.host_targets), 12), dtype=self.dtype)
            pairs = ((1, 2), (2, 0), (0, 1))
            for derivative in (-1, 0, 1, 2):
                result_hat = [cp.zeros(self.spectrum_shape, dtype=self.complex_type) for _ in range(3)]
                for odd, shifts in families.items():
                    for axis in range(3):
                        with self._phase(diagnostics, "kernel_build_and_fft"):
                            kernel = cp.empty(self.fft_shape, dtype=self.dtype)
                            self._launch("gaussian_kernel", self.volume, (np.int64(self.volume),
                                *(np.int32(n) for n in self.shape), *(np.int32(n) for n in self.fft_shape),
                                *(self.real_type(h) for h in self.steps), shifts, np.int32(len(shifts)),
                                self.real_type(self.tau), np.int32(axis), np.int32(derivative), kernel))
                            kernel_hat = cp.fft.rfftn(kernel)
                            diagnostics["kernel_forward_transforms"] += 1
                            del kernel
                        with self._phase(diagnostics, "spectral_product"):
                            for component, (a, b) in enumerate(pairs):
                                if axis in (a, b):
                                    source = b if axis == a else a
                                    sign = 1 if axis == a else -1
                                    self._launch("product_add", self.spectrum_count,
                                        (np.int64(self.spectrum_count), kernel_hat, self.spectra[odd][source],
                                         self.real_type(sign), result_hat[component]))
                            del kernel_hat
                for component in range(3):
                    with self._phase(diagnostics, "inverse_fft_and_gather"):
                        field = cp.fft.irfftn(result_hat[component], s=self.fft_shape)
                        column = component if derivative < 0 else 3+3*component+derivative
                        self._launch("gather", len(self.host_targets), (np.int32(len(self.host_targets)),
                            np.int32(self.order), self._first, self._weight, np.int32(self.fft_shape[1]),
                            np.int32(self.fft_shape[2]), field, np.int32(column), output))
                        del field
                del result_hat
            self.stream.synchronize()
            if not bool(cp.isfinite(output).all()):
                raise FloatingPointError("nonfinite smooth GPU output; nothing published")
            free, total = cp.cuda.runtime.memGetInfo()
            diagnostics.update(seconds=time.perf_counter()-started, pool_used_bytes=self.pool.used_bytes(),
                               pool_reserved_bytes=self.pool.total_bytes(), plan_bytes=self.cache.get_curr_memsize(),
                               device_free_bytes=free, device_total_bytes=total)
            return output[:, :3], output[:, 3:].reshape(-1, 3, 3), diagnostics

    def close(self):
        if getattr(self, "closed", True):
            return
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("FFT scope must close on its creating thread")
        with self.cp.cuda.Device(self.device_id), self.stream:
            self.stream.synchronize()
            self.closed = True
            if hasattr(self, "spectra"):
                self.spectra.clear()
            if hasattr(self, "_device_stencils"):
                self._device_stencils.clear()
            if hasattr(self, "_host_stencils"):
                self._host_stencils.clear()
            self._target_stencil = None
            self._first = self._weight = self.source_x = self.source_gamma = self.target_x = None
            self.program = None
            if hasattr(self, "old_cache_limits"):
                self.cache.clear()
                self.cache.set_size(self.old_cache_limits[0])
                self.cache.set_memsize(self.old_cache_limits[1])
            if hasattr(self, "pool"):
                self.pool.free_all_blocks()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
