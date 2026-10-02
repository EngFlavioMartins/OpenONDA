"""UNWIRED fused-kernel qualification variant of the finite slab field mesh.

The supplied finite images, radial formulas, source spectra, interpolation and
core convention are unchanged. Each grid node evaluates its Gaussian radial
factors once per image, accumulating three gradient and six symmetric Hessian
channels. Hessian symmetry reuses one multiplication order; it is not a claim of
bitwise identity to the original independently evaluated transposed channels.

Nine real grids and twelve result spectra are private, bounded scratch. Channels
are transformed sequentially and discarded; source families are processed one
at a time, then combined before the same twelve inverse transforms. Nothing is
published if allocation, evaluation or finite checks fail. This is not a
production backend and does not change or waive the image-tail controller.
"""

import time

import numpy as np

from tests.vpm._cupy_slab_field_mesh import _CUDA, SlabFieldMeshGPU
from tests.vpm._finite_slab_field_mesh_reference import slab_world_images

_FUSED_CUDA = r'''
extern "C" __global__ void gaussian_kernel_fused(
    const long long volume, const int nx,const int ny,const int nz,
    const int fx,const int fy,const int fz, const real hx,const real hy,const real hz,
    const long long *shifts,const int images,const real tau,real *kernel) {
  long long lane=(long long)blockIdx.x*blockDim.x+threadIdx.x;
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
      values[0]+=-a*r[0]; values[1]+=-a*r[1]; values[2]+=-a*r[2];
      values[3]+=b*r[0]*r[0]-a;
      values[4]+=b*r[0]*r[1]; values[5]+=b*r[0]*r[2];
      values[6]+=b*r[1]*r[1]-a;
      values[7]+=b*r[1]*r[2]; values[8]+=b*r[2]*r[2]-a;
    }
  }
  for(int channel=0;channel<9;++channel) kernel[channel*volume+lane]=values[channel];
}
'''


# Each pair is (gradient axis, differentiation axis); -1 denotes velocity.
_CHANNEL_COMPONENTS = (((0, -1),), ((1, -1),), ((2, -1),),
                       ((0, 0),), ((0, 1), (1, 0)), ((0, 2), (2, 0)),
                       ((1, 1),), ((1, 2), (2, 1)), ((2, 2),))


def fused_payload_bytes(base_payload, spectrum_count, volume, itemsize):
    """Conservative admission before any larger fused scratch is allocated.

Relative to the base's12 complex+4 real grids, reserve20 complex+10 real:
six cached source spectra, twelve result spectra, one kernel spectrum and
one additional FFT temporary; nine kernel grids and one inverse field.
Plan memory is charged separately by the owner. The enforced pool remains
the independent hard bound, including allocator fragmentation and temporaries.
"""
    return int(base_payload+(16*spectrum_count+6*volume)*itemsize)


def channel_routes(channel):
    """Map a symmetric potential derivative to curl output products."""
    routes = []
    for axis, derivative in _CHANNEL_COMPONENTS[channel]:
        for component, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
            if axis in (a, b):
                source = b if axis == a else a
                sign = 1 if axis == a else -1
                column = component if derivative < 0 else 3+3*component+derivative
                routes.append((column, source, sign))
    return tuple(routes)


class SlabFieldMeshFusedGPU(SlabFieldMeshGPU):
    """Same private lifecycle and inputs; finite-block fused radial experiment."""

    def __init__(self, *args, **kwargs):
        self._kernel_scratch = None
        self._result_spectra = ()
        super().__init__(*args, **kwargs)
        try:
            self.estimated_payload_bytes = fused_payload_bytes(
                self.estimated_payload_bytes, self.spectrum_count, self.volume, self.dtype.itemsize)
            if self.estimated_payload_bytes > self.max_scratch_bytes-self.max_plan_bytes:
                raise MemoryError("estimated fused smooth-operator payload exceeds private pool admission")
        except BaseException:
            self.close()
            raise

    def _compile_program(self):
        float32 = self.dtype == np.dtype("float32")
        code = (_CUDA+_FUSED_CUDA).replace("REAL", "float" if float32 else "double")
        for name in ("SQRT", "EXP", "ERF"):
            code = code.replace(name, name.lower()+("f" if float32 else ""))
        return self.cp.RawModule(code=code, options=("--std=c++11",), backend="nvrtc")

    def evaluate(self, images):
        """Return privately computed smooth fields; no core or tail admission."""
        self._admit()
        cp = self.cp
        world, integer = slab_world_images(images, self.zmin, self.zmax, self.cells, self.max_images)
        if not integer or any(shift == 0 and not odd for shift, odd in integer):
            raise ValueError("nonempty image-only block required; physical primary excluded")
        if self.dtype == np.dtype("float32") and any(abs(shift) > 2**24 for shift, _ in integer):
            raise ValueError("image indices exceed exact float32 kernel-coordinate qualification")
        diagnostics = {"runtime_admissible": False, "dtype": self.dtype.name,
                       "variant": "fused-nine-radial-channels", "finite_image_count": len(integer),
                       "world_images": world, "integer_images": integer, "shape": self.shape,
                       "fft_shape": self.fft_shape, "spacing": self.steps.tolist(),
                       "estimated_payload_bytes": self.estimated_payload_bytes, "passes": {},
                       "inverse_transforms": 12, "kernel_forward_transforms": 0,
                       "radial_kernel_launches": 0, "core_correction_included": False}
        started = time.perf_counter()
        with cp.cuda.using_allocator(self.pool.malloc):
            families = {}
            for odd in (False, True):
                shifts = [shift for shift, reflection in integer if reflection == odd]
                if shifts:
                    self._prepare_family(odd, diagnostics)
                    families[odd] = cp.asarray(shifts, dtype=cp.int64)
            # Keep the large allocations intact across calls. Releasing this
            # nine-grid block allowed subsequent smaller FFT allocations to
            # split it, eventually exceeding the unchanged private pool cap.
            # These buffers are never returned to callers; output owns its
            # allocation and remains valid after later evaluations.
            if self._kernel_scratch is None:
                self._kernel_scratch = cp.empty((9, *self.fft_shape), dtype=self.dtype)
            if not self._result_spectra:
                self._result_spectra = tuple(cp.empty(self.spectrum_shape, dtype=self.complex_type)
                                             for _ in range(12))
            output = cp.empty((len(self.host_targets), 12), dtype=self.dtype)
            result_hat = self._result_spectra
            for spectrum in result_hat:
                spectrum.fill(0)
            for odd, shifts in families.items():
                with self._phase(diagnostics, "fused_kernel_build"):
                    kernels = self._kernel_scratch
                    self._launch("gaussian_kernel_fused", self.volume, (np.int64(self.volume),
                        *(np.int32(n) for n in self.shape), *(np.int32(n) for n in self.fft_shape),
                        *(self.real_type(h) for h in self.steps), shifts, np.int32(len(shifts)),
                        self.real_type(self.tau), kernels))
                    diagnostics["radial_kernel_launches"] += 1
                for channel in range(9):
                    with self._phase(diagnostics, "kernel_fft"):
                        kernel_hat = cp.fft.rfftn(kernels[channel])
                        diagnostics["kernel_forward_transforms"] += 1
                    with self._phase(diagnostics, "spectral_product"):
                        for column, source, sign in channel_routes(channel):
                            self._launch("product_add", self.spectrum_count,
                                (np.int64(self.spectrum_count), kernel_hat, self.spectra[odd][source],
                                 self.real_type(sign), result_hat[column]))
                        del kernel_hat
            for column in range(12):
                with self._phase(diagnostics, "inverse_fft_and_gather"):
                    field = cp.fft.irfftn(result_hat[column], s=self.fft_shape)
                    self._launch("gather", len(self.host_targets), (np.int32(len(self.host_targets)),
                        np.int32(self.order), self._first, self._weight, np.int32(self.fft_shape[1]),
                        np.int32(self.fft_shape[2]), field, np.int32(column), output))
                    del field
            self.stream.synchronize()
            if not bool(cp.isfinite(output).all()):
                raise FloatingPointError("nonfinite fused smooth GPU output; nothing published")
            free, total = cp.cuda.runtime.memGetInfo()
            diagnostics.update(seconds=time.perf_counter()-started, pool_used_bytes=self.pool.used_bytes(),
                               pool_reserved_bytes=self.pool.total_bytes(), plan_bytes=self.cache.get_curr_memsize(),
                               device_free_bytes=free, device_total_bytes=total)
            return output[:, :3], output[:, 3:].reshape(-1, 3, 3), diagnostics

    def close(self):
        if getattr(self, "closed", True):
            return
        # Let the base enforce ownership before dropping any buffers.
        self._admit()
        self.stream.synchronize()
        self._kernel_scratch = None
        self._result_spectra = ()
        super().close()
