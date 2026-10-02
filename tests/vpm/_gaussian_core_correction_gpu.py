"""UNWIRED bounded CuPy local source-core Gaussian image correction.

This is a qualification prototype, not an admitted production backend. It
computes a specified FINITE image list and omits pairs at r >= cutoff. The
analytic omitted-correction bound is separate from mesh/image-tail/roundoff
certification. Physical mean-core particle induction is NOT part of this API.

The source cell index is built once and queried at inverse-image target
positions. No pair list is stored: a warp visits the cutoff-overlapping cells and
reduces its own twelve output components. Odd images reflect source position
and axial strength, not the velocity or Jacobian after accumulation. Every
published output belongs to a complete successful query.

For dA=A_sigma-A_tau and dB=B_sigma-B_tau,
    u = dA (Gamma cross r)
    J = dA [Gamma]_cross - dB (Gamma cross r) outer r.
At the origin u=0 but J=[Gamma]_cross*(sigma^-3-tau^-3)/(3*pi^1.5).
An origin power series and a near-equal-core positive quadrature avoid the
two cancellation-prone subtractions in the ordinary erf representation.
The independent oracle is _gaussian_broadening_reference.correction_fields.

All arrays/sort temporaries allocated by this object use a PRIVATE capped
CuPy memory pool. Context/module/RawKernel compilation allocations are not
pool-accounted and must additionally be bounded by the external GPU runner.
Pool total_bytes is the retained allocation high-water, not a CUDA-driver
total-memory certificate. No installs, runtime selection, or solver changes.
"""

from contextlib import contextmanager
import math
from numbers import Integral
import time

import numpy as np

_QUAD_X, _QUAD_W = np.polynomial.legendre.leggauss(8)
_PI15 = math.pi ** -1.5


def correction_factors_host(radius, sigma, tau):
    """f64 translation of the device branches; independent oracle is integral."""
    radius, sigma, tau = map(float, (radius, sigma, tau))
    if not all(math.isfinite(v) for v in (radius, sigma, tau)) or radius < 0 or not 0 < sigma <= tau:
        raise ValueError("finite r>=0 and 0<sigma<=tau required")
    if sigma == tau:
        return 0., 0.
    rho = radius / sigma
    log_ratio = math.log1p((tau-sigma)/sigma)
    if rho < .5:
        term, a, b = 1., 0., 0.
        for n in range(18):
            a += term * -math.expm1(-(2*n+3)*log_ratio)/(2*n+3)
            b += term * -math.expm1(-(2*n+5)*log_ratio)/(2*n+5)
            term *= -rho*rho/(n+1)
        return _PI15*a/sigma**3, 2*_PI15*b/sigma**5
    if log_ratio < .01:
        lower = 1/tau
        half = (tau-sigma)/(2*sigma*tau)
        nodes = lower + half*(_QUAD_X+1)
        weight = half*_QUAD_W*np.exp(-(radius*nodes)**2)
        return _PI15*float(weight @ nodes**2), 2*_PI15*float(weight @ nodes**4)
    wide = radius/tau
    e_sigma, e_tau = math.exp(-rho*rho), math.exp(-wide*wide)
    t_sigma = (math.erfc(rho)+2/math.sqrt(math.pi)*rho*e_sigma)/(4*math.pi)
    t_tau = (math.erfc(wide)+2/math.sqrt(math.pi)*wide*e_tau)/(4*math.pi)
    a = (t_tau-t_sigma)/radius**3
    z = _PI15*(e_sigma/sigma**3-e_tau/tau**3)
    return a, (3*a-z)/radius**2


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _image_list(images, max_images):
    result = []
    for shift, odd in images:
        if len(result) >= max_images:
            raise ValueError("finite image-count cap exceeded")
        if type(odd) not in (bool, np.bool_) or not math.isfinite(float(shift)):
            raise ValueError("images require finite world shift and boolean parity")
        result.append((float(shift), bool(odd)))
    return result


def _possibly_near(source_min, source_max, target_min, target_max, shift, odd, cutoff):
    """Conservative host AABB exclusion only; exact strict cutoff is on device."""
    lo, hi = np.array(source_min, copy=True), np.array(source_max, copy=True)
    if odd:
        lo[2], hi[2] = shift-hi[2], shift-lo[2]
    else:
        lo[2] += shift
        hi[2] += shift
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("transformed source bounds are not representable")
    scale = max(1., float(np.max(np.abs(np.concatenate((lo, hi, target_min, target_max))))))
    margin = 64*np.finfo(float).eps*scale
    distance = np.maximum(np.maximum(lo-target_max, target_min-hi)-margin, 0.)
    # A single safely separated axis suffices; not squaring avoids overflow.
    return bool(np.max(distance) <= cutoff)


def _kernel_source(dtype):
    real = "double" if dtype == "float64" else "float"
    suffix = "" if dtype == "float64" else "f"
    quad_x = ",".join(format(float(v), ".18e") for v in _QUAD_X)
    quad_w = ",".join(format(float(v), ".18e") for v in _QUAD_W)
    return r'''
    typedef REAL Real;
    __device__ inline Real rr_exp(Real x) { return expSUFFIX(x); }
    __device__ inline Real rr_expm1(Real x) { return expm1SUFFIX(x); }
    __device__ inline Real rr_erfc(Real x) { return erfcSUFFIX(x); }
    __device__ inline Real rr_log1p(Real x) { return log1pSUFFIX(x); }
    __device__ inline Real rr_sqrt(Real x) { return sqrtSUFFIX(x); }
    __device__ inline void factors(Real r, Real sigma, Real tau, Real &a, Real &b) {
        const Real pi15 = (Real)0.179587122125166561689;
        const Real inv4pi = (Real)0.0795774715459476678844;
        const Real two_sqrtpi = (Real)1.12837916709551257390;
        a=0; b=0;
        if (sigma == tau) return;
        const Real invsigma=1/sigma, rho=r*invsigma;
        const Real ratio=sigma/tau;
        const Real logratio=ratio>(Real).99?rr_log1p((tau-sigma)*invsigma):(Real)1;
        if (rho < (Real).5) {
            Real term=1, aa=0, bb=0;
            Real ratio_power=ratio*ratio*ratio,ratio2=ratio*ratio;
            #pragma unroll
            for (int n=0;n<18;n++) {
                Real da=ratio>(Real).99?-rr_expm1(-(2*n+3)*logratio):1-ratio_power;
                Real db=ratio>(Real).99?-rr_expm1(-(2*n+5)*logratio):1-ratio_power*ratio2;
                aa += term*da/(2*n+3);
                bb += term*db/(2*n+5);
                term *= -rho*rho/(n+1);
                ratio_power*=ratio2;
            }
            const Real inv3=invsigma*invsigma*invsigma;
            a=pi15*aa*inv3;
            b=2*pi15*bb*inv3*invsigma*invsigma;
        } else if (logratio < (Real).01) {
            const Real nodes[8]={QUAD_X};
            const Real weights[8]={QUAD_W};
            const Real lower=1/tau, half=(tau-sigma)/(2*sigma*tau);
            Real aa=0,bb=0;
            #pragma unroll
            for (int n=0;n<8;n++) {
                Real s=lower+half*(nodes[n]+1), s2=s*s;
                Real w=half*weights[n]*rr_exp(-r*r*s2);
                aa += w*s2; bb += w*s2*s2;
            }
            a=pi15*aa; b=2*pi15*bb;
        } else {
            Real wide=r/tau, es=rr_exp(-rho*rho), et=rr_exp(-wide*wide);
            Real ts=(rr_erfc(rho)+two_sqrtpi*rho*es)*inv4pi;
            Real tt=(rr_erfc(wide)+two_sqrtpi*wide*et)*inv4pi;
            Real invr=1/r, invr2=invr*invr;
            a=(tt-ts)*invr*invr2;
            Real z=pi15*(es*invsigma*invsigma*invsigma-et/(tau*tau*tau));
            b=(3*a-z)*invr2;
        }
    }
    extern "C" __global__ void build_keys(const double* x, int n,
        double ox,double oy,double oz,double width,int nx,int ny,int nz,
        int* key,int* error) {
        int i=blockDim.x*blockIdx.x+threadIdx.x;
        if (i>=n) return;
        long long ix=(long long)floor((x[3*i]-ox)/width);
        long long iy=(long long)floor((x[3*i+1]-oy)/width);
        long long iz=(long long)floor((x[3*i+2]-oz)/width);
        if(ix<0||iy<0||iz<0||ix>=nx||iy>=ny||iz>=nz) {atomicExch(error,1);key[i]=0;return;}
        key[i]=(int)((ix*ny+iy)*nz+iz);
    }
    extern "C" __global__ void cell_ranges(const int* sorted_key,int n,int* begin,int* end) {
        int i=blockDim.x*blockIdx.x+threadIdx.x;
        if (i>=n) return;
        int key=sorted_key[i];
        if(i==0||sorted_key[i-1]!=key) begin[key]=i;
        if(i==n-1||sorted_key[i+1]!=key) end[key]=i+1;
    }
    extern "C" __global__ void correction(const double* x,const double* gamma,
        const double* sigma,const double* target,int nt,const int* begin,const int* end,
        double ox,double oy,double oz,double width,int nx,int ny,int nz,
        double shift,int odd,double tau_double,double cutoff2,
        Real* velocity,Real* jacobian,unsigned long long* work,int* error) {
        int lane=threadIdx.x&31;
        int t=(blockDim.x*blockIdx.x+threadIdx.x)>>5;
        if(t>=nt)return;
        double qx=target[3*t],qy=target[3*t+1],qz=target[3*t+2];
        double inverse_z=odd?shift-qz:qz-shift;
        // Skip far cells before integer conversion; arbitrary remote queries
        // cannot overflow int64 cell indices or read outside the source grid.
        if(qx<ox-width||qy<oy-width||inverse_z<oz-width||
           qx>ox+(nx+1.)*width||qy>oy+(ny+1.)*width||inverse_z>oz+(nz+1.)*width)return;
        double scale=fmax(1.,fmax(fabs(shift),fmax(fabs(qx),fmax(fabs(qy),fabs(qz)))));
        scale=fmax(scale,fmax(fabs(ox),fmax(fabs(oy),fabs(oz))));
        double margin=64*2.22044604925031308085e-16*scale;
        if(!isfinite(inverse_z)||margin>width/(double)8) {if(lane==0)atomicExch(error,3);return;}
        long long ix0=max(0LL,(long long)floor((qx-width-margin-ox)/width));
        long long iy0=max(0LL,(long long)floor((qy-width-margin-oy)/width));
        long long iz0=max(0LL,(long long)floor((inverse_z-width-margin-oz)/width));
        long long ix1=min((long long)nx-1,(long long)floor((qx+width+margin-ox)/width));
        long long iy1=min((long long)ny-1,(long long)floor((qy+width+margin-oy)/width));
        long long iz1=min((long long)nz-1,(long long)floor((inverse_z+width+margin-oz)/width));
        Real out[12]={0,0,0,0,0,0,0,0,0,0,0,0};
        unsigned long long candidate=0,accepted=0;
        for(long long cx=ix0;cx<=ix1;cx++) for(long long cy=iy0;cy<=iy1;cy++) for(long long cz=iz0;cz<=iz1;cz++) {
            int cell=(int)((cx*ny+cy)*nz+cz);
            for(int s=begin[cell]+lane;s<end[cell];s+=32) {
                candidate++;
                double rx=qx-x[3*s],ry=qy-x[3*s+1];
                double rz=qz-(odd?shift-x[3*s+2]:shift+x[3*s+2]);
                double r2=rx*rx+ry*ry+rz*rz;
                if(r2>=cutoff2)continue;
                accepted++;
                Real r[3]={(Real)rx,(Real)ry,(Real)rz};
                Real gx=(Real)(odd?-gamma[3*s]:gamma[3*s]);
                Real gy=(Real)(odd?-gamma[3*s+1]:gamma[3*s+1]);
                Real gz=(Real)gamma[3*s+2];
                Real cross[3]={gy*r[2]-gz*r[1],gz*r[0]-gx*r[2],gx*r[1]-gy*r[0]};
                Real c[9]={0,-gz,gy,gz,0,-gx,-gy,gx,0};
                Real a,b;
                factors((Real)sqrt(r2),(Real)sigma[s],(Real)tau_double,a,b);
                #pragma unroll
                for(int row=0;row<3;row++) {
                    out[row]+=a*cross[row];
                    #pragma unroll
                    for(int col=0;col<3;col++) out[3+3*row+col]+=a*c[3*row+col]-b*cross[row]*r[col];
                }
            }
        }
        #pragma unroll
        for(int distance=16;distance>0;distance/=2) {
            #pragma unroll
            for(int c=0;c<12;c++)out[c]+=__shfl_down_sync(0xffffffff,out[c],distance);
            candidate+=__shfl_down_sync(0xffffffff,candidate,distance);
            accepted+=__shfl_down_sync(0xffffffff,accepted,distance);
        }
        if(lane==0) {
            #pragma unroll
            for(int c=0;c<12;c++)if(!isfinite(out[c]))atomicExch(error,2);
            #pragma unroll
            for(int c=0;c<3;c++)velocity[3*t+c]+=out[c];
            #pragma unroll
            for(int c=0;c<9;c++)jacobian[9*t+c]+=out[3+c];
            atomicAdd(work,candidate);atomicAdd(work+1,accepted);
        }
    }
    '''.replace("REAL", real).replace("SUFFIX", suffix).replace("QUAD_X", quad_x).replace("QUAD_W", quad_w)


class GaussianCoreCorrectionGPU:
    """Reusable, private, capped source index; no solver/runtime ownership."""

    def __init__(self, source_x, source_gamma, source_sigma, *, tau, cutoff,
                 max_scratch_bytes=256*1024**2, accumulation_dtype="float64", max_images=513):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.runtime.getDevice())
        self.stream = cp.cuda.get_current_stream()
        self.stream_pointer = int(self.stream.ptr)
        self.closed = False
        self.max_scratch_bytes = _positive_integer(max_scratch_bytes, "max_scratch_bytes")
        self.max_images = _positive_integer(max_images, "max_images")
        self.tau, self.cutoff = float(tau), float(cutoff)
        if not all(math.isfinite(v) and v>0 for v in (self.tau, self.cutoff)):
            raise ValueError("finite positive tau and cutoff required")
        if not math.isfinite(self.cutoff*self.cutoff):
            raise ValueError("squared cutoff must be representable")
        if accumulation_dtype not in ("float32", "float64"):
            raise ValueError("accumulation_dtype must be float32 or float64")
        self.dtype = np.dtype(accumulation_dtype)
        self.pool = cp.cuda.MemoryPool()
        self.pool.set_limit(size=self.max_scratch_bytes)
        self._owned = []
        self.diagnostics = {"production_qualified": False, "source_only_cores": True,
                            "dtype": accumulation_dtype, "cutoff_strict": True,
                            "pair_storage": False, "max_scratch_bytes": self.max_scratch_bytes,
                            "build_seconds": 0., "pool_high_water_bytes": 0}
        try:
            start = time.perf_counter()
            with cp.cuda.using_allocator(self.pool.malloc):
                x = cp.array(source_x, dtype=cp.float64, order="C", copy=True)
                gamma = cp.array(source_gamma, dtype=cp.float64, order="C", copy=True)
                sigma = cp.array(source_sigma, dtype=cp.float64, order="C", copy=True)
                if x.ndim != 2 or x.shape[1:] != (3,) or gamma.shape != x.shape or sigma.shape != x.shape[:1]:
                    raise ValueError("require source (N,3), strength (N,3), core (N,)")
                self.count = len(x)
                if not 1 <= self.count <= np.iinfo(np.int32).max:
                    raise ValueError("nonempty int32-indexed sources required")
                if not bool((cp.isfinite(x).all() & cp.isfinite(gamma).all() & cp.isfinite(sigma).all()
                             & (sigma>0).all() & (sigma<=self.tau).all()).item()):
                    raise ValueError("finite arrays and 0<source core<=tau required")
                lower_sigma = float(sigma.min().item())
                finfo = np.finfo(self.dtype)
                if (lower_sigma < finfo.tiny or self.tau > finfo.max or
                        not math.isfinite(lower_sigma**-5) or lower_sigma**-5 > finfo.max/16):
                    raise ValueError("core-factor scale unsupported by selected arithmetic dtype")
                self.source_min = cp.asnumpy(x.min(axis=0))
                self.source_max = cp.asnumpy(x.max(axis=0))
                extent = (self.source_max-self.source_min)/self.cutoff
                if not np.isfinite(extent).all() or np.any(extent > np.iinfo(np.int32).max-2):
                    raise ValueError("source grid extent exceeds index range")
                self.shape = tuple((np.floor(extent).astype(np.int64)+1).tolist())
                cells = math.prod(self.shape)
                if cells > np.iinfo(np.int32).max or 8*cells+56*self.count > self.max_scratch_bytes:
                    raise MemoryError("bounded source-cell storage exceeds configured cap")
                source = _kernel_source(accumulation_dtype)
                self._keys = cp.RawKernel(source, "build_keys", options=("--std=c++11",))
                self._ranges = cp.RawKernel(source, "cell_ranges", options=("--std=c++11",))
                self._query = cp.RawKernel(source, "correction", options=("--std=c++11",))
                keys = cp.empty(self.count, dtype=cp.int32)
                error = cp.zeros(1, dtype=cp.int32)
                self._keys(((self.count+255)//256,), (256,),
                           (x, np.int32(self.count), *map(np.float64, self.source_min), np.float64(self.cutoff),
                            *map(np.int32, self.shape), keys, error))
                if int(error.item()):
                    raise RuntimeError("source-cell key construction failed")
                order = cp.argsort(keys)
                self.x, self.gamma, self.sigma = x[order], gamma[order], sigma[order]
                sorted_keys = keys[order]
                self.begin, self.end = cp.zeros(cells, dtype=cp.int32), cp.zeros(cells, dtype=cp.int32)
                self._ranges(((self.count+255)//256,), (256,),
                             (sorted_keys, np.int32(self.count), self.begin, self.end))
                self._owned = [self.x, self.gamma, self.sigma, self.begin, self.end]
                cp.cuda.get_current_stream().synchronize()
                self.diagnostics.update(source_count=self.count, cell_shape=self.shape,
                                        build_seconds=time.perf_counter()-start,
                                        pool_high_water_bytes=self.pool.total_bytes())
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _allocation_scope(self):
        if self.closed:
            raise RuntimeError("correction owner is closed")
        if int(self.cp.cuda.runtime.getDevice()) != self.device_id or int(self.cp.cuda.get_current_stream().ptr) != self.stream_pointer:
            raise RuntimeError("correction qualification requires the original device and CUDA stream")
        with self.cp.cuda.using_allocator(self.pool.malloc):
            yield

    def evaluate(self, target_x, images):
        """Complete finite-image correction; strict cutoff, private publication."""
        cp = self.cp
        images = _image_list(images, self.max_images)
        start = time.perf_counter()
        with self._allocation_scope():
            target = cp.array(target_x, dtype=cp.float64, order="C", copy=True)
            if target.ndim != 2 or target.shape[1:] != (3,) or len(target) > np.iinfo(np.int32).max:
                raise ValueError("require int32-indexed target (M,3)")
            if not bool(cp.isfinite(target).all().item()):
                raise ValueError("finite targets required")
            count = len(target)
            u = cp.zeros((count, 3), dtype=self.dtype)
            j = cp.zeros((count, 3, 3), dtype=self.dtype)
            work = cp.zeros(2, dtype=cp.uint64)
            error = cp.zeros(1, dtype=cp.int32)
            near_images, skipped_images = [], []
            if count:
                lo, hi = cp.asnumpy(target.min(axis=0)), cp.asnumpy(target.max(axis=0))
                for shift, odd in images:
                    if _possibly_near(self.source_min, self.source_max, lo, hi, shift, odd, self.cutoff):
                        near_images.append((shift, odd))
                    else:
                        skipped_images.append((shift, odd))
                for shift, odd in near_images:
                    self._query(((count*32+127)//128,), (128,),
                                (self.x, self.gamma, self.sigma, target, np.int32(count), self.begin, self.end,
                                 *map(np.float64, self.source_min), np.float64(self.cutoff), *map(np.int32, self.shape),
                                 np.float64(shift), np.int32(odd), np.float64(self.tau), np.float64(self.cutoff**2),
                                 u, j, work, error))
                cp.cuda.get_current_stream().synchronize()
                if int(error.item()) or not bool((cp.isfinite(u).all() & cp.isfinite(j).all()).item()):
                    raise FloatingPointError("nonfinite Gaussian correction; no fields published")
            candidate, accepted = map(int, cp.asnumpy(work))
            self.diagnostics["pool_high_water_bytes"] = max(
                self.diagnostics["pool_high_water_bytes"], self.pool.total_bytes())
            report = {**self.diagnostics, "target_count": count, "images": images,
                      "evaluated_images": near_images, "aabb_skipped_images": skipped_images,
                      "candidate_pairs": candidate, "accepted_pairs": accepted,
                      "query_seconds": time.perf_counter()-start,
                      "pool_used_bytes": self.pool.used_bytes(), "pool_reserved_bytes": self.pool.total_bytes()}
            return u, j, report

    def close(self):
        if getattr(self, "closed", True):
            return
        self.closed = True
        # No implicit cross-stream ownership protocol: drain the recorded
        # stream before releasing our allocation references on every path.
        if hasattr(self, "stream"):
            self.stream.synchronize()
        self._owned = []
        for name in ("x", "gamma", "sigma", "begin", "end"):
            setattr(self, name, None)
        if hasattr(self, "pool"):
            self.pool.free_all_blocks()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("correction owner is closed")
        return self

    def __exit__(self, *_):
        self.close()


def gaussian_core_correction(source_x, source_gamma, source_sigma, target_x, *, tau, cutoff, images,
                             max_scratch_bytes=256*1024**2, accumulation_dtype="float64", max_images=513):
    with GaussianCoreCorrectionGPU(source_x, source_gamma, source_sigma, tau=tau, cutoff=cutoff,
                                   max_scratch_bytes=max_scratch_bytes, accumulation_dtype=accumulation_dtype,
                                   max_images=max_images) as owner:
        return owner.evaluate(target_x, images)
