"""Bounded, single-thread/device/stream CuPy ownership; lazy dependency.

No process-global allocator or FFT-plan-cache settings are changed. Explicit
PlanNd objects avoid get_fft_plan's possible borrowing from the global cache.
This small adapter deliberately supports only dense C-order three-axis FFTs.
CuPy's PlanNd constructor itself may evict its DEFAULT unused memory pool on
an internal cuFFT planning allocation failure; that upstream failure-path
behaviour cannot be disabled through its public plan API. Normal execution,
our allocation failures and teardown never clear an unrelated pool/cache.
CUDA context/JIT driver allocations are outside the private array cap.
"""

from contextlib import contextmanager
import math
from numbers import Integral
import threading
import time

import numpy as np


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class DeviceOwner:
    """Own a capped allocation pool without selecting a device or stream."""

    def __init__(self, max_bytes):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.runtime.getDevice())
        self.stream = cp.cuda.get_current_stream()
        self.owner_thread = threading.get_ident()
        self.closed = False
        self.max_bytes = positive_integer(max_bytes, "max_bytes")
        self.pool = cp.cuda.MemoryPool()
        self.pool.set_limit(size=self.max_bytes)

    def admit(self):
        if self.closed:
            raise RuntimeError("Gaussian mesh owner is closed")
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("Gaussian mesh requires its creating thread")
        if (int(self.cp.cuda.runtime.getDevice()) != self.device_id
                or self.cp.cuda.get_current_stream().ptr != self.stream.ptr):
            raise RuntimeError("Gaussian mesh requires its original device and stream")

    @contextmanager
    def allocation_scope(self):
        self.admit()
        with self.cp.cuda.using_allocator(self.pool.malloc):
            yield

    def close(self):
        if self.closed:
            return
        self.admit()
        self.stream.synchronize()
        self.closed = True
        # Only unused blocks belonging to THIS owner are released. Caller
        # results keep their own MemoryPointers valid after owner teardown.
        self.pool.free_all_blocks()


class FFTPlanPair:
    """Owned FFT directions, simultaneous or streamed, under one work cap."""

    def __init__(self, owner, shape, dtype, max_plan_bytes, *, single_workspace=False):
        owner.admit()
        self.owner = owner
        self.cp = owner.cp
        self.forward = self.inverse = None
        self.closed = False
        self.shape = tuple(positive_integer(n, "FFT dimension") for n in shape)
        if len(self.shape) != 3 or any(n > np.iinfo(np.int32).max for n in self.shape):
            raise ValueError("three bounded FFT dimensions required")
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("float32/float64 FFT required")
        self.complex_dtype = np.dtype("complex64" if self.dtype.itemsize == 4 else "complex128")
        self.spectrum_shape = (*self.shape[:2], self.shape[2]//2+1)
        self.max_plan_bytes = positive_integer(max_plan_bytes, "max_plan_bytes")
        self.work_bytes = 0
        self.single_workspace = bool(single_workspace)
        self._release_idle_workspaces = False
        self.plan_builds = 0
        self.peak_work_bytes = 0
        self.plan_build_seconds = 0.
        self.alignment_copies = 0
        self.alignment_copy_bytes = 0
        # CuPy exposes PlanNd as its supported returned plan type; invoking
        # its constructor, unlike get_fft_plan, never borrows cached plans.
        from cupy.cuda import cufft

        self.cufft = cufft
        try:
            with owner.allocation_scope():
                self._ensure_direction(True)
                if not self.single_workspace:
                    self._ensure_direction(False)
        except BaseException:
            self.close()
            raise

    def _check_work(self):
        self.work_bytes = sum(int(plan.work_area.mem.size) for plan in (self.forward, self.inverse)
                              if plan is not None)
        if self.work_bytes > self.max_plan_bytes:
            raise MemoryError("explicit FFT workspaces exceed plan cap")
        self.peak_work_bytes = max(self.peak_work_bytes, self.work_bytes)

    def _ensure_direction(self, forward):
        if self.closed:
            raise RuntimeError("FFT plans are closed")
        if (self.forward if forward else self.inverse) is not None:
            return
        if self.single_workspace:
            # PlanNd does not expose a public workspace-rebind API. Retire
            # only our opposite plan, after draining our stream, before
            # constructing the replacement under the same private cap.
            self.owner.stream.synchronize()
            self.forward = self.inverse = None
            self.work_bytes = 0
            if self._release_idle_workspaces:
                self.owner.pool.free_all_blocks()
        started = time.perf_counter()
        try:
            if forward:
                kind = self.cufft.CUFFT_R2C if self.dtype.itemsize == 4 else self.cufft.CUFFT_D2Z
                self.forward = self.cufft.PlanNd(self.shape, self.shape, 1, 1,
                    self.spectrum_shape, 1, 1, kind, 1, "C", 2, self.spectrum_shape[2])
            else:
                kind = self.cufft.CUFFT_C2R if self.dtype.itemsize == 4 else self.cufft.CUFFT_Z2D
                self.inverse = self.cufft.PlanNd(self.shape, self.spectrum_shape, 1, 1,
                    self.shape, 1, 1, kind, 1, "C", 2, self.shape[2])
            self.plan_builds += 1
            self._check_work()
        except BaseException:
            self.close()
            raise
        finally:
            self.plan_build_seconds += time.perf_counter()-started

    def measure_directional_work(self):
        """Measure one owned direction at a time for low-memory admission.

        Configured plan bytes remain an upper bound. Retired idle workspaces
        are released only from this owner's pool; unrelated caches are never
        touched. The inverse plan remains live for the selected execution.
        """
        if not self.single_workspace:
            raise ValueError("directional measurement requires a single workspace")
        self._release_idle_workspaces = True
        with self.owner.allocation_scope():
            self._ensure_direction(True)
            forward_bytes = self.work_bytes
            self._ensure_direction(False)
            inverse_bytes = self.work_bytes
        return forward_bytes, inverse_bytes

    def retain_both_directions(self):
        """Retain both already-measured directions after sum-based admission."""
        self.single_workspace = False
        with self.owner.allocation_scope():
            self._ensure_direction(True)

    def _array(self, array, shape, dtype, *, require_complex_alignment=True):
        self.owner.admit()
        if (not isinstance(array, self.cp.ndarray) or array.shape != shape
                or array.dtype != dtype or not array.flags.c_contiguous
                or array.device.id != self.owner.device_id):
            raise ValueError("FFT array shape/dtype/layout/device mismatch")
        # CuFFT R2C/C2R requires complex-element alignment even for REAL
        # pointers. A C-contiguous odd-volume channel slice only guarantees
        # scalar alignment and native cufftExecR2C then returns INVALID_VALUE.
        if require_complex_alignment and array.data.ptr % self.complex_dtype.itemsize:
            raise ValueError("FFT pointer requires complex-element alignment: "
                             f"pointer={array.data.ptr}, alignment={self.complex_dtype.itemsize}")

    def _execution_failed(self, error, *, direction, source, destination):
        error.add_note(f"Owned CuFFT {direction}: shape={self.shape}, dtype={self.dtype}, "
                       f"input_pointer={source.data.ptr}, output_pointer={destination.data.ptr}, "
                       f"complex_alignment={self.complex_dtype.itemsize}, "
                       f"plan_work_bytes={self.work_bytes}, single_workspace={self.single_workspace}")
        try:
            self.close()
        except BaseException as cleanup_error:
            error.add_note(f"FFT cleanup also failed: {cleanup_error!r}")

    def rfft(self, array, *, aligned_scratch=None):
        """Transform real data without allocating an implicit alignment copy.

        An odd-volume channel view may be scalar- but not complex-aligned.
        Its caller must supply a distinct, already-budgeted real workspace.
        Copying on the owned stream changes no bits or transform dimensions.
        The field owner lends its inverse grid after the prior crop is done.
        """
        self._array(array, self.shape, self.dtype, require_complex_alignment=False)
        if aligned_scratch is not None:
            self._array(aligned_scratch, self.shape, self.dtype)
            # Both arrays have already passed exact dense C-order admission.
            # Their byte intervals therefore decide overlap exactly. CuPy's
            # general shares_memory enumerates element addresses on the GPU
            # and can allocate multiple full-grid temporaries here.
            start, scratch_start = array.data.ptr, aligned_scratch.data.ptr
            if start < scratch_start+aligned_scratch.nbytes and scratch_start < start+array.nbytes:
                raise ValueError("FFT alignment scratch must not alias its input")
        with self.owner.allocation_scope():
            if array.data.ptr % self.complex_dtype.itemsize:
                if aligned_scratch is None:
                    raise ValueError("misaligned real FFT input requires explicit aligned scratch: "
                                     f"shape={self.shape}, pointer={array.data.ptr}, "
                                     f"alignment={self.complex_dtype.itemsize}")
                self.cp.copyto(aligned_scratch, array)
                self.alignment_copies += 1
                self.alignment_copy_bytes += array.nbytes
                array = aligned_scratch
            self._ensure_direction(True)
            result = self.cp.empty(self.spectrum_shape, dtype=self.complex_dtype)
            try:
                self.forward.fft(array, result, self.cufft.CUFFT_FORWARD)
            except BaseException as error:
                self._execution_failed(error, direction="R2C", source=array, destination=result)
                raise
            return result

    def irfft(self, array, output):
        self._array(array, self.spectrum_shape, self.complex_dtype)
        self._array(output, self.shape, self.dtype)
        with self.owner.allocation_scope():
            self._ensure_direction(False)
            try:
                self.inverse.fft(array, output, self.cufft.CUFFT_INVERSE)
            except BaseException as error:
                self._execution_failed(error, direction="C2R", source=array, destination=output)
                raise
            output /= math.prod(self.shape)
        return output

    def close(self):
        self.closed = True
        if self.forward is None and self.inverse is None:
            return
        self.owner.admit()
        self.owner.stream.synchronize()
        self.forward = self.inverse = None
        self.work_bytes = 0
