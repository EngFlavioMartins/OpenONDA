"""UNWIRED coherent source-only PRIMARY + finite slab-image experiment.

This separate opt-in owner does not relax either preserved image-only guard.
It inserts the physical primary exactly once into the same smooth convolution
and source-core correction as its images. It is an arbitrary-target operator,
NOT a particle stage: pair-mean particle cores and self-stage contracts are not
implemented. No tail decision, freestream, or production backend is selected.
"""

from itertools import islice
import time

import numpy as np

from tests.vpm._cupy_slab_compact_fields import CompactSlabFieldMeshGPU
from tests.vpm._cupy_slab_field_mesh_fused import channel_routes
from tests.vpm._finite_slab_field_mesh_reference import slab_world_images


class CoherentSourceTargetsGPU(CompactSlabFieldMeshGPU):
    """Explicit immutable-source, common-grid primary/reflection qualification.

    ``prepare`` accepts image-only descriptors and inserts primary (0,False).
    The explicit constructor opt-in is required even though this class name
    describes its semantics. ``max_images`` counts all terms, primary included.
    Returned fields already contain the source-only physical core correction.
    """

    def __init__(self, *args, include_physical_primary=False, **kwargs):
        if include_physical_primary is not True:
            raise ValueError("explicit include_physical_primary=True source-only opt-in required")
        kwargs.setdefault("max_images", 514)
        super().__init__(*args, **kwargs)

    def _whole_descriptors(self, images):
        # Consume at most the remaining image allowance plus one. The primary
        # must never be caller-supplied or duplicated, even with explicit opt-in.
        descriptors = tuple(islice(iter(images), self.max_images))
        if len(descriptors) >= self.max_images:
            raise ValueError("image count leaves no capacity for the physical primary")
        slab_world_images(descriptors, self.zmin, self.zmax, self.cells, self.max_images-1)
        if any(k == 0 and not odd for k, odd in descriptors):
            raise ValueError("prepare accepts image-only descriptors; primary is inserted exactly once")
        return ((0, False), *descriptors)

    def prepare(self, images):
        """Publish only a fully prepared coherent finite field; failure revokes."""
        self._admit()
        self._prepared_images = self._prepared_world_images = None
        self._prepare_diagnostics = None
        descriptors = self._whole_descriptors(images)
        world, _ = slab_world_images(descriptors, self.zmin, self.zmax,
                                     self.cells, self.max_images)
        started = time.perf_counter()
        with self.cp.cuda.using_allocator(self.pool.malloc):
            if self._compact_fields is None:
                self._compact_fields = self.cp.empty((12, *self.shape), dtype=self.dtype)
            self._capture_columns = set()
            try:
                velocity, gradient, diagnostics = self._coherent_smooth(descriptors)
                del velocity, gradient
                self.stream.synchronize()
                if self._capture_columns != set(range(12)):
                    raise RuntimeError("incomplete coherent compact-grid capture")
                if not bool(self.cp.isfinite(self._compact_fields).all()):
                    raise FloatingPointError("nonfinite coherent compact field; cache not published")
                self._prepared_images = descriptors
                self._prepared_world_images = tuple(world)
                self._prepare_diagnostics = {
                    "runtime_admissible": False, "tail_certified": False,
                    "primary_included": True, "primary_core_contract": "source-only",
                    "particle_stage_supported": False,
                    "finite_term_count": len(descriptors), "image_count": len(descriptors)-1,
                    "compact_bytes": self.compact_bytes,
                    "logical_shape": self.shape, "fft_shape": self.fft_shape,
                    "preparation_seconds": time.perf_counter()-started, "smooth": diagnostics,
                }
                return dict(self._prepare_diagnostics)
            finally:
                self._capture_columns = None

    def evaluate_prepared(self, targets):
        velocity, gradient, diagnostics = super().evaluate_prepared(targets)
        diagnostics.update(primary_included=True, primary_core_contract="source-only",
                           particle_stage_supported=False,
                           finite_term_count=diagnostics.pop("finite_image_count"))
        diagnostics["image_count"] = diagnostics["finite_term_count"]-1
        return velocity, gradient, diagnostics

    def _coherent_smooth(self, descriptors):
        """Same qualified fused arithmetic, separately admitting one primary.

        Deliberately explicit here rather than monkey-patching the preserved
        image-only operator's descriptor validation. The common source spectra,
        signed finite kernels, twelve inverse transforms and common interpolation
        are identical; only the explicit finite term set includes the primary.
        """
        self._admit()
        cp = self.cp
        world, integer = slab_world_images(descriptors, self.zmin, self.zmax,
                                           self.cells, self.max_images)
        if sum(shift == 0 and not odd for shift, odd in integer) != 1:
            raise ValueError("coherent source-only field requires exactly one primary")
        if self.dtype == np.dtype("float32") and any(abs(shift) > 2**24 for shift, _ in integer):
            raise ValueError("image indices exceed exact float32 kernel-coordinate qualification")
        diagnostics = {"runtime_admissible": False, "dtype": self.dtype.name,
                       "variant": "coherent-source-only-fused-nine-radial-channels",
                       "finite_term_count": len(integer), "primary_included": True,
                       "world_images": world, "integer_images": integer,
                       "shape": self.shape, "fft_shape": self.fft_shape,
                       "spacing": self.steps.tolist(),
                       "estimated_payload_bytes": self.estimated_payload_bytes,
                       "passes": {}, "inverse_transforms": 12,
                       "kernel_forward_transforms": 0, "radial_kernel_launches": 0,
                       "core_correction_included": False}
        started = time.perf_counter()
        with cp.cuda.using_allocator(self.pool.malloc):
            families = {}
            for odd in (False, True):
                shifts = [shift for shift, reflection in integer if reflection == odd]
                if shifts:
                    self._prepare_family(odd, diagnostics)
                    families[odd] = cp.asarray(shifts, dtype=cp.int64)
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
                raise FloatingPointError("nonfinite coherent smooth GPU output; nothing published")
            free, total = cp.cuda.runtime.memGetInfo()
            diagnostics.update(seconds=time.perf_counter()-started, pool_used_bytes=self.pool.used_bytes(),
                               pool_reserved_bytes=self.pool.total_bytes(), plan_bytes=self.cache.get_curr_memsize(),
                               device_free_bytes=free, device_total_bytes=total)
            return output[:, :3], output[:, 3:].reshape(-1, 3, 3), diagnostics
