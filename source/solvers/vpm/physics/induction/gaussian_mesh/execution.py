"""Portable execution of one unchanged finite Gaussian field operator.

CUDA allocation failures subdivide the same field into smaller FFT jobs before
host execution is needed. No CUDA execution/cleanup fault is treated as memory
pressure, and nothing is published until evaluation succeeds.
"""

import logging


class PortableGaussianImageFields:
    def __init__(self, *args, execution_backend="cpu", **kwargs):
        self._owner = self._failed_owner = None
        self._args, self._kwargs = args, kwargs
        self._images = None
        self.closed = False
        self.fallback_reason = None
        self.subdivision_reason = None
        self._using_cuda_blocks = False
        self.execution_backend = execution_backend
        self._build_owner(execution_backend)

    def _construct(self, backend):
        if backend == "cupy_cuda":
            from .fields import GaussianImageFields as Implementation
        elif backend == "cupy_cuda_blocked":
            from .blocked_fields import GaussianBlockedCUDAFields as Implementation
        else:
            from .host_fields import GaussianHostImageFields as Implementation
        owner = Implementation.__new__(Implementation)
        try:
            owner.__init__(*self._args, **self._kwargs)
        except BaseException as error:
            try:
                owner.close()
            except BaseException as cleanup_error:
                self._failed_owner = owner
                error.add_note(f"Finite field construction cleanup failed: {cleanup_error!r}")
                raise error from cleanup_error
            raise
        return owner

    def _build_owner(self, backend):
        while True:
            try:
                self._owner = self._construct(backend)
                return
            except MemoryError as error:
                backend = self._recover_memory(error)
            # Leave the exception scope before allocating again: its traceback
            # can otherwise retain temporary arrays from the failed FFT job.

    def _recover_memory(self, error):
        if self.execution_backend != "cupy_cuda" or self._failed_owner is not None:
            raise error
        owner, self._owner = self._owner, None
        if owner is not None:
            try:
                owner.close()
            except BaseException as cleanup_error:
                self._failed_owner = owner
                error.add_note(f"Finite field fallback cleanup failed: {cleanup_error!r}")
                raise error from cleanup_error
        if not self._using_cuda_blocks:
            self._using_cuda_blocks = True
            self.subdivision_reason = str(error)
            return "cupy_cuda_blocked"
        self.execution_backend = "cpu"
        self.fallback_reason = str(error)
        logging.getLogger("vpm").warning(
            "Gaussian slab field cannot fit a CUDA block; evaluating on CPU: %s",
            self.fallback_reason,
        )
        return "cpu"

    def __getattr__(self, name):
        owner = self.__dict__.get("_owner")
        if owner is None:
            raise AttributeError(name)
        return getattr(owner, name)

    def prepare(self, images):
        self._images = tuple(images)
        while True:
            try:
                return self._owner.prepare(self._images)
            except MemoryError as error:
                backend = self._recover_memory(error)
            self._build_owner(backend)

    def evaluate_prepared(self, targets):
        while True:
            try:
                velocity, gradient, diagnostics = self._owner.evaluate_prepared(targets)
                break
            except MemoryError as error:
                backend = self._recover_memory(error)
            self._build_owner(backend)
            if self._images is not None:
                self.prepare(self._images)
        diagnostics["execution_backend"] = self.execution_backend
        diagnostics["memory_fallback"] = self.fallback_reason
        diagnostics["memory_subdivision"] = self.subdivision_reason
        return velocity, gradient, diagnostics

    def close(self):
        if self._failed_owner is not None:
            raise RuntimeError("Finite Gaussian field cleanup remains uncertain")
        if self.closed:
            return
        if self._owner is not None:
            self._owner.close()
            self._owner = None
        self.closed = True
