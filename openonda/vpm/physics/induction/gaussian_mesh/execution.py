"""Portable execution of one unchanged finite Gaussian field operator.

CUDA allocation failures subdivide the same field into smaller FFT jobs before
host execution is needed. No CUDA execution/cleanup fault is treated as memory
pressure, and nothing is published until evaluation succeeds.
"""

from collections.abc import Callable, Sequence
from functools import partial
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .runtime import Diagnostics

if TYPE_CHECKING:
    from .blocked_fields import GaussianBlockedCUDAFields
    from .fields import GaussianImageFields
    from .host_fields import GaussianHostImageFields

    ImageFields = GaussianImageFields | GaussianBlockedCUDAFields | GaussianHostImageFields


class PortableGaussianImageFields:
    """Image field that falls back from CUDA to CUDA blocks to the host on memory pressure."""

    def __init__(self, *args, execution_backend: str = "cpu", **kwargs):
        self._implementation: ImageFields | None = None
        self._failed_implementation: ImageFields | None = None
        self._args, self._kwargs = args, kwargs
        self._images: tuple[tuple[int, bool], ...] | None = None
        self.closed = False
        self.fallback_reason: str | None = None
        self.subdivision_reason: str | None = None
        self._using_cuda_blocks = False
        self.execution_backend = execution_backend
        self._build_implementation(execution_backend)

    def _construct(self, backend: str) -> "ImageFields":
        implementation: ImageFields
        initialize: Callable[[], None]
        if backend == "cupy_cuda":
            from .fields import GaussianImageFields

            implementation = GaussianImageFields.__new__(GaussianImageFields)

            initialize = partial(
                GaussianImageFields.__init__, implementation, *self._args, **self._kwargs
            )
        elif backend == "cupy_cuda_blocked":
            from .blocked_fields import GaussianBlockedCUDAFields

            implementation = GaussianBlockedCUDAFields.__new__(GaussianBlockedCUDAFields)

            initialize = partial(
                GaussianBlockedCUDAFields.__init__, implementation, *self._args, **self._kwargs
            )
        else:
            from .host_fields import GaussianHostImageFields

            implementation = GaussianHostImageFields.__new__(GaussianHostImageFields)

            initialize = partial(
                GaussianHostImageFields.__init__, implementation, *self._args, **self._kwargs
            )
        try:
            initialize()
        except BaseException as error:
            try:
                implementation.close()
            except BaseException as cleanup_error:
                self._failed_implementation = implementation
                error.add_note(f"Finite field construction cleanup failed: {cleanup_error!r}")
                raise error from cleanup_error
            raise
        return implementation

    def _build_implementation(self, backend: str) -> None:
        while True:
            try:
                self._implementation = self._construct(backend)
                return
            except MemoryError as error:
                backend = self._recover_memory(error)
            # Leave the exception scope before allocating again: its traceback
            # can otherwise retain temporary arrays from the failed FFT job.

    def _recover_memory(self, error: BaseException) -> str:
        if self.execution_backend != "cupy_cuda" or self._failed_implementation is not None:
            raise error
        implementation, self._implementation = self._implementation, None
        if implementation is not None:
            try:
                implementation.close()
            except BaseException as cleanup_error:
                self._failed_implementation = implementation
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

    @property
    def _active(self) -> "ImageFields":
        if self._implementation is None:
            raise RuntimeError("Finite Gaussian field has no active implementation")
        return self._implementation

    def __getattr__(self, name: str) -> Any:
        # Delegates the remaining field attributes to the active implementation.
        implementation = self.__dict__.get("_implementation")
        if implementation is None:
            raise AttributeError(name)
        return getattr(implementation, name)

    def prepare(self, images: Sequence[tuple[int, bool]]) -> Diagnostics:
        """Prepare ``images`` on the current backend, falling back on memory pressure."""
        self._images = tuple(images)
        while True:
            try:
                return self._active.prepare(self._images)
            except MemoryError as error:
                backend = self._recover_memory(error)
            self._build_implementation(backend)

    def evaluate_prepared(self, targets: np.ndarray) -> tuple[Any, Any, Diagnostics]:
        """Evaluate the prepared field at ``targets`` with the same fallback."""
        while True:
            try:
                velocity, gradient, diagnostics = self._active.evaluate_prepared(targets)
                break
            except MemoryError as error:
                backend = self._recover_memory(error)
            self._build_implementation(backend)
            if self._images is not None:
                self.prepare(self._images)
        diagnostics["execution_backend"] = self.execution_backend
        diagnostics["memory_fallback"] = self.fallback_reason
        diagnostics["memory_subdivision"] = self.subdivision_reason
        return velocity, gradient, diagnostics

    def close(self) -> None:
        """Close the active implementation; idempotent."""
        if self._failed_implementation is not None:
            raise RuntimeError("Finite Gaussian field cleanup remains uncertain")
        if self.closed:
            return
        if self._implementation is not None:
            self._implementation.close()
            self._implementation = None
        self.closed = True
