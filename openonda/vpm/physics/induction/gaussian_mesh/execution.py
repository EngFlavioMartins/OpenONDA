"""Portable execution of one unchanged finite Gaussian field operator.

CUDA allocation failures subdivide the same field into smaller FFT jobs before
host execution is needed. No CUDA execution/cleanup fault is treated as memory
pressure, and nothing is published until evaluation succeeds.
"""

from collections.abc import Sequence
import logging

import numpy as np


class PortableGaussianImageFields:
    """Image field that falls back from CUDA to CUDA blocks to the host on memory pressure."""

    def __init__(self, *args, execution_backend: str = "cpu", **kwargs):
        self._implementation = self._failed_implementation = None
        self._args, self._kwargs = args, kwargs
        self._images = None
        self.closed = False
        self.fallback_reason = None
        self.subdivision_reason = None
        self._using_cuda_blocks = False
        self.execution_backend = execution_backend
        self._build_implementation(execution_backend)

    def _construct(self, backend: str) -> object | None:
        if backend == "cupy_cuda":
            from .fields import GaussianImageFields as Implementation
        elif backend == "cupy_cuda_blocked":
            from .blocked_fields import GaussianBlockedCUDAFields as Implementation
        else:
            from .host_fields import GaussianHostImageFields as Implementation
        implementation = Implementation.__new__(Implementation)
        try:
            implementation.__init__(*self._args, **self._kwargs)
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

    def _recover_memory(self, error: BaseException) -> str | None:
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

    def __getattr__(self, name: str) -> object:
        implementation = self.__dict__.get("_implementation")
        if implementation is None:
            raise AttributeError(name)
        return getattr(implementation, name)

    def prepare(self, images: Sequence[tuple[int, bool]]) -> dict[str, object]:
        """Prepare ``images`` on the current backend, falling back on memory pressure."""
        self._images = tuple(images)
        while True:
            try:
                return self._implementation.prepare(self._images)
            except MemoryError as error:
                backend = self._recover_memory(error)
            self._build_implementation(backend)

    def evaluate_prepared(
        self, targets: np.ndarray
    ) -> (
        tuple[np.ndarray, np.ndarray, dict[str, str]]
        | tuple[str, str, dict[str, int]]
        | tuple[str, str, dict[str, str]]
    ):
        """Evaluate the prepared field at ``targets`` with the same fallback."""
        while True:
            try:
                velocity, gradient, diagnostics = self._implementation.evaluate_prepared(targets)
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
