"""Portable execution of one unchanged finite Gaussian field operator.

CUDA allocation failures may retire this owner's private resources and rebuild
the same field with bounded host FFT blocks. No CUDA execution/cleanup fault is
treated as memory pressure, and nothing is published until evaluation succeeds.
"""


class PortableGaussianImageFields:
    def __init__(self, *args, execution_backend="cpu", **kwargs):
        self._owner = self._failed_owner = None
        self._args, self._kwargs = args, kwargs
        self._images = None
        self.closed = False
        self.fallback_reason = None
        self.execution_backend = execution_backend
        try:
            self._owner = self._construct(execution_backend)
        except MemoryError as error:
            if execution_backend != "cupy_cuda":
                raise
            self._switch_to_host(error)

    def _construct(self, backend):
        if backend == "cupy_cuda":
            from .fields import GaussianImageFields as Implementation
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

    def _switch_to_host(self, error):
        if self._failed_owner is not None:
            raise error
        owner, self._owner = self._owner, None
        if owner is not None:
            try:
                owner.close()
            except BaseException as cleanup_error:
                self._failed_owner = owner
                error.add_note(f"Finite field fallback cleanup failed: {cleanup_error!r}")
                raise error from cleanup_error
        self.execution_backend = "cpu"
        self.fallback_reason = str(error)
        self._owner = self._construct("cpu")
        if self._images is not None:
            self._owner.prepare(self._images)

    def __getattr__(self, name):
        owner = self.__dict__.get("_owner")
        if owner is None:
            raise AttributeError(name)
        return getattr(owner, name)

    def prepare(self, images):
        self._images = tuple(images)
        try:
            result = self._owner.prepare(self._images)
        except MemoryError as error:
            if self.execution_backend != "cupy_cuda":
                raise
            self._switch_to_host(error)
            result = {"execution_backend": "cpu", "memory_fallback": self.fallback_reason}
        return result

    def evaluate_prepared(self, targets):
        try:
            velocity, gradient, diagnostics = self._owner.evaluate_prepared(targets)
        except MemoryError as error:
            if self.execution_backend != "cupy_cuda":
                raise
            self._switch_to_host(error)
            velocity, gradient, diagnostics = self._owner.evaluate_prepared(targets)
        diagnostics["execution_backend"] = self.execution_backend
        diagnostics["memory_fallback"] = self.fallback_reason
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
